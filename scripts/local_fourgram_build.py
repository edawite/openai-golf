#!/usr/bin/env python3
"""Build a sparse train-only 4-gram top-continuation expert.

The expert maps a three-token SP1024 context to its most frequent training
continuation. Target positions inside the reserved calibration window are
excluded so that window stays a genuine train-only tuning holdout, and the
validation shard is never read by this script.
"""

from __future__ import annotations

import argparse
import io
import json
import lzma
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SHARD_HEADER_INTS = 256
SHARD_HEADER_BYTES = SHARD_HEADER_INTS * np.dtype("<i4").itemsize
VOCAB_SIZE = 1024
CONTEXT_TOKENS = 3
CALIBRATION_START = 20_000_000
CALIBRATION_STOP = 20_008_192
MAGIC = b"P4G2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count exact 4-grams over the canonical training shard and store the "
            "highest-support contexts as a compressed sidecar."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024",
    )
    parser.add_argument("--train-shard", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs" / "fourgram_expert.ptz",
    )
    parser.add_argument(
        "--minimum-support",
        type=int,
        default=2,
        help="Discard contexts observed fewer times than this.",
    )
    parser.add_argument(
        "--minimum-confidence",
        type=float,
        default=0.5,
        help="Discard contexts whose top continuation share is below this.",
    )
    parser.add_argument(
        "--maximum-contexts",
        type=int,
        default=1_200_000,
        help="Keep at most this many contexts, preferring the best supported.",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=10_000_000,
        help="Encoding chunk size used to limit peak memory.",
    )
    return parser.parse_args()


def shard_memmap(path: Path) -> np.memmap:
    header = np.fromfile(path, dtype="<i4", count=SHARD_HEADER_INTS)
    if (
        header.size != SHARD_HEADER_INTS
        or int(header[0]) != 20240520
        or int(header[1]) != 1
    ):
        raise ValueError(f"Unexpected shard header for {path}")
    token_count = int(header[2])
    expected = SHARD_HEADER_BYTES + token_count * np.dtype("<u2").itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"Shard size mismatch for {path}: expected {expected}")
    return np.memmap(
        path,
        dtype="<u2",
        mode="r",
        offset=SHARD_HEADER_BYTES,
        shape=(token_count,),
    )


def find_train_shard(args: argparse.Namespace) -> Path:
    if args.train_shard is not None:
        if not args.train_shard.is_file():
            raise FileNotFoundError(f"Training shard not found: {args.train_shard}")
        return args.train_shard
    shards = sorted(args.data_path.glob("fineweb_train_*.bin"))
    if not shards:
        raise FileNotFoundError(f"No training shard found in {args.data_path}")
    return shards[0]


def encode_range(
    destination: np.ndarray,
    cursor: int,
    tokens: np.memmap,
    target_start: int,
    target_stop: int,
    chunk_tokens: int,
) -> int:
    for start in range(target_start, target_stop, chunk_tokens):
        stop = min(start + chunk_tokens, target_stop)
        third = np.asarray(tokens[start - 3 : stop - 3], dtype=np.uint64)
        second = np.asarray(tokens[start - 2 : stop - 2], dtype=np.uint64)
        first = np.asarray(tokens[start - 1 : stop - 1], dtype=np.uint64)
        target = np.asarray(tokens[start:stop], dtype=np.uint64)
        if (
            target.size
            and max(
                int(third.max()),
                int(second.max()),
                int(first.max()),
                int(target.max()),
            )
            >= VOCAB_SIZE
        ):
            raise ValueError("Training shard contains a token outside SP1024")
        count = stop - start
        destination[cursor : cursor + count] = (
            (third << np.uint64(30))
            | (second << np.uint64(20))
            | (first << np.uint64(10))
            | target
        )
        cursor += count
    return cursor


def build_expert(
    tokens: np.memmap, chunk_tokens: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    if CALIBRATION_STOP >= tokens.size:
        raise ValueError("Calibration window extends beyond the training shard")
    excluded = CALIBRATION_STOP - CALIBRATION_START + 1
    total = tokens.size - CONTEXT_TOKENS
    encoded = np.empty(total - excluded, dtype=np.uint64)
    cursor = encode_range(
        encoded, 0, tokens, CONTEXT_TOKENS, CALIBRATION_START, chunk_tokens
    )
    cursor = encode_range(
        encoded, cursor, tokens, CALIBRATION_STOP + 1, tokens.size, chunk_tokens
    )
    if cursor != encoded.size:
        raise RuntimeError(f"encoded {cursor} records, allocated {encoded.size}")

    encoded.sort(kind="quicksort")
    boundaries = np.flatnonzero(encoded[1:] != encoded[:-1]) + 1
    run_starts = np.empty(boundaries.size + 1, dtype=np.int64)
    run_starts[0] = 0
    run_starts[1:] = boundaries
    unique_keys = encoded[run_starts].copy()
    run_stops = np.empty_like(run_starts)
    run_stops[:-1] = run_starts[1:]
    run_stops[-1] = encoded.size
    run_counts = (run_stops - run_starts).astype(np.uint32)
    del encoded, boundaries, run_starts, run_stops

    contexts = (unique_keys >> np.uint64(10)).astype(np.uint32)
    continuations = (unique_keys & np.uint64(1023)).astype(np.uint16)
    del unique_keys

    group_boundaries = np.flatnonzero(contexts[1:] != contexts[:-1]) + 1
    group_starts = np.empty(group_boundaries.size + 1, dtype=np.int64)
    group_starts[0] = 0
    group_starts[1:] = group_boundaries
    unique_contexts = contexts[group_starts].copy()
    support = np.add.reduceat(run_counts.astype(np.uint64), group_starts).astype(
        np.uint32
    )
    # Pack count with an inverted token so ties resolve to the lowest token id.
    packed = (run_counts.astype(np.uint64) << np.uint64(10)) | (
        np.uint64(1023) - continuations.astype(np.uint64)
    )
    best = np.maximum.reduceat(packed, group_starts)
    top_count = (best >> np.uint64(10)).astype(np.uint32)
    top_token = (np.uint64(1023) - (best & np.uint64(1023))).astype(np.uint16)
    stats = {
        "encoded_records": int(cursor),
        "unique_fourgrams": int(contexts.size),
        "observed_contexts": int(unique_contexts.size),
        "excluded_target_positions": int(excluded),
    }
    return unique_contexts, top_token, top_count, support, stats


def validate_args(args: argparse.Namespace) -> None:
    if args.minimum_support < 1:
        raise ValueError("--minimum-support must be positive")
    if not 0.0 <= args.minimum_confidence <= 1.0:
        raise ValueError("--minimum-confidence must be within [0, 1]")
    if args.maximum_contexts < 1:
        raise ValueError("--maximum-contexts must be positive")
    if args.chunk_tokens < 1:
        raise ValueError("--chunk-tokens must be positive")
    if args.output.suffix.lower() != ".ptz":
        raise ValueError("--output must end in .ptz; a matching .lzma file is also written")


def main() -> None:
    args = parse_args()
    validate_args(args)

    shard = find_train_shard(args)
    tokens = shard_memmap(shard)
    print(f"counting shard:{shard.name} tokens:{tokens.size}")
    contexts, top_token, top_count, support, stats = build_expert(
        tokens, args.chunk_tokens
    )

    confidence = top_count.astype(np.float64) / np.maximum(support, 1)
    keep = (support >= args.minimum_support) & (
        confidence >= args.minimum_confidence
    )
    contexts = contexts[keep]
    top_token = top_token[keep]
    top_count = top_count[keep]
    support = support[keep]
    if contexts.size > args.maximum_contexts:
        # Highest support first: those contexts generalize best per stored byte.
        order = np.argsort(support, kind="stable")[::-1][: args.maximum_contexts]
        order.sort()
        contexts = contexts[order]
        top_token = top_token[order]
        top_count = top_count[order]
        support = support[order]

    payload = {
        "__format__": "local_sparse_fourgram_v1",
        "context_u32": torch.from_numpy(np.ascontiguousarray(contexts)),
        "top_token_u16": torch.from_numpy(np.ascontiguousarray(top_token)),
        "top_count_u32": torch.from_numpy(np.ascontiguousarray(top_count)),
        "support_u32": torch.from_numpy(np.ascontiguousarray(support)),
        "metadata": {
            **stats,
            "context_tokens": CONTEXT_TOKENS,
            "kept_contexts": int(contexts.size),
            "minimum_confidence": args.minimum_confidence,
            "minimum_support": args.minimum_support,
            "train_shard_name": shard.name,
            "calibration_exclusion_start": CALIBRATION_START,
            "calibration_exclusion_stop_inclusive": CALIBRATION_STOP,
        },
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    compressed = zlib.compress(buffer.getvalue(), level=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(compressed)

    # The evaluator only needs the context key and its top continuation, so the
    # shipped sidecar stores just those two arrays. Sorted keys are delta coded
    # before LZMA, which more than halves the bytes per retained context.
    deltas = np.diff(contexts.astype(np.int64), prepend=0).astype(np.uint32)
    stored_confidence = np.rint(
        np.clip(top_count.astype(np.float64) / np.maximum(support, 1), 0.0, 1.0)
        * 255.0
    ).astype(np.uint8)
    body = (
        MAGIC
        + np.asarray(contexts.size, dtype="<u4").tobytes()
        + np.ascontiguousarray(deltas, dtype="<u4").tobytes()
        + np.ascontiguousarray(top_token, dtype="<u2").tobytes()
        + np.ascontiguousarray(stored_confidence).tobytes()
    )
    compact = lzma.compress(body, preset=9)
    compact_path = args.output.with_suffix(".lzma")
    compact_path.write_bytes(compact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sidecar_bytes": len(compressed),
                "compact_output": str(compact_path),
                "compact_bytes": len(compact),
                "bytes_per_context": round(len(compact) / max(contexts.size, 1), 3),
                "kept_contexts": int(contexts.size),
                **stats,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

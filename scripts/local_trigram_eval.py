#!/usr/bin/env python3
"""Evaluate a compact train-only top-continuation trigram expert on CPU.

All expert selection is completed on the excluded training calibration segment.
The validation shard is not opened until the legal, size-checked sidecar is saved.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_gpt as baseline
from local_cpu_smoke import LocalGPT, load_compressed_state
from scripts.local_eval_common import (
    ARTIFACT_LIMIT,
    apply_calibration_logits,
    load_calibration_sidecar,
    mix_log_probabilities,
)

SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
CALIBRATION_START = 20_000_000
CALIBRATION_STOP = 20_008_192


def comma_floats(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid numeric grid: {text}") from exc
    if not values:
        raise argparse.ArgumentTypeError("grid must not be empty")
    return values


def comma_ints(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer grid: {text}") from exc
    if not values:
        raise argparse.ArgumentTypeError("grid must not be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an exact top-continuation trigram expert from the canonical "
            "100M-token SP1024 train shard, tune it only on the excluded train "
            "segment, enforce the 16MB combined budget, then evaluate validation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=REPO_ROOT / "logs" / "target2_stage4.ptz",
        help="Neural artifact produced by local_cpu_smoke.py.",
    )
    parser.add_argument(
        "--bigram-sidecar",
        type=Path,
        default=REPO_ROOT / "logs" / "extended_heldout.calibrated.ptz",
        help="Self-contained calibrated bigram sidecar.",
    )
    parser.add_argument(
        "--trigram-sidecar",
        type=Path,
        default=REPO_ROOT / "logs" / "extended_heldout.trigram.ptz",
        help="Output compact trigram expert.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024",
    )
    parser.add_argument("--train-shard", type=Path)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model",
    )
    parser.add_argument("--calibration-offset", type=int, default=CALIBRATION_START)
    parser.add_argument("--calibration-tokens", type=int, default=8192)
    parser.add_argument("--validation-offset", type=int, default=16384)
    parser.add_argument("--validation-tokens", type=int, default=65536)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-tokens", type=int, default=4096)
    parser.add_argument(
        "--encode-chunk-tokens",
        type=int,
        default=8_000_000,
        help="Peak-memory control while encoding triples before the in-place sort.",
    )
    parser.add_argument(
        "--confidence-thresholds",
        type=comma_floats,
        default=comma_floats("0,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.75"),
    )
    parser.add_argument(
        "--minimum-supports",
        type=comma_ints,
        default=comma_ints("1,2,4,8,16,32,64,128,256"),
    )
    parser.add_argument(
        "--logit-boosts",
        type=comma_floats,
        default=comma_floats("0,0.125,0.25,0.5,0.75,1,1.5,2,3"),
        help="Boost applied to the expert's top token before exact renormalization.",
    )
    parser.add_argument("--threads", type=int, default=8)

    architecture = parser.add_argument_group(
        "artifact architecture (target2_stage4 defaults)"
    )
    architecture.add_argument("--vocab-size", type=int, default=1024)
    architecture.add_argument("--num-layers", type=int, default=2)
    architecture.add_argument("--model-dim", type=int, default=256)
    architecture.add_argument("--num-heads", type=int, default=8)
    architecture.add_argument("--num-kv-heads", type=int, default=4)
    architecture.add_argument("--mlp-mult", type=int, default=2)
    architecture.add_argument("--bigram-vocab-size", type=int, default=65536)
    architecture.add_argument("--bigram-dim", type=int, default=256)
    architecture.add_argument("--qk-gain-init", type=float, default=1.5)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.vocab_size != 1024:
        raise ValueError("uint32 triple encoding requires the canonical SP1024 vocabulary")
    if args.calibration_offset != CALIBRATION_START or args.calibration_tokens != 8192:
        raise ValueError(
            "this experiment requires calibration positions "
            "20,000,000..20,008,192"
        )
    for name in (
        "validation_tokens",
        "seq_len",
        "batch_tokens",
        "encode_chunk_tokens",
        "threads",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.validation_offset <= 0:
        raise ValueError("--validation-offset must be positive to supply two prior tokens")
    if args.validation_tokens % args.seq_len:
        raise ValueError("--validation-tokens must be divisible by --seq-len")
    if args.calibration_tokens % args.seq_len:
        raise ValueError("--calibration-tokens must be divisible by --seq-len")
    if args.batch_tokens < args.seq_len:
        raise ValueError("--batch-tokens must be at least --seq-len")
    if args.batch_tokens % args.seq_len:
        raise ValueError("--batch-tokens must be divisible by --seq-len")
    if any(not 0 <= value <= 1 for value in args.confidence_thresholds):
        raise ValueError("confidence thresholds must be in [0, 1]")
    if any(value < 1 for value in args.minimum_supports):
        raise ValueError("minimum supports must be positive")
    if any(value < 0 for value in args.logit_boosts):
        raise ValueError("logit boosts must be non-negative")
    if args.num_layers <= 0 or args.model_dim <= 0 or args.mlp_mult <= 0:
        raise ValueError("artifact architecture dimensions must be positive")
    if args.num_heads <= 0 or args.num_kv_heads <= 0:
        raise ValueError("artifact head counts must be positive")
    if args.model_dim % args.num_heads:
        raise ValueError("--model-dim must be divisible by --num-heads")
    if args.num_heads % args.num_kv_heads:
        raise ValueError("--num-heads must be divisible by --num-kv-heads")
    if args.bigram_vocab_size == 1 or args.bigram_vocab_size < 0:
        raise ValueError("--bigram-vocab-size must be zero or at least two")
    if args.bigram_dim <= 0 or args.qk_gain_init <= 0:
        raise ValueError("bigram dimension and QK gain must be positive")


def shard_memmap(path: Path) -> np.memmap:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
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
    paths = sorted(args.data_path.glob("fineweb_train_*.bin"))
    if not paths:
        raise FileNotFoundError(f"No training shard in {args.data_path}")
    return paths[0]


def encode_target_range(
    output: np.ndarray,
    output_start: int,
    tokens: np.memmap,
    target_start: int,
    target_stop: int,
    chunk_tokens: int,
) -> int:
    cursor = output_start
    for start in range(target_start, target_stop, chunk_tokens):
        stop = min(start + chunk_tokens, target_stop)
        previous2 = np.asarray(tokens[start - 2 : stop - 2], dtype=np.uint32)
        previous1 = np.asarray(tokens[start - 1 : stop - 1], dtype=np.uint32)
        target = np.asarray(tokens[start:stop], dtype=np.uint32)
        if (
            previous2.size
            and max(
                int(previous2.max()), int(previous1.max()), int(target.max())
            )
            >= 1024
        ):
            raise ValueError("training shard contains a token outside SP1024")
        count = stop - start
        output[cursor : cursor + count] = (
            (previous2 << np.uint32(20))
            | (previous1 << np.uint32(10))
            | target
        )
        cursor += count
    return cursor


def build_exact_top_trigrams(
    tokens: np.memmap, chunk_tokens: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    # Target positions in this inclusive interval are absent from the expert.
    excluded_start = CALIBRATION_START
    excluded_stop_inclusive = CALIBRATION_STOP
    if excluded_stop_inclusive >= tokens.size:
        raise ValueError("calibration exclusion extends beyond training shard")
    total_triples = tokens.size - 2
    excluded = excluded_stop_inclusive - excluded_start + 1
    encoded = np.empty(total_triples - excluded, dtype=np.uint32)
    cursor = encode_target_range(
        encoded, 0, tokens, 2, excluded_start, chunk_tokens
    )
    cursor = encode_target_range(
        encoded,
        cursor,
        tokens,
        excluded_stop_inclusive + 1,
        tokens.size,
        chunk_tokens,
    )
    if cursor != encoded.size:
        raise RuntimeError(f"encoded {cursor} triples, allocated {encoded.size}")

    print(f"sorting encoded_triples:{encoded.size}")
    encoded.sort(kind="quicksort")
    run_starts = np.empty(
        int(np.count_nonzero(encoded[1:] != encoded[:-1])) + 1,
        dtype=np.int64,
    )
    run_starts[0] = 0
    run_starts[1:] = np.flatnonzero(encoded[1:] != encoded[:-1]) + 1
    unique_keys = encoded[run_starts].copy()
    run_stops = np.empty_like(run_starts)
    run_stops[:-1] = run_starts[1:]
    run_stops[-1] = encoded.size
    run_counts = (run_stops - run_starts).astype(np.uint32)
    del encoded, run_stops, run_starts

    contexts = unique_keys >> np.uint32(10)
    continuations = (unique_keys & np.uint32(1023)).astype(np.uint16)
    group_starts = np.empty(
        int(np.count_nonzero(contexts[1:] != contexts[:-1])) + 1,
        dtype=np.int64,
    )
    group_starts[0] = 0
    group_starts[1:] = np.flatnonzero(contexts[1:] != contexts[:-1]) + 1
    observed_contexts = contexts[group_starts].astype(np.int64)
    support_values = np.add.reduceat(
        run_counts.astype(np.uint64), group_starts
    ).astype(np.uint32)
    # Pack count and reverse token ID into one score: maximum count wins, then
    # the smallest token ID wins deterministic ties.
    scores = (run_counts.astype(np.uint64) << np.uint64(10)) | (
        np.uint64(1023) - continuations.astype(np.uint64)
    )
    maximum_scores = np.maximum.reduceat(scores, group_starts)
    top_count_values = (maximum_scores >> np.uint64(10)).astype(np.uint32)
    top_token_values = (
        np.uint64(1023) - (maximum_scores & np.uint64(1023))
    ).astype(np.uint16)

    context_count = 1024 * 1024
    top_token = np.full(context_count, np.uint16(65535), dtype=np.uint16)
    top_count = np.zeros(context_count, dtype=np.uint32)
    support = np.zeros(context_count, dtype=np.uint32)
    top_token[observed_contexts] = top_token_values
    top_count[observed_contexts] = top_count_values
    support[observed_contexts] = support_values
    stats = {
        "encoded_triples": cursor,
        "unique_triples": int(unique_keys.size),
        "observed_contexts": int(observed_contexts.size),
        "excluded_target_positions": excluded,
    }
    return top_token, top_count, support, stats


def load_model(args: argparse.Namespace) -> LocalGPT:
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer_path))
    if int(tokenizer.vocab_size()) != args.vocab_size:
        raise ValueError("tokenizer does not match --vocab-size")
    model = LocalGPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10_000.0,
        qk_gain_init=args.qk_gain_init,
        smear_gate=False,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        activation="relu2",
        orthogonal_init=False,
    ).float()
    model.load_state_dict(load_compressed_state(args.artifact), strict=True)
    model.eval()
    return model


def load_bigram_sidecar(
    path: Path,
    *,
    artifact_path: Path | None = None,
) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    return load_calibration_sidecar(
        path,
        expected_vocab_size=1024,
        artifact_path=artifact_path,
    )


def batched_sequences(
    tokens: Tensor, seq_len: int, batch_tokens: int
) -> Iterable[tuple[Tensor, Tensor]]:
    sequence_count = (tokens.numel() - 1) // seq_len
    batch_sequences = max(batch_tokens // seq_len, 1)
    for first in range(0, sequence_count, batch_sequences):
        last = min(first + batch_sequences, sequence_count)
        raw_start = first * seq_len
        raw_stop = last * seq_len + 1
        local = tokens[raw_start:raw_stop].to(torch.int64)
        yield (
            local[:-1].reshape(-1, seq_len),
            local[1:].reshape(-1, seq_len),
        )


def calibrated_log_probabilities(
    logits: Tensor,
    previous1: Tensor,
    targets: Tensor,
    expert_tokens: Tensor,
    log_bigram: Tensor,
    log_unigram: Tensor,
    bias: Tensor,
    params: dict[str, float],
) -> tuple[Tensor, Tensor]:
    adjusted = apply_calibration_logits(
        logits,
        previous1,
        log_bigram,
        log_unigram,
        bias,
        params,
    )
    normalizer = torch.logsumexp(adjusted, dim=1)
    target_neural = (
        adjusted.gather(1, targets[:, None]).squeeze(1) - normalizer
    )
    safe_expert = expert_tokens.clamp_min(0)
    expert_neural = (
        adjusted.gather(1, safe_expert[:, None]).squeeze(1) - normalizer
    )
    mixture = params["mixture"]
    target_base = mix_log_probabilities(
        target_neural,
        log_bigram[previous1, targets],
        mixture,
    )
    expert_base = mix_log_probabilities(
        expert_neural,
        log_bigram[previous1, safe_expert],
        mixture,
    )
    return target_base, expert_base


def collect_calibration(
    model: LocalGPT,
    model_tokens: Tensor,
    context_ids: Tensor,
    top_token: Tensor,
    log_bigram: Tensor,
    log_unigram: Tensor,
    bias: Tensor,
    bigram_params: dict[str, float],
    seq_len: int,
    batch_tokens: int,
) -> tuple[Tensor, Tensor, Tensor]:
    base_parts: list[Tensor] = []
    top_probability_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    cursor = 0
    with torch.inference_mode():
        for inputs, targets in batched_sequences(model_tokens, seq_len, batch_tokens):
            flat_targets = targets.reshape(-1)
            count = flat_targets.numel()
            local_contexts = context_ids[cursor : cursor + count]
            expert = top_token[local_contexts]
            logits = model.forward_logits(inputs).reshape(-1, 1024)
            base_logp, expert_logp = calibrated_log_probabilities(
                logits,
                inputs.reshape(-1),
                flat_targets,
                expert,
                log_bigram,
                log_unigram,
                bias,
                bigram_params,
            )
            base_parts.append(base_logp)
            top_probability_parts.append(expert_logp.exp())
            target_parts.append(flat_targets)
            cursor += count
    return (
        torch.cat(base_parts),
        torch.cat(top_probability_parts),
        torch.cat(target_parts),
    )


def trigram_nll(
    base_logp: Tensor,
    top_probability: Tensor,
    targets: Tensor,
    expert_token: Tensor,
    top_count: Tensor,
    support: Tensor,
    confidence_threshold: float,
    minimum_support: int,
    logit_boost: float,
) -> tuple[float, int]:
    confidence = top_count.to(torch.float32) / support.clamp_min(1).to(torch.float32)
    active = (
        (support >= minimum_support)
        & (confidence >= confidence_threshold)
        & (expert_token >= 0)
    )
    if not logit_boost:
        return -float(base_logp.sum(dtype=torch.float64).item()) / targets.numel(), int(
            active.sum().item()
        )
    normalization = torch.zeros_like(base_logp)
    normalization[active] = torch.log1p(
        top_probability[active] * math.expm1(logit_boost)
    )
    hit = active & (targets == expert_token)
    final_logp = base_logp - normalization
    final_logp[hit] += logit_boost
    return -float(final_logp.sum(dtype=torch.float64).item()) / targets.numel(), int(
        active.sum().item()
    )


def sidecar_payload(
    context_ids: np.ndarray,
    top_token: np.ndarray,
    top_count: np.ndarray,
    support: np.ndarray,
    params: dict[str, float | int],
    stats: dict[str, int],
) -> bytes:
    payload = {
        "__format__": "local_top_trigram_v1",
        "context_u32": torch.from_numpy(context_ids.astype(np.uint32, copy=False)),
        "top_token_u16": torch.from_numpy(top_token.astype(np.uint16, copy=False)),
        "top_count_u32": torch.from_numpy(top_count.astype(np.uint32, copy=False)),
        "support_u32": torch.from_numpy(support.astype(np.uint32, copy=False)),
        "params": params,
        "metadata": {
            **stats,
            "calibration_exclusion_start": CALIBRATION_START,
            "calibration_exclusion_stop_inclusive": CALIBRATION_STOP,
        },
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return zlib.compress(buffer.getvalue(), level=9)


def runtime_code_paths() -> tuple[Path, ...]:
    return (
        Path(__file__),
        SCRIPT_DIR / "local_cpu_smoke.py",
        SCRIPT_DIR / "local_eval_common.py",
        REPO_ROOT / "train_gpt.py",
    )


def select_legal_expert(
    args: argparse.Namespace,
    base_logp: Tensor,
    top_probability: Tensor,
    targets: Tensor,
    context_ids: Tensor,
    dense_top_token: np.ndarray,
    dense_top_count: np.ndarray,
    dense_support: np.ndarray,
    build_stats: dict[str, int],
) -> tuple[dict[str, float | int], float, bytes, int, int]:
    context_np = context_ids.numpy()
    local_token = torch.from_numpy(dense_top_token[context_np].astype(np.int64))
    local_token[local_token == 65535] = -1
    local_top_count = torch.from_numpy(dense_top_count[context_np].astype(np.int64))
    local_support = torch.from_numpy(dense_support[context_np].astype(np.int64))

    candidates: list[tuple[float, float, int, float, int]] = []
    for threshold in args.confidence_thresholds:
        for minimum_support in args.minimum_supports:
            for boost in args.logit_boosts:
                nll, active = trigram_nll(
                    base_logp,
                    top_probability,
                    targets,
                    local_token,
                    local_top_count,
                    local_support,
                    threshold,
                    minimum_support,
                    boost,
                )
                candidates.append(
                    (nll, float(threshold), int(minimum_support), float(boost), active)
                )
    candidates.sort()
    fixed_bytes = (
        args.artifact.stat().st_size
        + args.bigram_sidecar.stat().st_size
        + sum(path.stat().st_size for path in runtime_code_paths())
    )
    confidence = dense_top_count.astype(np.float64) / np.maximum(dense_support, 1)
    for nll, threshold, minimum_support, boost, active in candidates:
        mask = (
            (dense_support >= minimum_support)
            & (confidence >= threshold)
            & (dense_top_token != 65535)
        )
        selected_context = np.flatnonzero(mask).astype(np.uint32)
        params: dict[str, float | int] = {
            "confidence_threshold": threshold,
            "minimum_support": minimum_support,
            "logit_boost": boost,
        }
        compressed = sidecar_payload(
            selected_context,
            dense_top_token[selected_context],
            dense_top_count[selected_context],
            dense_support[selected_context],
            params,
            build_stats,
        )
        total = fixed_bytes + len(compressed)
        if total <= ARTIFACT_LIMIT:
            return params, nll, compressed, int(selected_context.size), active
    raise RuntimeError("No searched trigram expert fits the 16,000,000-byte limit")


def load_trigram_sidecar(
    path: Path, context_count: int
) -> tuple[Tensor, Tensor, Tensor, dict[str, float | int]]:
    payload = torch.load(
        io.BytesIO(zlib.decompress(path.read_bytes())),
        map_location="cpu",
        weights_only=True,
    )
    if payload.get("__format__") != "local_top_trigram_v1":
        raise ValueError(f"Unexpected trigram sidecar format: {path}")
    required = ("context_u32", "top_token_u16", "top_count_u32", "support_u32")
    if any(not isinstance(payload.get(name), Tensor) for name in required):
        raise ValueError(f"Trigram sidecar has missing tensor fields: {path}")
    lengths = {payload[name].numel() for name in required}
    if len(lengths) != 1 or any(payload[name].ndim != 1 for name in required):
        raise ValueError("Trigram sidecar arrays must be one-dimensional and equal length")
    context = payload["context_u32"].to(torch.int64)
    if context.numel() and (
        int(context.min()) < 0
        or int(context.max()) >= context_count
        or torch.unique(context).numel() != context.numel()
    ):
        raise ValueError("Trigram sidecar contains invalid or duplicate contexts")
    stored_token = payload["top_token_u16"].to(torch.int64)
    stored_count = payload["top_count_u32"].to(torch.int64)
    stored_support = payload["support_u32"].to(torch.int64)
    if stored_token.numel() and (
        int(stored_token.min()) < 0
        or int(stored_token.max()) >= 1024
        or (stored_count > stored_support).any()
        or (stored_support <= 0).any()
    ):
        raise ValueError("Trigram sidecar contains invalid token counts")
    top_token = torch.full((context_count,), -1, dtype=torch.int64)
    top_count = torch.zeros(context_count, dtype=torch.int64)
    support = torch.zeros(context_count, dtype=torch.int64)
    top_token[context] = stored_token
    top_count[context] = stored_count
    support[context] = stored_support
    params_payload = payload.get("params")
    if not isinstance(params_payload, dict):
        raise ValueError("Trigram sidecar is missing its parameters")
    try:
        params: dict[str, float | int] = {
            "confidence_threshold": float(params_payload["confidence_threshold"]),
            "minimum_support": int(params_payload["minimum_support"]),
            "logit_boost": float(params_payload["logit_boost"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Trigram sidecar has invalid parameters") from exc
    if (
        not 0.0 <= float(params["confidence_threshold"]) <= 1.0
        or int(params["minimum_support"]) < 1
        or float(params["logit_boost"]) < 0
    ):
        raise ValueError("Trigram sidecar parameters are out of range")
    return top_token, top_count, support, params


def evaluate(
    model: LocalGPT,
    model_tokens: Tensor,
    context_ids: Tensor,
    log_bigram: Tensor,
    log_unigram: Tensor,
    bias: Tensor,
    bigram_params: dict[str, float],
    trigram_token: Tensor,
    trigram_count: Tensor,
    trigram_support: Tensor,
    trigram_params: dict[str, float | int],
    byte_luts: tuple[Tensor, Tensor, Tensor],
    seq_len: int,
    batch_tokens: int,
) -> dict[str, float | int]:
    baseline_sum = 0.0
    calibrated_sum = 0.0
    trigram_sum = 0.0
    token_count = 0
    byte_count = 0
    active_count = 0
    hit_count = 0
    cursor = 0
    base_bytes, leading_space, boundary = byte_luts
    with torch.inference_mode():
        for inputs, targets in batched_sequences(model_tokens, seq_len, batch_tokens):
            target = targets.reshape(-1)
            previous1 = inputs.reshape(-1)
            count = target.numel()
            local_context = context_ids[cursor : cursor + count]
            expert = trigram_token[local_context]
            local_top_count = trigram_count[local_context]
            local_support = trigram_support[local_context]
            logits = model.forward_logits(inputs).reshape(-1, 1024)
            baseline_sum += float(
                F.cross_entropy(logits, target, reduction="sum").item()
            )
            base_logp, expert_logp = calibrated_log_probabilities(
                logits,
                previous1,
                target,
                expert,
                log_bigram,
                log_unigram,
                bias,
                bigram_params,
            )
            calibrated_sum -= float(base_logp.sum(dtype=torch.float64).item())
            confidence = local_top_count.to(torch.float32) / local_support.clamp_min(
                1
            ).to(torch.float32)
            active = (
                (local_support >= int(trigram_params["minimum_support"]))
                & (
                    confidence
                    >= float(trigram_params["confidence_threshold"])
                )
                & (expert >= 0)
            )
            boost = float(trigram_params["logit_boost"])
            normalization = torch.zeros_like(base_logp)
            normalization[active] = torch.log1p(
                expert_logp[active].exp() * math.expm1(boost)
            )
            hit = active & (target == expert)
            final_logp = base_logp - normalization
            final_logp[hit] += boost
            trigram_sum -= float(final_logp.sum(dtype=torch.float64).item())
            active_count += int(active.sum().item())
            hit_count += int(hit.sum().item())
            target_bytes = base_bytes[target].to(torch.int16)
            target_bytes += (
                leading_space[target] & ~boundary[previous1]
            ).to(torch.int16)
            byte_count += int(target_bytes.sum().item())
            token_count += count
            cursor += count
    factor = 1.0 / (byte_count * math.log(2.0))
    return {
        "tokens": token_count,
        "bytes": byte_count,
        "active_predictions": active_count,
        "expert_hits": hit_count,
        "baseline_nll": baseline_sum / token_count,
        "baseline_bpb": baseline_sum * factor,
        "calibrated_nll": calibrated_sum / token_count,
        "calibrated_bpb": calibrated_sum * factor,
        "trigram_nll": trigram_sum / token_count,
        "trigram_bpb": trigram_sum * factor,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(min(args.threads, 4))
    for path, label in (
        (args.artifact, "artifact"),
        (args.bigram_sidecar, "bigram sidecar"),
        (args.tokenizer_path, "tokenizer"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.trigram_sidecar.resolve() in {
        args.artifact.resolve(),
        args.bigram_sidecar.resolve(),
    }:
        raise ValueError("--trigram-sidecar must not overwrite an input artifact")

    started = time.perf_counter()
    train_path = find_train_shard(args)
    train_tokens = shard_memmap(train_path)
    print(f"encoding train_shard:{train_path.name} tokens:{train_tokens.size}")
    dense_token, dense_count, dense_support, build_stats = (
        build_exact_top_trigrams(train_tokens, args.encode_chunk_tokens)
    )
    extended_calibration = np.asarray(
        train_tokens[
            args.calibration_offset - 1 :
            args.calibration_offset + args.calibration_tokens + 1
        ],
        dtype=np.int64,
    ).copy()
    del train_tokens
    calibration_model_tokens = torch.from_numpy(extended_calibration[1:].copy())
    calibration_contexts = torch.from_numpy(
        (
            extended_calibration[:-2] * args.vocab_size
            + extended_calibration[1:-1]
        ).astype(np.int64)
    )
    del extended_calibration

    model = load_model(args)
    log_bigram, log_unigram, bias, bigram_params = load_bigram_sidecar(
        args.bigram_sidecar,
        artifact_path=args.artifact,
    )
    dense_token_tensor = torch.from_numpy(dense_token.astype(np.int64))
    dense_token_tensor[dense_token_tensor == 65535] = -1
    base_logp, top_probability, calibration_targets = collect_calibration(
        model,
        calibration_model_tokens,
        calibration_contexts,
        dense_token_tensor,
        log_bigram,
        log_unigram,
        bias,
        bigram_params,
        args.seq_len,
        args.batch_tokens,
    )
    calibration_base_nll = -float(
        base_logp.sum(dtype=torch.float64).item()
    ) / base_logp.numel()
    params, calibration_trigram_nll, sidecar_bytes, context_entries, active = (
        select_legal_expert(
            args,
            base_logp,
            top_probability,
            calibration_targets,
            calibration_contexts,
            dense_token,
            dense_count,
            dense_support,
            build_stats,
        )
    )
    args.trigram_sidecar.parent.mkdir(parents=True, exist_ok=True)
    args.trigram_sidecar.write_bytes(sidecar_bytes)
    del (
        dense_token,
        dense_count,
        dense_support,
        dense_token_tensor,
        base_logp,
        top_probability,
        calibration_targets,
        calibration_model_tokens,
        calibration_contexts,
    )
    trigram_token, trigram_count, trigram_support, params = (
        load_trigram_sidecar(args.trigram_sidecar, args.vocab_size**2)
    )

    # Selection and persistence are complete. Validation is first accessed here.
    validation_paths = sorted(args.data_path.glob("fineweb_val_*.bin"))
    if not validation_paths:
        raise FileNotFoundError(f"No validation shard in {args.data_path}")
    validation_shard = baseline.load_data_shard(validation_paths[0])
    stop = args.validation_offset + args.validation_tokens + 1
    if stop > validation_shard.numel():
        raise ValueError("requested validation interval exceeds the first shard")
    extended_validation = validation_shard[
        args.validation_offset - 1 : stop
    ].to(torch.int64)
    validation_model_tokens = extended_validation[1:].contiguous()
    validation_contexts = (
        extended_validation[:-2] * args.vocab_size
        + extended_validation[1:-1]
    ).contiguous()
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer_path))
    byte_luts = baseline.build_sentencepiece_luts(
        tokenizer, args.vocab_size, torch.device("cpu")
    )
    validation = evaluate(
        model,
        validation_model_tokens,
        validation_contexts,
        log_bigram,
        log_unigram,
        bias,
        bigram_params,
        trigram_token,
        trigram_count,
        trigram_support,
        params,
        byte_luts,
        args.seq_len,
        args.batch_tokens,
    )

    artifact_bytes = args.artifact.stat().st_size
    bigram_bytes = args.bigram_sidecar.stat().st_size
    trigram_bytes = args.trigram_sidecar.stat().st_size
    runtime_code_bytes = sum(path.stat().st_size for path in runtime_code_paths())
    total_bytes = (
        artifact_bytes + bigram_bytes + trigram_bytes + runtime_code_bytes
    )
    if total_bytes > ARTIFACT_LIMIT:
        raise RuntimeError(
            f"combined artifact is {total_bytes:,} bytes, above {ARTIFACT_LIMIT:,}"
        )
    summary = {
        "artifact_bytes": artifact_bytes,
        "bigram_sidecar_bytes": bigram_bytes,
        "trigram_sidecar_bytes": trigram_bytes,
        "runtime_code_bytes": runtime_code_bytes,
        "combined_bytes": total_bytes,
        "remaining_bytes": ARTIFACT_LIMIT - total_bytes,
        "trigram_sidecar": str(args.trigram_sidecar),
        "trigram_context_entries": context_entries,
        "build": build_stats,
        "selected": params,
        "calibration_base_nll": calibration_base_nll,
        "calibration_trigram_nll": calibration_trigram_nll,
        "calibration_active_predictions": active,
        "validation_offset": args.validation_offset,
        "validation_tokens": args.validation_tokens,
        "validation": validation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

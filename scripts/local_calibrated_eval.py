#!/usr/bin/env python3
"""Calibrate a local CPU checkpoint with a train-derived bigram sidecar.

The validation split is loaded only after every calibration parameter has been
selected and the sidecar has been written. No validation value participates in
temperature, mixture, product-of-experts, or unigram-bias selection.
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
    CALIBRATION_FORMAT,
    apply_calibration_logits,
    artifact_payload_sha256,
    load_calibration_sidecar,
    mix_log_probabilities,
)

SHARD_HEADER_INTS = 256
SHARD_HEADER_BYTES = SHARD_HEADER_INTS * np.dtype("<i4").itemsize
MIN_CALIBRATION_OFFSET = 20_000_000


def comma_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers: {value}") from exc
    if not result:
        raise argparse.ArgumentTypeError("grid must contain at least one number")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a local_cpu_smoke.py artifact, build an exact train-only "
            "1024x1024 bigram model, calibrate without validation access, save "
            "a compact sidecar, and finally report non-overlapping validation BPB."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=REPO_ROOT / "logs" / "target2_stage4.ptz",
        help="Compressed int8 artifact produced by local_cpu_smoke.py.",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        help="Output path. Defaults beside the artifact with '.calibrated.ptz'.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024",
        help="Canonical SP1024 shard directory.",
    )
    parser.add_argument(
        "--train-shard",
        type=Path,
        help="Canonical train shard. By default the first sorted train shard is used.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model",
    )
    parser.add_argument(
        "--calibration-offset",
        type=int,
        default=MIN_CALIBRATION_OFFSET,
        help="First calibration input token; values below 20M are rejected.",
    )
    parser.add_argument(
        "--calibration-tokens",
        type=int,
        default=8192,
        help="Number of train-only next-token predictions used for calibration.",
    )
    parser.add_argument("--validation-offset", type=int, default=0)
    parser.add_argument("--validation-tokens", type=int, default=16384)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument(
        "--batch-tokens",
        type=int,
        default=4096,
        help="Maximum model tokens per CPU forward batch.",
    )
    parser.add_argument(
        "--count-chunk-tokens",
        type=int,
        default=4_000_000,
        help="Chunk size used while counting the full train shard.",
    )
    parser.add_argument(
        "--bigram-smoothing",
        type=float,
        default=32.0,
        help="Unigram-backoff pseudocount per bigram row.",
    )
    parser.add_argument(
        "--bias-prior-tokens",
        type=float,
        default=4096.0,
        help="Training-unigram prior strength for calibration target marginals.",
    )
    parser.add_argument(
        "--bias-shrink-count",
        type=float,
        default=32.0,
        help="Per-token shrinkage denominator for the fitted unigram bias.",
    )
    parser.add_argument(
        "--temperatures",
        type=comma_floats,
        default=comma_floats("0.85,0.925,1.0,1.075,1.15"),
        help="Coordinate-search grid.",
    )
    parser.add_argument(
        "--positive-softcaps",
        type=comma_floats,
        default=comma_floats("15,22.5,30,45,60,90"),
        help="AsymLogit positive softcap grid from public PR #1923.",
    )
    parser.add_argument(
        "--negative-softcaps",
        type=comma_floats,
        default=comma_floats("15,22.5,30,45,60,90"),
        help="AsymLogit negative softcap grid from public PR #1923.",
    )
    parser.add_argument(
        "--mixture-weights",
        type=comma_floats,
        default=comma_floats("0,0.025,0.05,0.1,0.2"),
        help="Arithmetic count-model mixture grid.",
    )
    parser.add_argument(
        "--poe-strengths",
        type=comma_floats,
        default=comma_floats("0,0.125,0.25,0.5,0.75"),
        help="Strength grid for the bigram log-odds product-of-experts tilt.",
    )
    parser.add_argument(
        "--bias-strengths",
        type=comma_floats,
        default=comma_floats("0,0.25,0.5,0.75,1.0"),
        help="Global multiplier grid for the shrunk per-token bias.",
    )
    parser.add_argument(
        "--search-passes",
        type=int,
        default=2,
        help="Coordinate-descent passes over all four calibration grids.",
    )
    parser.add_argument("--threads", type=int, default=8)

    architecture = parser.add_argument_group(
        "artifact architecture (defaults match target2_stage4)"
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
    architecture.add_argument(
        "--activation", choices=("relu2", "leaky_relu2"), default="relu2"
    )
    architecture.add_argument("--smear-gate", action="store_true")
    architecture.add_argument("--orthogonal-init", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.calibration_offset < MIN_CALIBRATION_OFFSET:
        raise ValueError(
            f"--calibration-offset must be at least {MIN_CALIBRATION_OFFSET:,}; "
            "the earlier region overlaps the local model's training stream"
        )
    for name in (
        "calibration_tokens",
        "validation_tokens",
        "seq_len",
        "batch_tokens",
        "count_chunk_tokens",
        "threads",
        "vocab_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.validation_offset < 0:
        raise ValueError("--validation-offset must be non-negative")
    if args.calibration_tokens % args.seq_len:
        raise ValueError("--calibration-tokens must be divisible by --seq-len")
    if args.validation_tokens % args.seq_len:
        raise ValueError("--validation-tokens must be divisible by --seq-len")
    if args.batch_tokens < args.seq_len:
        raise ValueError("--batch-tokens must be at least --seq-len")
    if args.batch_tokens % args.seq_len:
        raise ValueError("--batch-tokens must be divisible by --seq-len")
    if args.bigram_smoothing <= 0:
        raise ValueError("--bigram-smoothing must be positive")
    if args.bias_prior_tokens < 0 or args.bias_shrink_count < 0:
        raise ValueError("bias shrinkage values must be non-negative")
    if args.search_passes <= 0:
        raise ValueError("--search-passes must be positive")
    if any(
        value <= 0
        for values in (
            args.temperatures,
            args.positive_softcaps,
            args.negative_softcaps,
        )
        for value in values
    ):
        raise ValueError("temperatures and softcaps must be positive")
    if any(not 0 <= value <= 1 for value in args.mixture_weights):
        raise ValueError("all mixture weights must be between zero and one")
    if any(value < 0 for value in (*args.poe_strengths, *args.bias_strengths)):
        raise ValueError("product-of-experts and bias strengths must be non-negative")
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
    header = np.fromfile(path, dtype="<i4", count=SHARD_HEADER_INTS)
    if (
        header.size != SHARD_HEADER_INTS
        or int(header[0]) != 20240520
        or int(header[1]) != 1
    ):
        raise ValueError(f"Unexpected shard header for {path}")
    token_count = int(header[2])
    expected_size = SHARD_HEADER_BYTES + token_count * np.dtype("<u2").itemsize
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"Shard size mismatch for {path}: expected {expected_size} bytes"
        )
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
        raise FileNotFoundError(f"No canonical training shard found in {args.data_path}")
    return shards[0]


def add_pair_range(
    counts_flat: np.ndarray,
    tokens: np.memmap,
    pair_start: int,
    pair_stop: int,
    vocab_size: int,
    chunk_tokens: int,
) -> None:
    for start in range(pair_start, pair_stop, chunk_tokens):
        stop = min(start + chunk_tokens, pair_stop)
        previous = np.asarray(tokens[start:stop], dtype=np.int64)
        target = np.asarray(tokens[start + 1 : stop + 1], dtype=np.int64)
        if (
            previous.size
            and (int(previous.max()) >= vocab_size or int(target.max()) >= vocab_size)
        ):
            raise ValueError("Training shard contains a token outside --vocab-size")
        pair_ids = previous * vocab_size + target
        counts_flat += np.bincount(
            pair_ids, minlength=vocab_size * vocab_size
        ).astype(np.uint64, copy=False)


def build_bigram_counts(
    tokens: np.memmap,
    vocab_size: int,
    calibration_offset: int,
    calibration_tokens: int,
    chunk_tokens: int,
) -> np.ndarray:
    pair_count = tokens.size - 1
    calibration_stop = calibration_offset + calibration_tokens
    if calibration_stop > pair_count:
        raise ValueError("Calibration segment extends beyond the training shard")
    counts_flat = np.zeros(vocab_size * vocab_size, dtype=np.uint64)
    # Exclude every calibration prediction pair. This makes the segment a true
    # train-only holdout for selecting the fusion parameters.
    add_pair_range(
        counts_flat,
        tokens,
        0,
        calibration_offset,
        vocab_size,
        chunk_tokens,
    )
    add_pair_range(
        counts_flat,
        tokens,
        calibration_stop,
        pair_count,
        vocab_size,
        chunk_tokens,
    )
    return counts_flat.reshape(vocab_size, vocab_size)


def quantize_log_table(
    counts: np.ndarray, smoothing: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vocab_size = counts.shape[0]
    target_counts = counts.sum(axis=0, dtype=np.uint64)
    unigram = (target_counts.astype(np.float64) + 0.5) / (
        float(target_counts.sum()) + 0.5 * vocab_size
    )
    row_totals = counts.sum(axis=1, dtype=np.uint64).astype(np.float64)
    probabilities = (
        counts.astype(np.float64) + smoothing * unigram[None, :]
    ) / (row_totals[:, None] + smoothing)
    log_probabilities = np.log(probabilities).astype(np.float32)

    row_min = log_probabilities.min(axis=1)
    row_max = log_probabilities.max(axis=1)
    row_scale = np.maximum((row_max - row_min) / 255.0, 1e-8)
    quantized = np.rint(
        (log_probabilities - row_min[:, None]) / row_scale[:, None]
    ).clip(0, 255).astype(np.uint8)

    # Float16 metadata is what the sidecar stores. Search and evaluation use its
    # roundtrip, so reported scores include table quantization exactly.
    row_min_f16 = row_min.astype(np.float16)
    row_scale_f16 = row_scale.astype(np.float16)
    dequantized = (
        quantized.astype(np.float32) * row_scale_f16.astype(np.float32)[:, None]
        + row_min_f16.astype(np.float32)[:, None]
    )
    del probabilities, log_probabilities
    return quantized, row_min_f16, row_scale_f16, dequantized


def load_model(args: argparse.Namespace) -> LocalGPT:
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer_path))
    if int(tokenizer.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"Tokenizer has {tokenizer.vocab_size()} tokens, expected {args.vocab_size}"
        )
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
        smear_gate=args.smear_gate,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        activation=args.activation,
        orthogonal_init=args.orthogonal_init,
    ).float()
    model.load_state_dict(load_compressed_state(args.artifact), strict=True)
    model.eval()
    return model


def batched_sequences(
    tokens: Tensor, seq_len: int, batch_tokens: int
) -> Iterable[tuple[Tensor, Tensor]]:
    sequence_count = (tokens.numel() - 1) // seq_len
    batch_sequences = max(batch_tokens // seq_len, 1)
    for first in range(0, sequence_count, batch_sequences):
        last = min(first + batch_sequences, sequence_count)
        raw_start = first * seq_len
        raw_stop = last * seq_len + 1
        local = tokens[raw_start:raw_stop].to(dtype=torch.int64)
        yield (
            local[:-1].reshape(-1, seq_len),
            local[1:].reshape(-1, seq_len),
        )


def collect_calibration_logits(
    model: LocalGPT, tokens: Tensor, seq_len: int, batch_tokens: int
) -> tuple[Tensor, Tensor, Tensor]:
    logits_parts: list[Tensor] = []
    previous_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    vocab_size = model.tok_emb.num_embeddings
    with torch.inference_mode():
        for inputs, targets in batched_sequences(tokens, seq_len, batch_tokens):
            logits_parts.append(model.forward_logits(inputs).reshape(-1, vocab_size))
            previous_parts.append(inputs.reshape(-1))
            target_parts.append(targets.reshape(-1))
    return (
        torch.cat(logits_parts).contiguous(),
        torch.cat(previous_parts).contiguous(),
        torch.cat(target_parts).contiguous(),
    )


def fit_shrunk_bias(
    logits: Tensor,
    targets: Tensor,
    train_unigram: Tensor,
    prior_tokens: float,
    shrink_count: float,
    chunk_tokens: int,
) -> Tensor:
    predicted = torch.zeros(logits.size(1), dtype=torch.float64)
    with torch.inference_mode():
        for start in range(0, logits.size(0), chunk_tokens):
            predicted += F.softmax(
                logits[start : start + chunk_tokens], dim=-1
            ).sum(dim=0, dtype=torch.float64)
    predicted /= logits.size(0)
    observed_counts = torch.bincount(
        targets, minlength=logits.size(1)
    ).to(torch.float64)
    observed = (
        observed_counts + prior_tokens * train_unigram.to(torch.float64)
    ) / (logits.size(0) + prior_tokens)
    raw_bias = torch.log(observed.clamp_min(1e-12)) - torch.log(
        predicted.clamp_min(1e-12)
    )
    reliability = observed_counts / (observed_counts + shrink_count).clamp_min(1.0)
    bias = (raw_bias * reliability).clamp(-4.0, 4.0)
    bias -= (bias * observed).sum()
    return bias.to(torch.float16).to(torch.float32)


def fused_nll(
    logits: Tensor,
    previous: Tensor,
    targets: Tensor,
    log_bigram: Tensor,
    log_unigram: Tensor,
    bias: Tensor,
    positive_softcap: float,
    negative_softcap: float,
    temperature: float,
    mixture: float,
    poe_strength: float,
    bias_strength: float,
    chunk_tokens: int,
) -> float:
    total = 0.0
    with torch.inference_mode():
        for start in range(0, targets.numel(), chunk_tokens):
            stop = min(start + chunk_tokens, targets.numel())
            local_previous = previous[start:stop]
            local_targets = targets[start:stop]
            params = {
                "positive_softcap": positive_softcap,
                "negative_softcap": negative_softcap,
                "temperature": temperature,
                "mixture": mixture,
                "poe_strength": poe_strength,
                "bias_strength": bias_strength,
            }
            adjusted = apply_calibration_logits(
                logits[start:stop],
                local_previous,
                log_bigram,
                log_unigram,
                bias,
                params,
            )
            neural_log_probability = (
                adjusted.gather(1, local_targets[:, None]).squeeze(1)
                - torch.logsumexp(adjusted, dim=1)
            )
            final_log_probability = mix_log_probabilities(
                neural_log_probability,
                log_bigram[local_previous, local_targets],
                mixture,
            )
            total -= float(final_log_probability.sum(dtype=torch.float64).item())
    return total / targets.numel()


def coordinate_search(
    args: argparse.Namespace,
    logits: Tensor,
    previous: Tensor,
    targets: Tensor,
    log_bigram: Tensor,
    log_unigram: Tensor,
    bias: Tensor,
) -> tuple[dict[str, float], float]:
    params = {
        "positive_softcap": 30.0,
        "negative_softcap": 30.0,
        "temperature": 1.0,
        "mixture": 0.0,
        "poe_strength": 0.0,
        "bias_strength": 0.0,
    }
    grids = (
        ("positive_softcap", args.positive_softcaps),
        ("negative_softcap", args.negative_softcaps),
        ("temperature", args.temperatures),
        ("poe_strength", args.poe_strengths),
        ("bias_strength", args.bias_strengths),
        ("mixture", args.mixture_weights),
    )

    def score(candidate: dict[str, float]) -> float:
        return fused_nll(
            logits,
            previous,
            targets,
            log_bigram,
            log_unigram,
            bias,
            candidate["positive_softcap"],
            candidate["negative_softcap"],
            candidate["temperature"],
            candidate["mixture"],
            candidate["poe_strength"],
            candidate["bias_strength"],
            args.batch_tokens,
        )

    best_nll = score(params)
    print(f"calibration baseline_nll:{best_nll:.8f}")
    for pass_index in range(args.search_passes):
        changed = False
        for name, values in grids:
            candidates: list[tuple[float, float]] = []
            for value in set(values) | {params[name]}:
                candidate = dict(params)
                candidate[name] = float(value)
                candidates.append((score(candidate), float(value)))
            axis_nll, axis_value = min(candidates, key=lambda item: (item[0], item[1]))
            changed = changed or params[name] != axis_value
            params[name] = axis_value
            best_nll = axis_nll
            print(
                f"search pass:{pass_index + 1} axis:{name} "
                f"value:{params[name]:.6g} nll:{best_nll:.8f}"
            )
        if not changed:
            break
    # Re-score the final joint setting rather than relying on an axis intermediate.
    return params, score(params)


def save_sidecar(
    path: Path,
    quantized: np.ndarray,
    row_min: np.ndarray,
    row_scale: np.ndarray,
    log_unigram: Tensor,
    bias: Tensor,
    params: dict[str, float],
    args: argparse.Namespace,
    train_shard: Path,
    counted_pairs: int,
) -> int:
    payload = {
        "__format__": CALIBRATION_FORMAT,
        "log_bigram_u8": torch.from_numpy(quantized),
        "log_bigram_min_f16": torch.from_numpy(row_min),
        "log_bigram_scale_f16": torch.from_numpy(row_scale),
        "log_unigram_f16": log_unigram.to(torch.float16),
        "unigram_bias_f16": bias.to(torch.float16),
        "params": params,
        "metadata": {
            "vocab_size": args.vocab_size,
            "bigram_smoothing": args.bigram_smoothing,
            "bias_prior_tokens": args.bias_prior_tokens,
            "bias_shrink_count": args.bias_shrink_count,
            "calibration_offset": args.calibration_offset,
            "calibration_tokens": args.calibration_tokens,
            "counted_pairs": counted_pairs,
            "train_shard_name": train_shard.name,
            "artifact_payload_sha256": artifact_payload_sha256(args.artifact),
            "artifact_architecture": {
                "vocab_size": args.vocab_size,
                "num_layers": args.num_layers,
                "model_dim": args.model_dim,
                "num_heads": args.num_heads,
                "num_kv_heads": args.num_kv_heads,
                "mlp_mult": args.mlp_mult,
                "bigram_vocab_size": args.bigram_vocab_size,
                "bigram_dim": args.bigram_dim,
                "activation": args.activation,
                "smear_gate": args.smear_gate,
            },
        },
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    compressed = zlib.compress(buffer.getvalue(), level=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    return len(compressed)


def load_sidecar_roundtrip(
    path: Path,
    *,
    expected_vocab_size: int | None = None,
    artifact_path: Path | None = None,
) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    return load_calibration_sidecar(
        path,
        expected_vocab_size=expected_vocab_size,
        artifact_path=artifact_path,
    )


def evaluate(
    model: LocalGPT,
    tokens: Tensor,
    seq_len: int,
    batch_tokens: int,
    log_bigram: Tensor,
    log_unigram: Tensor,
    bias: Tensor,
    params: dict[str, float],
    byte_luts: tuple[Tensor, Tensor, Tensor],
) -> dict[str, float | int]:
    baseline_loss_sum = 0.0
    calibrated_loss_sum = 0.0
    count_loss_sum = 0.0
    token_count = 0
    byte_count = 0
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = byte_luts

    with torch.inference_mode():
        for inputs, targets in batched_sequences(tokens, seq_len, batch_tokens):
            logits = model.forward_logits(inputs).reshape(
                -1, model.tok_emb.num_embeddings
            )
            previous = inputs.reshape(-1)
            target = targets.reshape(-1)
            baseline_loss_sum += float(
                F.cross_entropy(logits, target, reduction="sum").item()
            )
            adjusted = apply_calibration_logits(
                logits,
                previous,
                log_bigram,
                log_unigram,
                bias,
                params,
            )
            neural_log_probability = (
                adjusted.gather(1, target[:, None]).squeeze(1)
                - torch.logsumexp(adjusted, dim=1)
            )
            final_log_probability = mix_log_probabilities(
                neural_log_probability,
                log_bigram[previous, target],
                params["mixture"],
            )
            calibrated_loss_sum -= float(
                final_log_probability.sum(dtype=torch.float64).item()
            )
            count_loss_sum -= float(
                log_bigram[previous, target].sum(dtype=torch.float64).item()
            )
            target_bytes = base_bytes_lut[target].to(torch.int16)
            target_bytes += (
                has_leading_space_lut[target]
                & ~is_boundary_token_lut[previous]
            ).to(torch.int16)
            byte_count += int(target_bytes.sum().item())
            token_count += target.numel()

    bpb_factor = token_count / (byte_count * math.log(2.0))
    return {
        "tokens": token_count,
        "bytes": byte_count,
        "baseline_nll": baseline_loss_sum / token_count,
        "baseline_bpb": baseline_loss_sum * bpb_factor / token_count,
        "count_nll": count_loss_sum / token_count,
        "count_bpb": count_loss_sum * bpb_factor / token_count,
        "calibrated_nll": calibrated_loss_sum / token_count,
        "calibrated_bpb": calibrated_loss_sum * bpb_factor / token_count,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(min(args.threads, 4))
    sidecar_path = args.sidecar or args.artifact.with_name(
        f"{args.artifact.stem}.calibrated.ptz"
    )
    if sidecar_path.resolve() == args.artifact.resolve():
        raise ValueError("--sidecar must not overwrite --artifact")
    for path, description in (
        (args.artifact, "artifact"),
        (args.tokenizer_path, "tokenizer"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")

    train_shard = find_train_shard(args)
    train_tokens = shard_memmap(train_shard)
    started = time.perf_counter()
    print(
        f"counting shard:{train_shard.name} tokens:{train_tokens.size} "
        f"excluded_pairs:{args.calibration_tokens}"
    )
    counts = build_bigram_counts(
        train_tokens,
        args.vocab_size,
        args.calibration_offset,
        args.calibration_tokens,
        args.count_chunk_tokens,
    )
    counted_pairs = int(counts.sum())
    train_target_counts = counts.sum(axis=0, dtype=np.uint64)
    train_unigram_np = (train_target_counts.astype(np.float64) + 0.5) / (
        float(train_target_counts.sum()) + 0.5 * args.vocab_size
    )
    quantized, row_min, row_scale, log_bigram_np = quantize_log_table(
        counts, args.bigram_smoothing
    )
    del counts
    log_bigram = torch.from_numpy(log_bigram_np)
    train_unigram = torch.from_numpy(train_unigram_np.astype(np.float32))
    log_unigram = train_unigram.log()
    calibration_slice = np.asarray(
        train_tokens[
            args.calibration_offset : args.calibration_offset
            + args.calibration_tokens
            + 1
        ],
        dtype=np.int64,
    ).copy()
    calibration_tokens = torch.from_numpy(calibration_slice)
    # Release the memory map before model work; this is especially helpful on Windows.
    del train_tokens, calibration_slice

    model = load_model(args)
    logits, previous, targets = collect_calibration_logits(
        model, calibration_tokens, args.seq_len, args.batch_tokens
    )
    bias = fit_shrunk_bias(
        logits,
        targets,
        train_unigram,
        args.bias_prior_tokens,
        args.bias_shrink_count,
        args.batch_tokens,
    )
    baseline_calibration_nll = fused_nll(
        logits,
        previous,
        targets,
        log_bigram,
        log_unigram,
        bias,
        30.0,
        30.0,
        1.0,
        0.0,
        0.0,
        0.0,
        args.batch_tokens,
    )
    params, selected_calibration_nll = coordinate_search(
        args,
        logits,
        previous,
        targets,
        log_bigram,
        log_unigram,
        bias,
    )
    sidecar_bytes = save_sidecar(
        sidecar_path,
        quantized,
        row_min,
        row_scale,
        log_unigram,
        bias,
        params,
        args,
        train_shard,
        counted_pairs,
    )
    # All selection is complete before this point. Only now open validation.
    del log_bigram, bias
    log_bigram, log_unigram, bias, params = load_sidecar_roundtrip(
        sidecar_path,
        expected_vocab_size=args.vocab_size,
        artifact_path=args.artifact,
    )
    roundtrip_calibration_nll = fused_nll(
        logits,
        previous,
        targets,
        log_bigram,
        log_unigram,
        bias,
        params["positive_softcap"],
        params["negative_softcap"],
        params["temperature"],
        params["mixture"],
        params["poe_strength"],
        params["bias_strength"],
        args.batch_tokens,
    )
    del logits, previous, targets, calibration_tokens

    val_files = sorted(args.data_path.glob("fineweb_val_*.bin"))
    if not val_files:
        raise FileNotFoundError(f"No validation shard found in {args.data_path}")
    validation_shard = baseline.load_data_shard(val_files[0])
    validation_stop = args.validation_offset + args.validation_tokens + 1
    if args.validation_offset < 0 or validation_stop > validation_shard.numel():
        raise ValueError("Requested validation range is outside the first validation shard")
    validation_tokens = validation_shard[
        args.validation_offset:validation_stop
    ].contiguous()
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer_path))
    byte_luts = baseline.build_sentencepiece_luts(
        tokenizer, args.vocab_size, torch.device("cpu")
    )
    validation = evaluate(
        model,
        validation_tokens,
        args.seq_len,
        args.batch_tokens,
        log_bigram,
        log_unigram,
        bias,
        params,
        byte_luts,
    )

    artifact_bytes = args.artifact.stat().st_size
    runtime_code_paths = (
        Path(__file__),
        SCRIPT_DIR / "local_cpu_smoke.py",
        SCRIPT_DIR / "local_eval_common.py",
        REPO_ROOT / "train_gpt.py",
    )
    runtime_code_bytes = sum(path.stat().st_size for path in runtime_code_paths)
    summary = {
        "artifact": str(args.artifact),
        "artifact_bytes": artifact_bytes,
        "sidecar": str(sidecar_path),
        "sidecar_bytes": sidecar_bytes,
        "runtime_code_bytes": runtime_code_bytes,
        "combined_model_bytes": artifact_bytes + sidecar_bytes,
        "combined_with_runtime_code_bytes": artifact_bytes
        + sidecar_bytes
        + runtime_code_bytes,
        "counted_train_pairs": counted_pairs,
        "calibration_offset": args.calibration_offset,
        "calibration_tokens": args.calibration_tokens,
        "calibration_baseline_nll": baseline_calibration_nll,
        "calibration_selected_nll": selected_calibration_nll,
        "calibration_roundtrip_nll": roundtrip_calibration_nll,
        "selected": params,
        "validation_offset": args.validation_offset,
        "validation": validation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

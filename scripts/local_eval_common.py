#!/usr/bin/env python3
"""Shared artifact and calibration helpers for the local CPU study."""

from __future__ import annotations

import hashlib
import io
import lzma
import math
import zlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

ARTIFACT_LIMIT = 16_000_000
BASE_LOGIT_SOFTCAP = 30.0
CALIBRATION_FORMAT = "local_calibrated_bigram_v1"
LZMA_MAGIC = b"\xfd7zXZ\x00"
CALIBRATION_PARAMS = (
    "positive_softcap",
    "negative_softcap",
    "temperature",
    "mixture",
    "poe_strength",
    "bias_strength",
)


def decompress_container(blob: bytes, source: str | Path = "<memory>") -> bytes:
    """Decompress a zlib or XZ/LZMA artifact, detected from its contents."""
    try:
        if blob.startswith(LZMA_MAGIC):
            return lzma.decompress(blob)
        return zlib.decompress(blob)
    except (lzma.LZMAError, zlib.error) as exc:
        raise ValueError(f"Unsupported or corrupt compressed payload: {source}") from exc


def load_torch_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Compressed payload not found: {path}")
    raw = decompress_container(path.read_bytes(), path)
    try:
        payload = torch.load(
            io.BytesIO(raw),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ValueError(f"Invalid torch payload in {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dictionary payload in {path}")
    return payload


def artifact_payload_sha256(path: Path) -> str:
    return hashlib.sha256(
        decompress_container(path.read_bytes(), path)
    ).hexdigest()


def exact_token_window(
    *,
    offset_tokens: int,
    requested_tokens: int,
    seq_len: int,
    available_token_count: int,
    label: str,
) -> slice:
    """Validate and return an exact sequence-aligned token window."""
    if offset_tokens < 0:
        raise ValueError(f"{label} offset must be non-negative")
    if requested_tokens <= 0 or seq_len <= 0:
        raise ValueError(f"{label} token count and sequence length must be positive")
    if offset_tokens % seq_len:
        raise ValueError(f"{label} offset must be divisible by sequence length")
    if requested_tokens % seq_len:
        raise ValueError(f"{label} token count must be divisible by sequence length")
    stop = offset_tokens + requested_tokens + 1
    if stop > available_token_count:
        raise ValueError(
            f"{label} range [{offset_tokens}, {offset_tokens + requested_tokens}] "
            f"exceeds the shard's {available_token_count} tokens"
        )
    return slice(offset_tokens, stop)


def mix_log_probabilities(
    neural_log_probability: Tensor,
    count_log_probability: Tensor,
    mixture: float,
) -> Tensor:
    """Combine normalized models without evaluating log(0) at endpoint weights."""
    if not 0.0 <= mixture <= 1.0:
        raise ValueError("mixture must be between zero and one")
    if mixture == 0.0:
        return neural_log_probability
    if mixture == 1.0:
        return count_log_probability
    return torch.logaddexp(
        neural_log_probability + math.log1p(-mixture),
        count_log_probability + math.log(mixture),
    )


def apply_calibration_logits(
    capped_logits: Tensor,
    previous_tokens: Tensor,
    log_bigram: Tensor,
    log_unigram: Tensor,
    bias: Tensor,
    params: dict[str, float],
) -> Tensor:
    """Apply the same calibrated-logit transform in every local evaluator."""
    if capped_logits.ndim != 2:
        raise ValueError("calibration expects flattened [tokens, vocabulary] logits")
    if previous_tokens.numel() != capped_logits.size(0):
        raise ValueError("previous-token count does not match calibration logits")
    raw_logits = BASE_LOGIT_SOFTCAP * torch.atanh(
        (capped_logits / BASE_LOGIT_SOFTCAP).clamp(-0.999999, 0.999999)
    )
    adjusted = torch.where(
        raw_logits >= 0,
        params["positive_softcap"]
        * torch.tanh(raw_logits / params["positive_softcap"]),
        params["negative_softcap"]
        * torch.tanh(raw_logits / params["negative_softcap"]),
    ) / params["temperature"]
    if params["poe_strength"]:
        adjusted = adjusted + params["poe_strength"] * (
            log_bigram[previous_tokens] - log_unigram[None, :]
        )
    if params["bias_strength"]:
        adjusted = adjusted + params["bias_strength"] * bias[None, :]
    return adjusted


def load_calibration_sidecar(
    path: Path,
    *,
    expected_vocab_size: int | None = None,
    artifact_path: Path | None = None,
) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    payload = load_torch_payload(path)
    if payload.get("__format__") != CALIBRATION_FORMAT:
        raise ValueError(f"Unexpected calibration sidecar format in {path}")

    required_tensors = (
        "log_bigram_u8",
        "log_bigram_min_f16",
        "log_bigram_scale_f16",
        "log_unigram_f16",
        "unigram_bias_f16",
    )
    if any(not isinstance(payload.get(name), Tensor) for name in required_tensors):
        raise ValueError(f"Calibration sidecar has missing tensor fields: {path}")

    quantized = payload["log_bigram_u8"]
    if quantized.ndim != 2 or quantized.size(0) != quantized.size(1):
        raise ValueError("Calibration bigram table must be square")
    vocab_size = quantized.size(0)
    if expected_vocab_size is not None and vocab_size != expected_vocab_size:
        raise ValueError(
            f"Calibration vocabulary is {vocab_size}, expected {expected_vocab_size}"
        )
    for name in (
        "log_bigram_min_f16",
        "log_bigram_scale_f16",
        "log_unigram_f16",
        "unigram_bias_f16",
    ):
        if payload[name].shape != (vocab_size,):
            raise ValueError(f"Calibration tensor {name} has the wrong shape")

    params_payload = payload.get("params")
    if not isinstance(params_payload, dict):
        raise ValueError("Calibration sidecar is missing its parameter dictionary")
    legacy_defaults = {
        "positive_softcap": BASE_LOGIT_SOFTCAP,
        "negative_softcap": BASE_LOGIT_SOFTCAP,
    }
    try:
        params = {
            name: float(params_payload.get(name, legacy_defaults[name]))
            if name in legacy_defaults
            else float(params_payload[name])
            for name in CALIBRATION_PARAMS
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Calibration sidecar has invalid parameters") from exc
    if any(not math.isfinite(value) for value in params.values()):
        raise ValueError("Calibration parameters must be finite")
    if (
        params["positive_softcap"] <= 0
        or params["negative_softcap"] <= 0
        or params["temperature"] <= 0
        or not 0.0 <= params["mixture"] <= 1.0
        or params["poe_strength"] < 0
        or params["bias_strength"] < 0
    ):
        raise ValueError("Calibration sidecar contains out-of-range parameters")

    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict):
        metadata_vocab = metadata.get("vocab_size")
        if metadata_vocab is not None and int(metadata_vocab) != vocab_size:
            raise ValueError("Calibration metadata vocabulary does not match its table")
        expected_hash = metadata.get("artifact_payload_sha256")
        if artifact_path is not None and expected_hash is not None:
            actual_hash = artifact_payload_sha256(artifact_path)
            if str(expected_hash) != actual_hash:
                raise ValueError(
                    f"Calibration sidecar {path} was built for a different model artifact"
                )

    row_min = payload["log_bigram_min_f16"].to(torch.float32)
    row_scale = payload["log_bigram_scale_f16"].to(torch.float32)
    log_bigram = quantized.to(torch.float32) * row_scale[:, None] + row_min[:, None]
    log_unigram = payload["log_unigram_f16"].to(torch.float32)
    bias = payload["unigram_bias_f16"].to(torch.float32)
    if (
        not torch.isfinite(log_bigram).all()
        or not torch.isfinite(log_unigram).all()
        or not torch.isfinite(bias).all()
        or not (row_scale > 0).all()
    ):
        raise ValueError(f"Calibration sidecar contains non-finite values: {path}")
    return log_bigram, log_unigram, bias, params

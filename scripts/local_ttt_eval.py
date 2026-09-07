#!/usr/bin/env python3
"""Evaluate a local artifact with legal score-first test-time training."""

from __future__ import annotations

import argparse
import lzma
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_gpt as baseline
from scripts.local_cpu_smoke import LocalGPT, load_compressed_state, loss_and_bpb
from scripts.local_eval_common import (
    ARTIFACT_LIMIT,
    apply_calibration_logits,
    exact_token_window,
    load_calibration_sidecar,
    load_torch_payload,
    mix_log_probabilities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score each token once, then adapt on only the tokens already scored. "
            "The base artifact remains unchanged."
        )
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=REPO_ROOT / "logs" / "target2_stage4.ptz",
    )
    parser.add_argument(
        "--calibration-sidecar",
        type=Path,
        help="Optional train-only calibration sidecar from local_calibrated_eval.py.",
    )
    parser.add_argument(
        "--trigram-sidecar",
        type=Path,
        help="Optional train-only static trigram expert from local_trigram_eval.py.",
    )
    parser.add_argument(
        "--trigram-boost-override",
        type=float,
        default=-1.0,
        help="Override the train-selected static trigram boost; negative uses sidecar.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model",
    )
    parser.add_argument("--source", choices=("train", "val"), default="val")
    parser.add_argument("--offset-tokens", type=int, default=16_384)
    parser.add_argument("--eval-tokens", type=int, default=65_536)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--chunk-tokens", type=int, default=4096)
    parser.add_argument("--microbatch-tokens", type=int, default=2048)
    parser.add_argument("--ttt-epochs", type=int, default=1)
    parser.add_argument("--ttt-learning-rate", type=float, default=1e-4)
    parser.add_argument("--ttt-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--ttt-scope",
        choices=("controls", "core", "embedding", "bigram", "all"),
        default="core",
    )
    parser.add_argument(
        "--online-ngram-order",
        type=int,
        default=0,
        help="Prefix-only online n-gram context length. Zero disables it.",
    )
    parser.add_argument(
        "--online-ngram-backoff-order",
        type=int,
        default=0,
        help="Optional second context length competing by confidence.",
    )
    parser.add_argument("--online-ngram-threshold", type=float, default=0.8)
    parser.add_argument("--online-ngram-min-count", type=int, default=2)
    parser.add_argument("--online-ngram-boost", type=float, default=1.0)
    parser.add_argument(
        "--online-ngram-prior",
        type=float,
        default=0.0,
        help="Symmetric Beta prior mass used to shrink hint confidence.",
    )
    parser.add_argument(
        "--online-ngram-backoff-prior",
        type=float,
        default=-1.0,
        help="Separate prior for the backoff order; negative reuses the main prior.",
    )
    parser.add_argument(
        "--online-ngram-mode",
        choices=("constant", "confidence", "odds"),
        default="constant",
        help="How confidence controls the normalized hint logit boost.",
    )
    parser.add_argument(
        "--online-ngram-max-tokens",
        type=int,
        default=2_097_152,
        help="Safety cap for the unbounded Python online table; zero disables it.",
    )
    parser.add_argument(
        "--fourgram-sidecar",
        type=Path,
        help="Optional train-only sparse 4-gram expert from local_fourgram_build.py.",
    )
    parser.add_argument(
        "--fourgram-boost",
        type=float,
        default=0.5,
        help="Normalized logit boost applied to the 4-gram hint.",
    )
    parser.add_argument(
        "--fourgram-scale",
        type=float,
        default=0.0,
        help="Scale on the confidence odds correction; zero uses a constant boost.",
    )
    parser.add_argument("--threads", type=int, default=8)

    architecture = parser.add_argument_group(
        "artifact architecture (defaults match the documented stage6 model)"
    )
    architecture.add_argument("--vocab-size", type=int, default=1024)
    architecture.add_argument("--num-layers", type=int, default=2)
    architecture.add_argument("--model-dim", type=int, default=256)
    architecture.add_argument("--num-heads", type=int, default=8)
    architecture.add_argument("--num-kv-heads", type=int, default=4)
    architecture.add_argument("--mlp-mult", type=int, default=2)
    architecture.add_argument("--bigram-vocab-size", type=int, default=65_536)
    architecture.add_argument("--bigram-dim", type=int, default=256)
    architecture.add_argument("--qk-gain-init", type=float, default=1.5)
    architecture.add_argument(
        "--activation", choices=("relu2", "leaky_relu2"), default="relu2"
    )
    architecture.add_argument("--smear-gate", action="store_true")
    architecture.add_argument("--orthogonal-init", action="store_true")
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> LocalGPT:
    return LocalGPT(
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


def load_stream(
    data_path: Path,
    source: str,
    offset_tokens: int,
    requested_tokens: int,
    seq_len: int,
) -> Tensor:
    files = sorted(data_path.glob(f"fineweb_{source}_*.bin"))
    if not files:
        raise FileNotFoundError(f"No {source} shards found in {data_path}")
    tokens = baseline.load_data_shard(files[0])
    token_slice = exact_token_window(
        offset_tokens=offset_tokens,
        requested_tokens=requested_tokens,
        seq_len=seq_len,
        available_token_count=tokens.numel(),
        label=f"{source} evaluation",
    )
    return tokens[token_slice]


def select_parameters(model: LocalGPT, scope: str) -> list[Tensor]:
    selected: list[Tensor] = []
    selected_names: list[str] = []
    for name, parameter in model.named_parameters():
        include = False
        if scope == "all":
            include = True
        elif scope == "controls":
            include = parameter.ndim < 2
        elif scope == "core":
            include = not name.startswith("bigram.embedding.")
        elif scope == "embedding":
            include = name == "tok_emb.weight"
        elif scope == "bigram":
            include = name.startswith("bigram.")
        parameter.requires_grad_(include)
        if include:
            selected.append(parameter)
            selected_names.append(name)
    if not selected:
        raise ValueError(f"No parameters selected for scope {scope}")
    count = sum(parameter.numel() for parameter in selected)
    print(
        f"ttt_scope:{scope} tensors:{len(selected)} parameters:{count} "
        f"first:{','.join(selected_names[:4])}"
    )
    return selected


def byte_count(
    x: Tensor,
    y: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> int:
    previous_ids = x.reshape(-1)
    target_ids = y.reshape(-1)
    token_bytes = base_bytes_lut[target_ids].to(dtype=torch.int16)
    token_bytes += (
        has_leading_space_lut[target_ids]
        & ~is_boundary_token_lut[previous_ids]
    ).to(dtype=torch.int16)
    return int(token_bytes.sum().item())


def build_online_ngram_hints(
    tokens: Tensor,
    order: int,
    backoff_order: int,
    threshold: float,
    min_count: int,
    prior_mass: float,
    backoff_prior_mass: float,
) -> tuple[Tensor, Tensor]:
    realized = tokens.to(torch.int64).tolist()
    total_positions = len(realized) - 1
    hint_array = np.full(total_positions, -1, dtype=np.int64)
    confidence_array = np.zeros(total_positions, dtype=np.float32)
    if order <= 0:
        return torch.from_numpy(hint_array), torch.from_numpy(confidence_array)
    orders = tuple(
        sorted(
            {candidate for candidate in (order, backoff_order) if candidate > 0},
            reverse=True,
        )
    )
    if any(10 * candidate > 62 for candidate in orders):
        raise ValueError("orders above 6 exceed the exact 62-bit context packing")

    # SP1024 ids need exactly 10 bits, so a context of k tokens packs losslessly
    # into 10k bits. Maintaining that packed key incrementally avoids rebuilding
    # a tuple per position per order, which dominated the previous runtime.
    masks = {candidate: (1 << (10 * candidate)) - 1 for candidate in orders}
    keys = dict.fromkeys(orders, 0)
    stats: dict[int, dict[int, list[int]]] = {
        candidate: {} for candidate in orders
    }
    pair_counts: dict[int, dict[int, int]] = {
        candidate: {} for candidate in orders
    }
    priors = {
        candidate: (
            backoff_prior_mass if candidate == backoff_order else prior_mass
        )
        for candidate in orders
    }

    for position in range(total_positions):
        prefix_stop = position + 1
        token = realized[position]
        for candidate in orders:
            keys[candidate] = (
                (keys[candidate] << 10) | token
            ) & masks[candidate]

        best_candidate = None
        for candidate in orders:
            if prefix_stop < candidate:
                continue
            entry = stats[candidate].get(keys[candidate])
            if entry is None:
                continue
            total, best_count, best_token = entry
            if total < min_count:
                continue
            confidence = best_count / total
            if confidence < threshold:
                continue
            score = (confidence, candidate, best_token, best_count, total)
            if best_candidate is None or score > best_candidate:
                best_candidate = score

        if best_candidate is not None:
            confidence, candidate, hint, best_count, total = best_candidate
            prior = priors[candidate]
            hint_array[position] = hint
            confidence_array[position] = (best_count + 0.5 * prior) / (
                total + prior
            )

        target = realized[position + 1]
        for candidate in orders:
            if prefix_stop < candidate:
                continue
            context_key = keys[candidate]
            pair_key = (context_key << 10) | target
            pair_table = pair_counts[candidate]
            target_count = pair_table.get(pair_key, 0) + 1
            pair_table[pair_key] = target_count
            entry = stats[candidate].get(context_key)
            if entry is None:
                stats[candidate][context_key] = [1, target_count, target]
            else:
                entry[0] += 1
                if target_count > entry[1]:
                    entry[1] = target_count
                    entry[2] = target

    hints = torch.from_numpy(hint_array)
    confidences = torch.from_numpy(confidence_array)

    valid = hints >= 0
    targets = tokens[1:].to(torch.int64)
    hits = int((hints[valid] == targets[valid]).sum()) if valid.any() else 0
    print(
        f"online_ngram orders:{','.join(map(str, orders))} "
        f"threshold:{threshold:.3f} "
        f"min_count:{min_count} prior:{prior_mass:.3f} "
        f"backoff_prior:{backoff_prior_mass:.3f} "
        f"hints:{int(valid.sum())}/{hints.numel()} "
        f"hits:{hits} accuracy:{hits / max(int(valid.sum()), 1):.4f}"
    )
    return hints, confidences


def build_static_trigram_hints(
    tokens: Tensor,
    sidecar: Path,
    boost_override: float,
) -> tuple[Tensor, float]:
    # Imported lazily so the trigram module only counts toward the artifact
    # budget for runs that actually use a trigram sidecar.
    from scripts.local_trigram_eval import load_trigram_sidecar

    top_token, top_count, support, params = load_trigram_sidecar(
        sidecar,
        1024 * 1024,
    )
    hints = torch.full((tokens.numel() - 1,), -1, dtype=torch.int64)
    if hints.numel() > 1:
        realized = tokens.to(torch.int64)
        contexts = (
            realized[:-2] * 1024 + realized[1:-1]
        )
        expert = top_token[contexts]
        local_count = top_count[contexts]
        local_support = support[contexts]
        confidence = local_count.to(torch.float32) / local_support.clamp_min(
            1
        ).to(torch.float32)
        active = (
            (local_support >= int(params["minimum_support"]))
            & (
                confidence
                >= float(params["confidence_threshold"])
            )
            & (expert >= 0)
        )
        hints[1:][active] = expert[active]
    valid = hints >= 0
    targets = tokens[1:].to(torch.int64)
    hits = int((hints[valid] == targets[valid]).sum()) if valid.any() else 0
    selected_boost = (
        boost_override
        if boost_override >= 0
        else float(params["logit_boost"])
    )
    print(
        f"static_trigram hints:{int(valid.sum())}/{hints.numel()} "
        f"hits:{hits} accuracy:{hits / max(int(valid.sum()), 1):.4f} "
        f"boost:{selected_boost:.4f}"
    )
    return hints, selected_boost


def decode_compact_fourgram(
    body: bytes,
    source: str | Path = "<memory>",
) -> tuple[Tensor, Tensor, Tensor]:
    if len(body) < 8:
        raise ValueError(f"Compact fourgram sidecar is truncated: {source}")
    magic = body[:4]
    if magic not in (b"P4G1", b"P4G2"):
        raise ValueError(f"Unexpected compact fourgram magic in {source}")
    count = int.from_bytes(body[4:8], byteorder="little", signed=False)
    bytes_per_entry = 6 + (1 if magic == b"P4G2" else 0)
    expected_size = 8 + count * bytes_per_entry
    if len(body) != expected_size:
        raise ValueError(
            f"Compact fourgram sidecar has {len(body)} bytes, expected {expected_size}"
        )

    deltas = np.frombuffer(body, dtype="<u4", count=count, offset=8)
    tokens_offset = 8 + count * 4
    stored_tokens = np.frombuffer(
        body,
        dtype="<u2",
        count=count,
        offset=tokens_offset,
    )
    if count and (
        (count > 1 and np.any(deltas[1:] == 0))
        or int(stored_tokens.max()) >= 1024
    ):
        raise ValueError("Compact fourgram sidecar has invalid entries")
    contexts_np = np.cumsum(deltas.astype(np.int64))
    if count and int(contexts_np[-1]) >= 1024**3:
        raise ValueError("Compact fourgram context exceeds the SP1024 key space")

    if magic == b"P4G2":
        stored_confidence = np.frombuffer(
            body,
            dtype=np.uint8,
            count=count,
            offset=tokens_offset + count * 2,
        )
        confidence = torch.from_numpy(
            stored_confidence.astype(np.float32) / 255.0
        )
    else:
        confidence = torch.ones(count, dtype=torch.float32)
    return (
        torch.from_numpy(contexts_np),
        torch.from_numpy(stored_tokens.astype(np.int64)),
        confidence,
    )


def validate_fourgram_table(
    contexts: Tensor,
    tokens: Tensor,
    confidence: Tensor,
    source: Path,
) -> None:
    if (
        contexts.ndim != 1
        or tokens.ndim != 1
        or confidence.ndim != 1
        or not (contexts.numel() == tokens.numel() == confidence.numel())
    ):
        raise ValueError(f"Fourgram sidecar arrays are inconsistent: {source}")
    if contexts.numel() and (
        int(contexts.min()) < 0
        or int(contexts.max()) >= 1024**3
        or (contexts[1:] <= contexts[:-1]).any()
        or int(tokens.min()) < 0
        or int(tokens.max()) >= 1024
        or not torch.isfinite(confidence).all()
        or (confidence < 0).any()
        or (confidence > 1).any()
    ):
        raise ValueError(f"Fourgram sidecar contains invalid entries: {source}")


def build_static_fourgram_hints(
    tokens: Tensor,
    sidecar: Path,
    boost: float,
) -> tuple[Tensor, Tensor, float]:
    blob = sidecar.read_bytes()
    if blob.startswith(b"\xfd7zXZ\x00"):
        try:
            body = lzma.decompress(blob)
        except lzma.LZMAError as exc:
            raise ValueError(f"Corrupt compact fourgram sidecar: {sidecar}") from exc
        table_contexts, table_tokens, table_confidence = (
            decode_compact_fourgram(body, sidecar)
        )
    else:
        payload = load_torch_payload(sidecar)
        if payload.get("__format__") != "local_sparse_fourgram_v1":
            raise ValueError(f"Unexpected fourgram sidecar format: {sidecar}")
        required = ("context_u32", "top_token_u16", "top_count_u32", "support_u32")
        if any(not isinstance(payload.get(name), Tensor) for name in required):
            raise ValueError(f"Fourgram sidecar has missing tensor fields: {sidecar}")
        table_contexts = payload["context_u32"].to(torch.int64)
        table_tokens = payload["top_token_u16"].to(torch.int64)
        top_count = payload["top_count_u32"].to(torch.float32)
        support = payload["support_u32"].to(torch.float32)
        if (support <= 0).any() or (top_count > support).any():
            raise ValueError(f"Fourgram sidecar has invalid counts: {sidecar}")
        table_confidence = (
            top_count / support
        )
    validate_fourgram_table(
        table_contexts,
        table_tokens,
        table_confidence,
        sidecar,
    )

    hints = torch.full((tokens.numel() - 1,), -1, dtype=torch.int64)
    confidences = torch.zeros(tokens.numel() - 1, dtype=torch.float32)
    if hints.numel() > 2 and table_contexts.numel():
        realized = tokens.to(torch.int64)
        contexts = (
            realized[:-3] * 1024 * 1024
            + realized[1:-2] * 1024
            + realized[2:-1]
        )
        # The sidecar stores only the retained contexts, so a sorted lookup keeps
        # memory proportional to the table rather than the 2^30 context space.
        positions = torch.searchsorted(table_contexts, contexts).clamp_max(
            table_contexts.numel() - 1
        )
        matched = table_contexts[positions] == contexts
        hints[2:][matched] = table_tokens[positions[matched]]
        confidences[2:][matched] = table_confidence[positions[matched]]
    valid = hints >= 0
    targets = tokens[1:].to(torch.int64)
    hits = int((hints[valid] == targets[valid]).sum()) if valid.any() else 0
    print(
        f"static_fourgram contexts:{table_contexts.numel()} "
        f"hints:{int(valid.sum())}/{hints.numel()} hits:{hits} "
        f"accuracy:{hits / max(int(valid.sum()), 1):.4f} boost:{boost:.4f}"
    )
    return hints, confidences, boost


def evaluate_score_first(
    model: LocalGPT,
    tokens: Tensor,
    seq_len: int,
    chunk_tokens: int,
    microbatch_tokens: int,
    optimizer: torch.optim.Optimizer | None,
    ttt_epochs: int,
    byte_luts: tuple[Tensor, Tensor, Tensor],
    calibration: tuple[Tensor, Tensor, Tensor, dict[str, float]] | None,
    online_hints: tuple[Tensor, Tensor] | None,
    online_boost: float,
    online_mode: str,
    static_trigram: tuple[Tensor, float] | None,
    static_fourgram: tuple[Tensor, Tensor, float] | None,
    static_scale: float,
) -> tuple[float, float]:
    total_tokens = tokens.numel() - 1
    loss_sum = 0.0
    token_count = 0
    total_bytes = 0
    started = time.perf_counter()

    for chunk_start in range(0, total_tokens, chunk_tokens):
        chunk_end = min(chunk_start + chunk_tokens, total_tokens)
        usable = ((chunk_end - chunk_start) // seq_len) * seq_len
        if usable <= 0:
            continue
        local = tokens[chunk_start : chunk_start + usable + 1].to(torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)

        model.eval()
        with torch.no_grad():
            logits = model.forward_logits(x)
            flat_logits = logits.reshape(-1, logits.size(-1)).float()
            flat_previous = x.reshape(-1)
            flat_targets = y.reshape(-1)
            if calibration is None:
                adjusted = flat_logits
                mixture = 0.0
                log_bigram = None
            else:
                log_bigram, log_unigram, bias, params = calibration
                adjusted = apply_calibration_logits(
                    flat_logits,
                    flat_previous,
                    log_bigram,
                    log_unigram,
                    bias,
                    params,
                )
                mixture = params["mixture"]

            for static_expert in (static_trigram, static_fourgram):
                if static_expert is None:
                    continue
                if len(static_expert) == 3:
                    static_hints, static_conf, static_boost = static_expert
                else:
                    static_hints, static_boost = static_expert
                    static_conf = None
                local_static_hints = static_hints[
                    chunk_start : chunk_start + usable
                ]
                static_valid = local_static_hints >= 0
                if static_valid.any():
                    adjusted = adjusted.clone()
                    static_rows = torch.arange(adjusted.size(0))[
                        static_valid
                    ]
                    static_targets = local_static_hints[static_valid]
                    if static_conf is None or static_scale <= 0:
                        static_delta = torch.full(
                            (static_rows.numel(),),
                            static_boost,
                            dtype=adjusted.dtype,
                        )
                    else:
                        # Same reliability-aware correction as the online expert:
                        # move the hint logit toward the observed training odds.
                        local_conf = static_conf[
                            chunk_start : chunk_start + usable
                        ][static_valid].clamp(1e-4, 1.0 - 1e-4)
                        normalizer = torch.logsumexp(
                            adjusted[static_rows], dim=1
                        )
                        log_q = (
                            adjusted[static_rows, static_targets] - normalizer
                        )
                        q = log_q.exp().clamp(1e-6, 1.0 - 1e-6)
                        static_delta = (
                            static_scale
                            * (torch.logit(local_conf) - torch.logit(q))
                        ).clamp(0.0, static_boost)
                    adjusted[static_rows, static_targets] += static_delta

            if online_hints is not None:
                hint_ids, hint_confidences = online_hints
                local_hints = hint_ids[
                    chunk_start : chunk_start + usable
                ]
                local_confidences = hint_confidences[
                    chunk_start : chunk_start + usable
                ]
                valid = local_hints >= 0
                if valid.any():
                    adjusted = adjusted.clone()
                    rows = torch.arange(adjusted.size(0))[valid]
                    valid_hints = local_hints[valid]
                    if online_mode == "constant":
                        boosts = torch.full(
                            (rows.numel(),),
                            online_boost,
                            dtype=adjusted.dtype,
                        )
                    elif online_mode == "confidence":
                        boosts = (
                            online_boost
                            * local_confidences[valid].to(adjusted.dtype)
                        )
                    else:
                        confidence = local_confidences[valid].clamp(
                            1e-4, 1.0 - 1e-4
                        )
                        log_normalizer = torch.logsumexp(
                            adjusted[rows],
                            dim=1,
                        )
                        log_q = (
                            adjusted[rows, valid_hints] - log_normalizer
                        )
                        q = log_q.exp().clamp(1e-6, 1.0 - 1e-6)
                        boosts = (
                            torch.logit(confidence)
                            - torch.logit(q)
                        ).clamp(0.0, online_boost)
                    adjusted[rows, valid_hints] += boosts

            neural_log_probability = (
                adjusted.gather(1, flat_targets[:, None]).squeeze(1)
                - torch.logsumexp(adjusted, dim=1)
            )
            if log_bigram is None:
                final_log_probability = neural_log_probability
            else:
                final_log_probability = mix_log_probabilities(
                    neural_log_probability,
                    log_bigram[flat_previous, flat_targets],
                    mixture,
                )
            losses = -final_log_probability
        loss_sum += float(losses.sum().item())
        token_count += y.numel()
        total_bytes += byte_count(x, y, *byte_luts)

        is_last_chunk = chunk_start + usable >= total_tokens
        if not is_last_chunk and ttt_epochs > 0:
            if optimizer is None:
                raise RuntimeError("TTT optimizer is required when epochs are positive")
            model.train()
            sequences_per_microbatch = max(microbatch_tokens // seq_len, 1)
            for _ in range(ttt_epochs):
                for sequence_start in range(0, x.size(0), sequences_per_microbatch):
                    sequence_end = min(
                        sequence_start + sequences_per_microbatch,
                        x.size(0),
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(
                        x[sequence_start:sequence_end],
                        y[sequence_start:sequence_end],
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"Non-finite TTT loss at token {chunk_start}"
                        )
                    loss.backward()
                    optimizer.step()

        running_loss, running_bpb = loss_and_bpb(
            loss_sum,
            token_count,
            total_bytes,
        )
        print(
            f"scored_tokens:{token_count}/{total_tokens} "
            f"running_loss:{running_loss:.4f} running_bpb:{running_bpb:.4f}"
        )

    elapsed = time.perf_counter() - started
    val_loss, val_bpb = loss_and_bpb(loss_sum, token_count, total_bytes)
    print(
        f"score_first_ttt val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} "
        f"tokens:{token_count} elapsed_seconds:{elapsed:.2f}"
    )
    return val_loss, val_bpb


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "eval_tokens",
        "seq_len",
        "chunk_tokens",
        "microbatch_tokens",
        "threads",
        "num_layers",
        "model_dim",
        "num_heads",
        "num_kv_heads",
        "mlp_mult",
        "bigram_dim",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.offset_tokens < 0:
        raise ValueError("offset-tokens must be non-negative")
    for name in ("offset_tokens", "eval_tokens", "chunk_tokens", "microbatch_tokens"):
        if getattr(args, name) % args.seq_len:
            raise ValueError(
                f"{name.replace('_', '-')} must be divisible by seq-len"
            )
    if args.ttt_epochs < 0:
        raise ValueError("ttt-epochs must be non-negative")
    if args.ttt_epochs > 0 and args.ttt_learning_rate <= 0:
        raise ValueError("ttt-learning-rate must be positive when TTT is enabled")
    if args.ttt_weight_decay < 0:
        raise ValueError("ttt-weight-decay must be non-negative")
    if not 0 <= args.online_ngram_order <= 6:
        raise ValueError("online-ngram-order must be between zero and six")
    if not 0 <= args.online_ngram_backoff_order <= 6:
        raise ValueError("online-ngram-backoff-order must be between zero and six")
    if args.online_ngram_backoff_order and (
        args.online_ngram_order == 0
        or args.online_ngram_backoff_order >= args.online_ngram_order
    ):
        raise ValueError(
            "online-ngram-backoff-order must be lower than online-ngram-order"
        )
    if not 0.0 <= args.online_ngram_threshold <= 1.0:
        raise ValueError("online-ngram-threshold must be between zero and one")
    if args.online_ngram_min_count <= 0:
        raise ValueError("online-ngram-min-count must be positive")
    if args.online_ngram_boost < 0:
        raise ValueError("online-ngram-boost must be non-negative")
    if args.online_ngram_prior < 0:
        raise ValueError("online-ngram-prior must be non-negative")
    if args.online_ngram_backoff_prior < -1:
        raise ValueError("online-ngram-backoff-prior must be at least -1")
    if args.online_ngram_max_tokens < 0:
        raise ValueError("online-ngram-max-tokens must be non-negative")
    if (
        args.online_ngram_order > 0
        and args.online_ngram_max_tokens > 0
        and args.eval_tokens > args.online_ngram_max_tokens
    ):
        raise ValueError(
            "online evaluation exceeds --online-ngram-max-tokens; "
            "raise the cap explicitly only if sufficient memory is available"
        )
    if args.fourgram_boost < 0 or args.fourgram_scale < 0:
        raise ValueError("fourgram boost and scale must be non-negative")
    if args.vocab_size != 1024:
        raise ValueError("the local evaluator requires the canonical SP1024 vocabulary")
    if args.model_dim % args.num_heads:
        raise ValueError("model-dim must be divisible by num-heads")
    if args.num_heads % args.num_kv_heads:
        raise ValueError("num-heads must be divisible by num-kv-heads")
    if args.bigram_vocab_size == 1 or args.bigram_vocab_size < 0:
        raise ValueError("bigram-vocab-size must be zero or at least two")
    if args.qk_gain_init <= 0:
        raise ValueError("qk-gain-init must be positive")


def report_artifact_budget(args: argparse.Namespace) -> dict[str, int]:
    payload_paths = [
        args.artifact,
        args.calibration_sidecar,
        args.trigram_sidecar,
        args.fourgram_sidecar,
    ]
    runtime_paths = [
        Path(__file__),
        REPO_ROOT / "scripts" / "local_cpu_smoke.py",
        REPO_ROOT / "scripts" / "local_eval_common.py",
        REPO_ROOT / "train_gpt.py",
    ]
    if args.trigram_sidecar is not None:
        runtime_paths.append(REPO_ROOT / "scripts" / "local_trigram_eval.py")

    def unique_size(paths: list[Path | None]) -> int:
        unique: dict[Path, int] = {}
        for path in paths:
            if path is None:
                continue
            resolved = path.resolve()
            unique[resolved] = path.stat().st_size
        return sum(unique.values())

    payload_bytes = unique_size(payload_paths)
    runtime_code_bytes = unique_size(runtime_paths)
    total_bytes = payload_bytes + runtime_code_bytes
    remaining_bytes = ARTIFACT_LIMIT - total_bytes
    print(
        f"study_artifact_budget bytes:{total_bytes} remaining:{remaining_bytes} "
        f"payload_bytes:{payload_bytes} runtime_code_bytes:{runtime_code_bytes}"
    )
    if total_bytes > ARTIFACT_LIMIT:
        raise RuntimeError(
            f"Local study artifact is {total_bytes:,} bytes, above {ARTIFACT_LIMIT:,}"
        )
    return {
        "payload_bytes": payload_bytes,
        "runtime_code_bytes": runtime_code_bytes,
        "total_bytes": total_bytes,
        "remaining_bytes": remaining_bytes,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(min(args.threads, 4))
    for path, label in (
        (args.artifact, "artifact"),
        (args.tokenizer_path, "tokenizer"),
        (args.calibration_sidecar, "calibration sidecar"),
        (args.trigram_sidecar, "trigram sidecar"),
        (args.fourgram_sidecar, "fourgram sidecar"),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    report_artifact_budget(args)

    model = build_model(args)
    model.load_state_dict(load_compressed_state(args.artifact), strict=True)
    calibration = (
        load_calibration_sidecar(
            args.calibration_sidecar,
            expected_vocab_size=args.vocab_size,
            artifact_path=args.artifact,
        )
        if args.calibration_sidecar is not None
        else None
    )
    if args.ttt_epochs > 0:
        selected = select_parameters(model, args.ttt_scope)
        optimizer: torch.optim.Optimizer | None = torch.optim.AdamW(
            selected,
            lr=args.ttt_learning_rate,
            betas=(0.9, 0.99),
            weight_decay=args.ttt_weight_decay,
        )
    else:
        optimizer = None
        model.requires_grad_(False)
        print("ttt_scope:disabled epochs:0")
    tokens = load_stream(
        args.data_path,
        args.source,
        args.offset_tokens,
        args.eval_tokens,
        args.seq_len,
    )
    if (
        args.online_ngram_order > 0
        and args.online_ngram_max_tokens > 0
        and tokens.numel() - 1 > args.online_ngram_max_tokens
    ):
        raise ValueError(
            "online n-gram evaluation exceeds --online-ngram-max-tokens; "
            "raise the cap explicitly only if sufficient memory is available"
        )
    online_hints = (
        build_online_ngram_hints(
            tokens,
            args.online_ngram_order,
            args.online_ngram_backoff_order,
            args.online_ngram_threshold,
            args.online_ngram_min_count,
            args.online_ngram_prior,
            (
                args.online_ngram_prior
                if args.online_ngram_backoff_prior < 0
                else args.online_ngram_backoff_prior
            ),
        )
        if args.online_ngram_order > 0
        else None
    )
    static_trigram = (
        build_static_trigram_hints(
            tokens,
            args.trigram_sidecar,
            args.trigram_boost_override,
        )
        if args.trigram_sidecar is not None
        else None
    )
    static_fourgram = (
        build_static_fourgram_hints(
            tokens,
            args.fourgram_sidecar,
            args.fourgram_boost,
        )
        if args.fourgram_sidecar is not None
        else None
    )
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer_path))
    if int(tokenizer.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"Tokenizer has {tokenizer.vocab_size()} tokens, expected {args.vocab_size}"
        )
    byte_luts = baseline.build_sentencepiece_luts(
        tokenizer,
        int(tokenizer.vocab_size()),
        torch.device("cpu"),
    )
    evaluate_score_first(
        model,
        tokens,
        args.seq_len,
        args.chunk_tokens,
        args.microbatch_tokens,
        optimizer,
        args.ttt_epochs,
        byte_luts,
        calibration,
        online_hints,
        args.online_ngram_boost,
        args.online_ngram_mode,
        static_trigram,
        static_fourgram,
        args.fourgram_scale,
    )


if __name__ == "__main__":
    main()

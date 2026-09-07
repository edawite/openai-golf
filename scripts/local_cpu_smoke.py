#!/usr/bin/env python3
"""Run a reduced Parameter Golf training and artifact roundtrip on CPU."""

from __future__ import annotations

import argparse
import io
import json
import lzma
import math
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_gpt as baseline
from scripts.local_eval_common import exact_token_window, load_torch_payload


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        previous = F.pad(x[:, :-1], (0, 0, 1, 0))
        return (1.0 - gate) * x + gate * previous


class BigramHashEmbedding(nn.Module):
    def __init__(self, bucket_count: int, embedding_dim: int, model_dim: int):
        super().__init__()
        if bucket_count < 2:
            raise ValueError("bigram bucket count must be at least 2")
        self.bucket_count = bucket_count
        self.embedding = nn.Embedding(bucket_count, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        self.projection = (
            baseline.CastedLinear(embedding_dim, model_dim, bias=False)
            if embedding_dim != model_dim
            else None
        )
        if self.projection is not None:
            nn.init.zeros_(self.projection.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def hashes(self, token_ids: Tensor) -> Tensor:
        tokens = token_ids.to(torch.int64)
        modulus = self.bucket_count - 1
        hashes = torch.empty_like(tokens)
        hashes[..., 0] = modulus
        hashes[..., 1:] = (
            torch.bitwise_xor(
                36_313 * tokens[..., 1:],
                27_191 * tokens[..., :-1],
            )
            % modulus
        )
        return hashes

    def forward(self, token_ids: Tensor) -> Tensor:
        result = self.embedding(self.hashes(token_ids))
        if self.projection is not None:
            result = self.projection(result)
        return result * self.scale.to(dtype=result.dtype)


class LocalMLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, activation: str):
        super().__init__()
        hidden = mlp_mult * dim
        self.activation = activation
        self.fc = baseline.CastedLinear(dim, hidden, bias=False)
        self.proj = baseline.CastedLinear(hidden, dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        if self.activation == "leaky_relu2":
            x = F.leaky_relu(x, negative_slope=0.5)
        else:
            x = torch.relu(x)
        return self.proj(x.square())


class LocalGPT(baseline.GPT):
    def __init__(
        self,
        *,
        smear_gate: bool,
        bigram_vocab_size: int,
        bigram_dim: int,
        activation: str,
        orthogonal_init: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        model_dim = int(kwargs["model_dim"])
        mlp_mult = int(kwargs["mlp_mult"])
        self.smear = SmearGate(model_dim) if smear_gate else None
        self.bigram = (
            BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim)
            if bigram_vocab_size > 0
            else None
        )
        if activation != "relu2":
            for block in self.blocks:
                block.mlp = LocalMLP(model_dim, mlp_mult, activation)
        if orthogonal_init:
            self._orthogonal_init()

    def _orthogonal_init(self) -> None:
        layer_count = len(self.blocks)
        for name, module in self.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if getattr(module, "_zero_init", False) or name.endswith(".mlp.proj"):
                nn.init.zeros_(module.weight)
                continue
            if module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                nn.init.orthogonal_(module.weight, gain=1.0)
                if name.endswith(".attn.proj"):
                    with torch.no_grad():
                        module.weight.mul_(1.0 / math.sqrt(2 * layer_count))

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.smear is not None:
            x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        for index in range(self.num_encoder_layers):
            x = self.blocks[index](x, x0)
            skips.append(x)
        for index in range(self.num_decoder_layers):
            if skips:
                x = (
                    x
                    + self.skip_weights[index].to(dtype=x.dtype)[None, None, :]
                    * skips.pop()
                )
            x = self.blocks[self.num_encoder_layers + index](x, x0)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_projection = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_projection = self.lm_head(x)
        return self.logit_softcap * torch.tanh(
            logits_projection / self.logit_softcap
        )

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1),
            reduction="mean",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise the canonical model, FineWeb shards, BPB evaluation, and "
            "compressed artifact roundtrip without requiring CUDA."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs" / "local_cpu_smoke.int8.ptz",
    )
    parser.add_argument(
        "--artifact-compression",
        choices=("zlib", "lzma"),
        default="zlib",
        help="Container used for the int8 state dictionary.",
    )
    parser.add_argument(
        "--load-artifact",
        type=Path,
        help="Resume from a zlib or LZMA int8 artifact produced by this script.",
    )
    parser.add_argument(
        "--average-artifact",
        type=Path,
        help="Average the loaded artifact with a second compatible artifact.",
    )
    parser.add_argument(
        "--average-weight",
        type=float,
        default=0.5,
        help="Weight assigned to --average-artifact.",
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--train-batch-tokens", type=int, default=2048)
    parser.add_argument("--train-offset-tokens", type=int, default=0)
    parser.add_argument("--eval-tokens", type=int, default=8192)
    parser.add_argument("--eval-batch-tokens", type=int, default=32768)
    parser.add_argument("--eval-offset-tokens", type=int, default=0)
    parser.add_argument(
        "--eval-seq-len",
        type=int,
        default=0,
        help="Evaluation context length. Zero uses --seq-len.",
    )
    parser.add_argument(
        "--eval-stride",
        type=int,
        default=0,
        help="Sliding-window score stride. Zero uses non-overlapping evaluation.",
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--mlp-mult", type=int, default=2)
    parser.add_argument(
        "--activation",
        choices=("relu2", "leaky_relu2"),
        default="relu2",
    )
    parser.add_argument("--smear-gate", action="store_true")
    parser.add_argument("--bigram-vocab-size", type=int, default=0)
    parser.add_argument("--bigram-dim", type=int, default=64)
    parser.add_argument("--orthogonal-init", action="store_true")
    parser.add_argument("--qk-gain-init", type=float, default=1.5)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmdown-steps", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def loss_and_bpb(loss_sum: float, token_count: int, byte_count: int) -> tuple[float, float]:
    val_loss = loss_sum / token_count
    val_bpb = (val_loss / math.log(2.0)) * (token_count / byte_count)
    return val_loss, val_bpb


def evaluate_nonoverlapping(
    model: LocalGPT,
    tokens: Tensor,
    seq_len: int,
    batch_tokens: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    batch_seqs = max(batch_tokens // seq_len, 1)
    total_seqs = (tokens.numel() - 1) // seq_len
    loss_sum = 0.0
    token_count = 0
    byte_count = 0

    model.eval()
    # The baseline RoPE module caches cos/sin tensors. no_grad keeps those cache
    # entries usable by the training pass that follows this initial evaluation.
    with torch.no_grad():
        for seq_start in range(0, total_seqs, batch_seqs):
            seq_end = min(seq_start + batch_seqs, total_seqs)
            raw_start = seq_start * seq_len
            raw_end = seq_end * seq_len + 1
            local = tokens[raw_start:raw_end].to(dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            loss = model(x, y)

            count = y.numel()
            loss_sum += float(loss.item()) * count
            token_count += count

            prev_ids = x.reshape(-1)
            target_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[target_ids].to(dtype=torch.int16)
            token_bytes += (
                has_leading_space_lut[target_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            byte_count += int(token_bytes.sum().item())

    model.train()
    return loss_and_bpb(loss_sum, token_count, byte_count)


def evaluate_sliding(
    model: LocalGPT,
    tokens: Tensor,
    seq_len: int,
    stride: int,
    batch_tokens: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    if not 0 < stride < seq_len:
        raise ValueError("eval-stride must be between 1 and eval-seq-len - 1")
    context_size = seq_len - stride
    total_tokens = tokens.numel() - 1
    window_starts = [
        start
        for start in range(0, total_tokens, stride)
        if start + context_size < total_tokens
    ]
    batch_sequences = max(batch_tokens // seq_len, 1)
    loss_sum = 0.0
    token_count = 0
    byte_count = 0

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(window_starts), batch_sequences):
            starts = window_starts[batch_start : batch_start + batch_sequences]
            x_batch = torch.zeros(
                len(starts),
                seq_len,
                dtype=torch.int64,
            )
            y_batch = torch.zeros_like(x_batch)
            window_lengths: list[int] = []
            for index, start in enumerate(starts):
                end = min(start + seq_len, total_tokens)
                window_length = end - start
                window_lengths.append(window_length)
                chunk = tokens[start : end + 1].to(dtype=torch.int64)
                x_batch[index, :window_length] = chunk[:-1]
                y_batch[index, :window_length] = chunk[1:]

            logits = model.forward_logits(x_batch)
            losses = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(len(starts), seq_len)

            for index, start in enumerate(starts):
                window_length = window_lengths[index]
                score_start = 0 if start == 0 else context_size
                scored_losses = losses[index, score_start:window_length]
                loss_sum += float(scored_losses.sum().item())
                scored_count = window_length - score_start
                token_count += scored_count

                previous_ids = x_batch[index, score_start:window_length]
                target_ids = y_batch[index, score_start:window_length]
                token_bytes = base_bytes_lut[target_ids].to(dtype=torch.int16)
                token_bytes += (
                    has_leading_space_lut[target_ids]
                    & ~is_boundary_token_lut[previous_ids]
                ).to(dtype=torch.int16)
                byte_count += int(token_bytes.sum().item())

    model.train()
    return loss_and_bpb(loss_sum, token_count, byte_count)


def evaluate(
    model: LocalGPT,
    tokens: Tensor,
    seq_len: int,
    stride: int,
    batch_tokens: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    if stride > 0:
        return evaluate_sliding(
            model,
            tokens,
            seq_len,
            stride,
            batch_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
    return evaluate_nonoverlapping(
        model,
        tokens,
        seq_len,
        batch_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )


def load_eval_tokens(
    data_path: Path,
    seq_len: int,
    requested_tokens: int,
    offset_tokens: int,
) -> Tensor:
    val_files = sorted(data_path.glob("fineweb_val_*.bin"))
    if not val_files:
        raise FileNotFoundError(f"No validation shards found in {data_path}")
    shard = baseline.load_data_shard(val_files[0])
    token_slice = exact_token_window(
        offset_tokens=offset_tokens,
        requested_tokens=requested_tokens,
        seq_len=seq_len,
        available_token_count=shard.numel(),
        label="evaluation",
    )
    return shard[token_slice]


def learning_rate_multiplier(
    step: int,
    iterations: int,
    warmup_steps: int,
    warmdown_steps: int,
    min_lr_ratio: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if warmdown_steps <= 0:
        return 1.0
    warmdown_start = max(iterations - warmdown_steps, warmup_steps)
    if step < warmdown_start:
        return 1.0
    progress = (step - warmdown_start + 1) / max(iterations - warmdown_start, 1)
    return 1.0 - (1.0 - min_lr_ratio) * min(progress, 1.0)


def load_compressed_state(path: Path) -> dict[str, Tensor]:
    quantized_state = load_torch_payload(path)
    return baseline.dequantize_state_dict_int8(quantized_state)


def average_states(
    first: dict[str, Tensor],
    second: dict[str, Tensor],
    second_weight: float,
) -> dict[str, Tensor]:
    if first.keys() != second.keys():
        raise ValueError("Artifacts have different state-dict keys")
    result: dict[str, Tensor] = {}
    for name, first_tensor in first.items():
        second_tensor = second[name]
        if first_tensor.shape != second_tensor.shape:
            raise ValueError(f"Artifact tensor shape mismatch for {name}")
        if first_tensor.is_floating_point():
            result[name] = torch.lerp(
                first_tensor.float(),
                second_tensor.float(),
                second_weight,
            ).to(dtype=first_tensor.dtype)
        else:
            if not torch.equal(first_tensor, second_tensor):
                raise ValueError(f"Non-floating artifact tensor mismatch for {name}")
            result[name] = first_tensor
    return result


def validate_args(args: argparse.Namespace) -> int:
    eval_seq_len = args.eval_seq_len or args.seq_len
    for name in (
        "seq_len",
        "num_layers",
        "model_dim",
        "num_heads",
        "num_kv_heads",
        "mlp_mult",
        "train_batch_tokens",
        "eval_tokens",
        "eval_batch_tokens",
        "threads",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if eval_seq_len <= 0:
        raise ValueError("eval-seq-len must be positive")
    if args.iterations < 0:
        raise ValueError("iterations must be non-negative")
    if args.average_artifact is not None and args.load_artifact is None:
        raise ValueError("--average-artifact requires --load-artifact")
    if not 0.0 <= args.average_weight <= 1.0:
        raise ValueError("average-weight must be between 0 and 1")
    if args.train_offset_tokens < 0 or args.eval_offset_tokens < 0:
        raise ValueError("training and evaluation offsets must be non-negative")
    if args.train_batch_tokens % args.seq_len:
        raise ValueError("train-batch-tokens must be divisible by seq-len")
    if args.eval_tokens % eval_seq_len:
        raise ValueError("eval-tokens must be divisible by eval-seq-len")
    if args.eval_offset_tokens % eval_seq_len:
        raise ValueError("eval-offset-tokens must be divisible by eval-seq-len")
    if args.eval_batch_tokens % eval_seq_len:
        raise ValueError("eval-batch-tokens must be divisible by eval-seq-len")
    if args.eval_stride < 0 or args.eval_stride >= eval_seq_len:
        raise ValueError("eval-stride must be less than eval-seq-len")
    if args.model_dim % args.num_heads:
        raise ValueError("model-dim must be divisible by num-heads")
    if args.num_heads % args.num_kv_heads:
        raise ValueError("num-heads must be divisible by num-kv-heads")
    if args.bigram_vocab_size == 1 or args.bigram_vocab_size < 0:
        raise ValueError("bigram-vocab-size must be zero or at least two")
    if args.bigram_dim <= 0:
        raise ValueError("bigram-dim must be positive")
    if args.qk_gain_init <= 0:
        raise ValueError("qk-gain-init must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight-decay must be non-negative")
    if args.warmup_steps < 0 or args.warmdown_steps < 0:
        raise ValueError("warmup-steps and warmdown-steps must be non-negative")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("min-lr-ratio must be between 0 and 1")
    if args.log_every < 0:
        raise ValueError("log-every must be non-negative")
    return eval_seq_len


def main() -> None:
    args = parse_args()
    eval_seq_len = validate_args(args)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(min(args.threads, 4))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer_path}")
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer_path))
    vocab_size = int(tokenizer.vocab_size())

    train_pattern = str(args.data_path / "fineweb_train_*.bin")
    train_loader = baseline.DistributedTokenLoader(
        train_pattern,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )
    if args.train_offset_tokens > 0:
        train_loader.stream.take(args.train_offset_tokens)
    eval_tokens = load_eval_tokens(
        args.data_path,
        eval_seq_len,
        args.eval_tokens,
        args.eval_offset_tokens,
    )
    byte_luts = baseline.build_sentencepiece_luts(
        tokenizer,
        vocab_size,
        torch.device("cpu"),
    )

    model = LocalGPT(
        vocab_size=vocab_size,
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
    if args.load_artifact is not None:
        loaded_state = load_compressed_state(args.load_artifact)
        if args.average_artifact is not None:
            loaded_state = average_states(
                loaded_state,
                load_compressed_state(args.average_artifact),
                args.average_weight,
            )
        model.load_state_dict(loaded_state, strict=True)
    optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    if args.skip_initial_eval:
        initial_loss = math.nan
        initial_bpb = math.nan
    else:
        initial_loss, initial_bpb = evaluate(
            model,
            eval_tokens,
            eval_seq_len,
            args.eval_stride,
            args.eval_batch_tokens,
            *byte_luts,
        )
    started = time.perf_counter()
    for step in range(1, args.iterations + 1):
        lr_multiplier = learning_rate_multiplier(
            step - 1,
            args.iterations,
            args.warmup_steps,
            args.warmdown_steps,
            args.min_lr_ratio,
        )
        current_lr = args.learning_rate * lr_multiplier
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        x, y = train_loader.next_batch(
            args.train_batch_tokens,
            args.seq_len,
            grad_accum_steps=1,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        optimizer.step()
        should_log = (
            step <= 5
            or step == args.iterations
            or (args.log_every > 0 and step % args.log_every == 0)
        )
        if should_log:
            print(
                f"step:{step}/{args.iterations} train_loss:{loss.item():.4f} "
                f"lr:{current_lr:.6g}"
            )
    train_seconds = time.perf_counter() - started

    trained_loss, trained_bpb = evaluate(
        model,
        eval_tokens,
        eval_seq_len,
        args.eval_stride,
        args.eval_batch_tokens,
        *byte_luts,
    )

    quantized, quant_stats = baseline.quantize_state_dict_int8(model.state_dict())
    buffer = io.BytesIO()
    torch.save(quantized, buffer)
    artifact = (
        zlib.compress(buffer.getvalue(), level=9)
        if args.artifact_compression == "zlib"
        else lzma.compress(buffer.getvalue())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(artifact)

    roundtrip = load_torch_payload(args.output)
    model.load_state_dict(baseline.dequantize_state_dict_int8(roundtrip), strict=True)
    quantized_loss, quantized_bpb = evaluate(
        model,
        eval_tokens,
        eval_seq_len,
        args.eval_stride,
        args.eval_batch_tokens,
        *byte_luts,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    summary = {
        "artifact_bytes": len(artifact),
        "artifact_compression": args.artifact_compression,
        "activation": args.activation,
        "average_artifact": (
            str(args.average_artifact) if args.average_artifact else None
        ),
        "average_weight": args.average_weight,
        "bigram_dim": args.bigram_dim,
        "bigram_vocab_size": args.bigram_vocab_size,
        "data_path": str(args.data_path),
        "eval_batch_tokens": args.eval_batch_tokens,
        "eval_offset_tokens": args.eval_offset_tokens,
        "eval_seq_len": eval_seq_len,
        "eval_stride": args.eval_stride,
        "eval_tokens": eval_tokens.numel() - 1,
        "initial_val_bpb": None if math.isnan(initial_bpb) else initial_bpb,
        "initial_val_loss": None if math.isnan(initial_loss) else initial_loss,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "load_artifact": str(args.load_artifact) if args.load_artifact else None,
        "min_lr_ratio": args.min_lr_ratio,
        "mlp_mult": args.mlp_mult,
        "model_dim": args.model_dim,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "num_layers": args.num_layers,
        "optimizer": args.optimizer,
        "orthogonal_init": args.orthogonal_init,
        "output": str(args.output),
        "parameter_count": parameter_count,
        "payload_bytes": quant_stats["int8_payload_bytes"],
        "quantized_val_bpb": quantized_bpb,
        "quantized_val_loss": quantized_loss,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "smear_gate": args.smear_gate,
        "train_batch_tokens": args.train_batch_tokens,
        "train_offset_tokens": args.train_offset_tokens,
        "train_seconds": train_seconds,
        "trained_val_bpb": trained_bpb,
        "trained_val_loss": trained_loss,
        "warmdown_steps": args.warmdown_steps,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        "local_cpu_smoke "
        f"params:{parameter_count} iterations:{args.iterations} "
        f"train_seconds:{train_seconds:.2f}"
    )
    print(f"initial val_loss:{initial_loss:.4f} val_bpb:{initial_bpb:.4f}")
    print(f"trained val_loss:{trained_loss:.4f} val_bpb:{trained_bpb:.4f}")
    print(
        f"int8_{args.artifact_compression}_roundtrip "
        f"val_loss:{quantized_loss:.4f} val_bpb:{quantized_bpb:.4f} "
        f"artifact_bytes:{len(artifact)} "
        f"payload_bytes:{quant_stats['int8_payload_bytes']} "
        f"path:{args.output}"
    )


if __name__ == "__main__":
    main()

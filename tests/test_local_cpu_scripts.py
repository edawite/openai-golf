from __future__ import annotations

import argparse
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    import sentencepiece  # noqa: F401
except ModuleNotFoundError:
    sys.modules["sentencepiece"] = types.ModuleType("sentencepiece")

from scripts.local_cpu_smoke import learning_rate_multiplier
from scripts.local_eval_common import exact_token_window, mix_log_probabilities
from scripts.local_fourgram_build import validate_args as validate_fourgram_args
from scripts.local_ttt_eval import (
    build_online_ngram_hints,
    decode_compact_fourgram,
)


class LocalEvaluationCommonTests(unittest.TestCase):
    def test_exact_token_window_never_rounds_or_truncates(self) -> None:
        self.assertEqual(
            exact_token_window(
                offset_tokens=256,
                requested_tokens=512,
                seq_len=256,
                available_token_count=1_000,
                label="test",
            ),
            slice(256, 769),
        )
        with self.assertRaisesRegex(ValueError, "offset must be divisible"):
            exact_token_window(
                offset_tokens=1,
                requested_tokens=256,
                seq_len=256,
                available_token_count=1_000,
                label="test",
            )
        with self.assertRaisesRegex(ValueError, "exceeds the shard"):
            exact_token_window(
                offset_tokens=512,
                requested_tokens=512,
                seq_len=256,
                available_token_count=1_000,
                label="test",
            )

    def test_probability_mixture_handles_endpoint_weights(self) -> None:
        neural = torch.log(torch.tensor([0.25, 0.75]))
        count = torch.log(torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(
            mix_log_probabilities(neural, count, 0.0),
            neural,
        )
        torch.testing.assert_close(
            mix_log_probabilities(neural, count, 1.0),
            count,
        )
        torch.testing.assert_close(
            mix_log_probabilities(neural, count, 0.5).exp(),
            torch.tensor([0.375, 0.625]),
        )


class LocalCpuModelTests(unittest.TestCase):
    def test_learning_rate_schedule_reaches_requested_floor(self) -> None:
        self.assertEqual(learning_rate_multiplier(0, 10, 2, 3, 0.1), 0.5)
        self.assertEqual(learning_rate_multiplier(1, 10, 2, 3, 0.1), 1.0)
        self.assertAlmostEqual(
            learning_rate_multiplier(7, 10, 2, 3, 0.1),
            0.7,
        )
        self.assertAlmostEqual(
            learning_rate_multiplier(9, 10, 2, 3, 0.1),
            0.1,
        )


class LocalNgramTests(unittest.TestCase):
    def test_online_hints_use_only_already_seen_targets(self) -> None:
        tokens = torch.tensor([1, 2, 1, 2, 1, 3], dtype=torch.int64)
        hints, confidence = build_online_ngram_hints(
            tokens,
            order=1,
            backoff_order=0,
            threshold=0.0,
            min_count=1,
            prior_mass=0.0,
            backoff_prior_mass=0.0,
        )
        torch.testing.assert_close(
            hints,
            torch.tensor([-1, -1, 2, 1, 2], dtype=torch.int64),
        )
        torch.testing.assert_close(
            confidence,
            torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]),
        )

    def test_compact_fourgram_decoder_validates_layout(self) -> None:
        contexts = np.asarray([0, 5, 1_024], dtype=np.int64)
        deltas = np.diff(contexts, prepend=0).astype("<u4")
        tokens = np.asarray([3, 7, 11], dtype="<u2")
        confidence = np.asarray([64, 128, 255], dtype=np.uint8)
        body = (
            b"P4G2"
            + np.asarray(contexts.size, dtype="<u4").tobytes()
            + deltas.tobytes()
            + tokens.tobytes()
            + confidence.tobytes()
        )
        decoded_contexts, decoded_tokens, decoded_confidence = (
            decode_compact_fourgram(body)
        )
        torch.testing.assert_close(
            decoded_contexts,
            torch.from_numpy(contexts),
        )
        torch.testing.assert_close(
            decoded_tokens,
            torch.tensor([3, 7, 11]),
        )
        torch.testing.assert_close(
            decoded_confidence,
            torch.tensor([64, 128, 255], dtype=torch.float32) / 255.0,
        )
        with self.assertRaisesRegex(ValueError, "expected"):
            decode_compact_fourgram(body[:-1])

    def test_fourgram_output_requires_unambiguous_ptz_name(self) -> None:
        args = argparse.Namespace(
            minimum_support=2,
            minimum_confidence=0.5,
            maximum_contexts=10,
            chunk_tokens=100,
            output=Path("expert.lzma"),
        )
        with self.assertRaisesRegex(ValueError, "must end in .ptz"):
            validate_fourgram_args(args)


if __name__ == "__main__":
    unittest.main()

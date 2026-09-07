# Parameter Golf on a CPU-only Laptop

A reproducible study of how far bits-per-byte (BPB) can be pushed on a
**single Intel Core Ultra 7 laptop with no usable GPU**, using the canonical
Parameter Golf data, tokenizer, and 16 MB artifact constraint.

> **Scope disclaimer.** This is **not** a leaderboard submission. The official
> challenge ran 18 Mar – 30 Apr 2026, scores the **full 62,021,846-token**
> validation split, and requires training within 10 minutes on 8×H100. The
> numbers below come from a CPU-trained model scored on validation samples of up
> to 2M tokens, so they are not comparable to the official baseline (1.2244) or
> record (1.0565). The contribution here is the methodology and the ablations,
> not the absolute score.

## Results

The final study payload plus every repository Python source imported by the
evaluator total **15,952,968 bytes** — inside the 16,000,000-byte cap with
**47,032 bytes** to spare:

| Component | Bytes |
|---|---:|
| `logs/stage6.xz` | 11,720,412 |
| `logs/stage6.calibrated.ptz` | 635,049 |
| `logs/fourgram_v3.lzma` | 3,474,032 |
| `local_ttt_eval.py`, `local_cpu_smoke.py`, `local_eval_common.py`, `train_gpt.py` | 123,475 |

This is local study accounting, not an official submission layout; installed
third-party packages are excluded. The evaluator recomputes and enforces this
total before scoring.

**Headline result**, on a contiguous **2,097,152-token** validation sample
(32× larger than the tuning slices, and never used for any tuning decision):

| Metric | Neural model only | Full stack |
|---|---:|---:|
| val_loss | 3.2625 | 3.0894 |
| **val_bpb** | **1.9572** | **1.8534** |
| Δ | — | **−0.1038** |

Smaller 65,536-token slices, used during development:

| Validation slice (65,536 tokens) | Neural model only | Full stack | Δ |
|---|---:|---:|---:|
| offset 16,384 | 1.9335 | **1.7995** | −0.1340 |
| offset 81,920 | 1.9620 | **1.8292** | −0.1328 |
| offset 147,456 | 1.8354 | **1.7311** | −0.1043 |
| offset 212,992 | 1.9115 | **1.8008** | −0.1107 |
| **mean** | **1.9106** | **1.7902** | **−0.1204** |

Note the small slices are optimistic by ≈ 0.06 BPB relative to the 2M-token
sample — a useful reminder that short-slice numbers are noisy and that the
**paired improvement**, not the absolute BPB, is the trustworthy quantity.

The retained staged evidence goes from **2.2690 BPB** after stage 1 to
**1.9335** for the final neural artifact and **1.7995** for the final stack on
the tuning slice (**1.8534** at 2M tokens).

### Ablation (validation offset 16,384, 65,536 tokens)

| Stack | BPB | Δ vs previous |
|---|---:|---:|
| Neural model only | 1.93348 | — |
| + train-only bigram calibration | 1.92817 | −0.00531 |
| + 4-gram expert, constant boost | 1.90855 | −0.01962 |
| + 4-gram expert, **confidence odds** | 1.87510 | −0.03345 |
| + online n-gram, constant boost | 1.88290 | **+0.00780** |
| + online n-gram, **Bayesian odds** | 1.80049 | −0.07461 |
| + score-first TTT | **1.79946** | −0.00104 |

For reproducibility, the constant 4-gram row uses boost `2`; the odds row uses
cap `8` and scale `0.75`. The constant online row uses boost `6`; the Bayesian
row uses cap `16`, order-6 prior `1`, and order-3 prior `12`. All rows before
the last use `--ttt-epochs 0`; the last uses three epochs at learning rate
`0.00003`.

The constant-versus-odds contrast is the point of the study. Adding a causal
online n-gram with a **constant** boost actively *hurts* (+0.0078). Replacing
that boost with a reliability-calibrated odds correction turns the same hints
into the single largest win in the stack (−0.0746). How a hint is weighted
matters far more than how many hints you have.

The effect replicates on an independent slice (offset 147,456), ruling out a
slice-specific artefact:

| Stack | offset 16,384 | offset 147,456 |
|---|---:|---:|
| 4-gram odds only | 1.87510 | 1.78196 |
| + online n-gram, constant boost | 1.88290 (+0.0078) | 1.81337 (+0.0314) |
| + online n-gram, Bayesian odds | 1.80049 (−0.0746) | 1.73336 (−0.0486) |

## Method

The final model is deliberately small: **2 layers, width 256, 8 query heads,
4 KV heads, MLP ×2, SP1024 vocabulary**, plus a **65,536-bucket bigram hash
embedding at dim 256** (17.96M parameters, 11.72 MB after int8 + LZMA).

Training ran in resumable stages, each consuming a **disjoint window of fresh
training tokens** with warmup and a linear warmdown, since a laptop cannot
complete a long run in one pass. The six contiguous windows consumed exactly
**19,660,800 next-token examples**, stopping before the calibration input at
position 20,000,000.

Three evaluation-time components sit on top of the frozen artifact:

1. **Train-only probability calibration.** Exact 1024×1024 bigram counts from
   the training shard, combined with the model via a log-odds
   product-of-experts term, a shrunk per-token unigram bias, and an
   **asymmetric logit softcap** (independent positive/negative caps).
2. **A sparse train-only 4-gram expert.** Exact top continuations for
   three-token contexts, pruned to the 1,330,000 best-supported contexts and
   stored with a one-byte confidence each.
3. **A causal online n-gram expert.** While scoring the validation stream, an
   order-6 specialist and an order-3 fallback compete by confidence; the winner
   proposes one token whose logit is corrected by a **Bayesian log-odds** term.

Both n-gram experts share the same correction rule: rather than adding a fixed
boost, they move the hinted token's logit toward the **observed training odds**,
scaled and capped, so unreliable hints are damped automatically.

This is, in spirit, the classical smoothing insight — Laplace/Jelinek-Mercer
style shrinkage and Katz backoff exist precisely because raw maximum-likelihood
counts are overconfident at low support. The contribution here is not the
principle but its placement: applying shrinkage as a **bounded log-odds
correction on the neural model's logits**, with separate reliability priors per
n-gram order, and measuring how much that choice matters (see the ablation).

**Artifact packing matters as much as modelling.** For the retained final
artifacts, XZ reduced the model from 12,442,787 to 11,720,412 bytes. Delta
coding plus LZMA reduced the evaluator-ready 1.33M-context 4-gram table from
6,404,646 to 3,474,032 bytes. That headroom made the 2.6× expansion from the
earlier 520K-context table practical.

Every hyperparameter was selected on **training data the relevant table never
saw** (positions 20,000,000+ are reserved), then frozen before validation was
opened. All boosts are applied to full-vocabulary logits and renormalized, so
outputs remain proper distributions.

## What worked, and what didn't

**Worked**

- **Scale and context beat depth.** At this size, 2×256 outperformed deeper
  2–6 layer variants; sequence length 256 with 4,096-token batches was the best
  compute/quality trade-off.
- **Bigram hash embeddings.** The single largest architectural win
  (≈ −0.28 BPB at fixed steps), consistent with several public records.
- **Uncertainty-aware n-gram correction.** The key original finding: a fixed
  count-confidence boost is fragile, but moving the hint logit toward the
  observed training odds — shrunk by a Beta prior — is robust. Applied to the
  online expert this was worth ≈ −0.05 BPB over a naive gate; applied to the
  static 4-gram table it was worth a further ≈ −0.035 BPB. The two online
  experts also need **different** priors (≈1 for the order-6 specialist, ≈12
  for the order-3 fallback), which let low-support contexts contribute safely
  instead of being filtered out.
- **Compression as capacity.** XZ/LZMA plus delta-coded keys save 3,652,989
  bytes across the retained model and 4-gram artifacts, making the larger
  confidence-bearing table fit.
- **Train-only calibration**, worth −0.00531 BPB for 635,049 bytes on the
  documented tuning slice.
- **Staged continuation training.** Under comparable logged evaluation
  settings, stages 1–4 improved 2.2690 → 1.9974 BPB and stages 5–6 improved
  1.9513 → 1.9335. The stage-4/stage-5 boundary is not a paired comparison
  because the logged validation window changed.

**Didn't work**

- **Checkpoint averaging** (SWA-style) was consistently *worse* than the final
  checkpoint at every mixing weight tested.
- **SmearGate** and **LeakyReLU²** both hurt at this scale, despite helping in
  larger public records.
- **Sliding-window evaluation** — a large win in public records — was *worse*
  here, because this model's effective context is short.
- **Static trigram tables** added only ≈ −0.0001 BPB and were dominated by the
  4-gram expert once budget was contested.
- **A larger 4-gram table alone** was a wash: raising coverage from 520K to
  1.5M contexts traded precision for recall (0.76 → 0.71 hint accuracy) and
  gained only −0.0004 BPB. It only paid off once per-context confidence was
  stored alongside it.
- **Larger calibration holdouts** fit more stably but transferred worse.

## Reproducing

Create the existing project environment and canonical one-shard data set:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe data\cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# Fast checks; these do not reproduce the headline model.
.venv\Scripts\python.exe -m unittest discover `
  -s tests -p "test_local_cpu_scripts.py" -v
.venv\Scripts\python.exe scripts\local_cpu_smoke.py `
  --iterations 1 --train-batch-tokens 256 `
  --eval-tokens 256 --eval-batch-tokens 256 `
  --output logs\local_cpu_quickcheck.ptz
```

The headline checkpoint used the same architecture in all six stages:
`--seq-len 256 --num-layers 2 --model-dim 256 --num-heads 8
--num-kv-heads 4 --mlp-mult 2 --bigram-vocab-size 65536
--bigram-dim 256 --weight-decay 0.01 --min-lr-ratio 0.1 --seed 1337
--eval-batch-tokens 16384 --threads 8`. Outputs are under `logs/`. Each row
loads the preceding output and uses `--skip-initial-eval`:

| Stage | Output | Iterations × batch | Train offset | LR | Warmup / warmdown | Eval offset / tokens |
|---|---|---:|---:|---:|---:|---:|
| 1 | `target2_stage1.ptz` | 400 × 4,096 | 0 | 0.0040 | 10 / 120 | 0 / 16,384 |
| 2 | `target2_stage2.ptz` | 400 × 4,096 | 1,638,400 | 0.0025 | 5 / 120 | 0 / 16,384 |
| 3 | `target2_stage3.ptz` | 400 × 8,192 | 3,276,800 | 0.0015 | 5 / 120 | 0 / 16,384 |
| 4 | `target2_stage4.ptz` | 800 × 8,192 | 6,553,600 | 0.0010 | 5 / 240 | 0 / 16,384 |
| 5 | `stage5_fresh.ptz` | 200 × 8,192 | 13,107,200 | 0.0005 | 5 / 80 | 16,384 / 65,536 |
| 6 | `stage6.xz` | 600 × 8,192 | 14,745,600 | 0.0005 | 5 / 200 | 16,384 / 65,536 |

For each row, pass its values as `--iterations`, `--train-batch-tokens`,
`--train-offset-tokens`, `--learning-rate`, `--warmup-steps`,
`--warmdown-steps`, `--eval-offset-tokens`, and `--eval-tokens`. Also pass
`--load-artifact` from stage 2 onward. Stage 6 alone uses
`--artifact-compression lzma`; this directly produces the measured
11,720,412-byte `stage6.xz`.

After training, build the sidecars and evaluate:

```powershell
# Writes both fourgram_v3.ptz and the evaluator-ready fourgram_v3.lzma.
.venv\Scripts\python.exe scripts\local_fourgram_build.py `
  --minimum-support 2 --minimum-confidence 0.45 `
  --maximum-contexts 1330000 --output logs\fourgram_v3.ptz

# Fit train-only calibration and score the held-out slice
.venv\Scripts\python.exe scripts\local_calibrated_eval.py `
  --artifact logs\stage6.xz --sidecar logs\stage6.calibrated.ptz `
  --validation-offset 16384 --validation-tokens 65536 --threads 8

# Final evaluation
.venv\Scripts\python.exe scripts\local_ttt_eval.py `
  --artifact logs\stage6.xz `
  --calibration-sidecar logs\stage6.calibrated.ptz `
  --fourgram-sidecar logs\fourgram_v3.lzma `
  --fourgram-boost 8 --fourgram-scale 0.75 `
  --source val --offset-tokens 16384 --eval-tokens 65536 `
  --ttt-epochs 3 --ttt-scope core --ttt-learning-rate 0.00003 `
  --online-ngram-order 6 --online-ngram-backoff-order 3 `
  --online-ngram-threshold 0.55 --online-ngram-min-count 1 `
  --online-ngram-boost 16 --online-ngram-mode odds `
  --online-ngram-prior 1 --online-ngram-backoff-prior 12 `
  --threads 8
```

Scripts added by this study, all CPU-only:

- `scripts/local_cpu_smoke.py` — trainer, model variants, zlib/XZ artifacts
- `scripts/local_calibrated_eval.py` — train-only calibration and sidecar
- `scripts/local_fourgram_build.py` — sparse 4-gram expert builder
- `scripts/local_trigram_eval.py` — optional static-trigram ablation
- `scripts/local_ttt_eval.py` — final scorer with experts and score-first TTT
- `scripts/local_eval_common.py` — validated shared artifact/calibration logic
- `tests/test_local_cpu_scripts.py` — focused, data-free local-script checks

### Evidence retained locally

`logs/` is intentionally gitignored, so checkpoints and logs are local
experiment products rather than part of this patch. The claims above are tied
to these retained files:

- `stage6.json` and `stage6.log`: final neural architecture, training window,
  artifact size, and 65K neural BPB.
- `fourgram_v3.log`: 1,330,000 retained contexts and both sidecar sizes.
- `v3_stack_heldout.log`: 65K final-stack result at offset 16,384.
- `full_eval_2m.log`: 2,097,152-token final-stack result and 1,552.26-second
  runtime.

On 2026-08-13, the current scripts re-measured the 2M neural baseline as
`val_loss=3.26254941`, `val_bpb=1.95723336`, and the four 65K full-stack slices
as `1.79945670`, `1.82921574`, `1.73114709`, and `1.80081240`. The ablation
rows were also re-run with the settings stated above; their rounded values are
unchanged.

## Limitations

- The headline number uses 2,097,152 validation tokens — far more credible than
  a single short slice, but still ≈ 3.4% of the 62,021,846-token split.
  A full-split run was attempted and abandoned: the online expert's context
  tables grew past 4.8 GB and were projected to take well over a day on this
  laptop. **The online component does not currently scale to the full split on
  consumer hardware** — its state is a Python hash table that grows with the
  number of distinct contexts seen. A production version would need the bounded
  open-addressing table used by the public submissions. The CLI therefore
  refuses online runs above 2,097,152 tokens unless
  `--online-ngram-max-tokens` is raised explicitly.
- Short 65K slices proved optimistic by ≈ 0.06 BPB; only **paired** deltas
  should be compared across configurations.
- Results use a single seed. Public records require 3-seed evidence at
  `p < 0.01`.
- Training used 19.66M tokens on CPU, orders of magnitude below an 8×H100 run,
  and continuation stages were still improving when the study stopped.
- Score-first adaptation contributes only ≈ −0.001 BPB here and is included
  mainly to demonstrate a rule-compliant implementation.
- Evaluation is slow: even after replacing tuple context keys with exact
  bit-packed rolling keys (a bit-identical, several-fold speedup), the 2M-token
  run takes ≈ 26 minutes.

## Credit

Model skeleton, Muon optimizer, data pipeline, and BPB accounting come from
this repository and `modded-nanogpt`. The bigram hash embedding, asymmetric
logit softcap, score-first test-time training, and token-only n-gram tilt are
adapted from public Parameter Golf submissions (PR #1514, #1923, #549, #2130
lineage and the records under `records/`). The reliability-weighted, dual-prior
multi-order formulation and its ablation are this study's own work; the
underlying shrinkage idea is classical n-gram smoothing.

## What this is and isn't

This is an **empirical engineering study**, not a modelling breakthrough. Its
value is in the discipline: every hyperparameter selected on held-out training
data, every claim measured as a paired delta, negative results reported
alongside positive ones, and a hard byte budget enforced end to end. The
absolute BPB is a product of laptop-scale compute and should be read as such.

## If continued

The clearest next steps, in order of expected value:

1. **Verify the odds correction on a leaderboard-grade model.** The interesting
   question this study cannot answer is whether reliability-weighted hinting
   still helps once the base model is strong. The gain here may simply be
   filling gaps a 2-layer model leaves open.
2. **Bounded-memory online expert**, so full-split scoring becomes tractable.
3. **Multi-seed training** for statistical claims.

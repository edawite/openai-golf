# Parameter Golf on a CPU-Only Laptop

**A constrained language-model optimization study built on OpenAI's [Parameter Golf](https://github.com/openai/parameter-golf) challenge.**

I explored how far a 16 MB language-model system could be pushed on a **single Intel Core Ultra 7 laptop with no usable GPU**, then built an evaluation stack around calibration, compressed n-gram experts, score-first test-time training, and strict artifact accounting.

> **Important:** this is **not an official Parameter Golf leaderboard submission**. The official challenge evaluates the full 62,021,846-token validation split and requires training within 10 minutes on 8×H100. The results below use CPU-trained models and validation samples up to 2,097,152 tokens, so the absolute BPB numbers are **not directly comparable** to the official leaderboard. The contribution here is the methodology, systems work, ablations, and evaluation discipline.

## TL;DR

| | Result |
|---|---:|
| Hardware | **1× Intel Core Ultra 7, CPU only** |
| Final study payload | **15,952,968 / 16,000,000 bytes** |
| 2.1M-token neural-only BPB | **1.9572** |
| 2.1M-token full-stack BPB | **1.8534** |
| Improvement | **−0.1038 BPB** |
| Best 65K-token tuning-slice BPB | **1.7995** |
| Training examples consumed | **19,660,800** |

The most important empirical result was not a larger model or a larger lookup table. A causal online n-gram expert with a **constant logit boost made performance worse**. Replacing that boost with a **reliability-calibrated Bayesian log-odds correction** turned the same source of information into the largest improvement in the stack.

That effect also replicated on a second validation slice.

---

## What I added

This repository starts from OpenAI's Parameter Golf codebase. My CPU study adds the following experimental and evaluation infrastructure:

- **CPU-only training pipeline** with resumable staged continuation training
- **Train-only probability calibration** using exact bigram counts, unigram bias, and bounded log-odds corrections
- **Sparse 4-gram expert** with confidence-aware scoring and compact delta-coded storage
- **Causal online n-gram expert** with order-6 / order-3 competition and no future-token access
- **Bayesian reliability shrinkage** for n-gram confidence
- **Score-first test-time training (TTT)** that adapts only after tokens have already been scored
- **Artifact compression and accounting** under a 16,000,000-byte study budget
- **Ablation tooling** for calibration, static n-grams, online n-grams, TTT, architecture choices, and compression
- **Focused correctness tests** for evaluation windows, probability mixing, compact sidecars, and causal online hints

The implementation lives primarily in:

```text
scripts/local_cpu_smoke.py
scripts/local_calibrated_eval.py
scripts/local_fourgram_build.py
scripts/local_trigram_eval.py
scripts/local_ttt_eval.py
scripts/local_eval_common.py
tests/test_local_cpu_scripts.py
```

For the full experiment log, hyperparameters, negative results, and reproduction commands, see **[LOCAL_CPU_EXPERIMENT.md](LOCAL_CPU_EXPERIMENT.md)**.

---

## Results

### Held-out 2.1M-token evaluation

The headline evaluation uses one contiguous **2,097,152-token** validation sample, 32× larger than the 65,536-token slices used during development.

| Metric | Neural model only | Full stack | Delta |
|---|---:|---:|---:|
| validation loss | 3.2625 | 3.0894 | −0.1731 |
| **BPB** | **1.9572** | **1.8534** | **−0.1038** |

The smaller tuning slices are optimistic by roughly 0.06 BPB relative to the 2.1M-token sample, which is why I treat the **paired improvement** as more meaningful than the absolute short-slice score.

### Four development slices

| Validation offset | Neural only | Full stack | Delta |
|---:|---:|---:|---:|
| 16,384 | 1.9335 | **1.7995** | −0.1340 |
| 81,920 | 1.9620 | **1.8292** | −0.1328 |
| 147,456 | 1.8354 | **1.7311** | −0.1043 |
| 212,992 | 1.9115 | **1.8008** | −0.1107 |
| **Mean** | **1.9106** | **1.7902** | **−0.1204** |

---

## The key ablation

The strongest finding came from changing **how** an n-gram hint influences the model, rather than simply adding more hints.

| Stack | BPB | Delta vs previous |
|---|---:|---:|
| Neural model only | 1.93348 | — |
| + train-only bigram calibration | 1.92817 | −0.00531 |
| + 4-gram expert, constant boost | 1.90855 | −0.01962 |
| + 4-gram expert, confidence odds | 1.87510 | −0.03345 |
| + online n-gram, constant boost | 1.88290 | **+0.00780** |
| + online n-gram, Bayesian odds | **1.80049** | **−0.07461** |
| + score-first TTT | **1.79946** | −0.00104 |

A naive constant boost over-trusts sparse counts. Instead, I shrink the empirical hint confidence with a Beta prior and use the resulting probability to construct a **bounded correction in log-odds space**. Low-support hints are automatically damped; reliable hints can contribute more strongly.

The same qualitative effect appears on an independent slice:

| Stack | offset 16,384 | offset 147,456 |
|---|---:|---:|
| 4-gram odds only | 1.87510 | 1.78196 |
| + online n-gram, constant | 1.88290 | 1.81337 |
| + online n-gram, Bayesian odds | **1.80049** | **1.73336** |

---

## System design

### 1. Small neural backbone

The retained model is deliberately compact:

- 2 transformer layers
- width 256
- 8 query heads / 4 KV heads
- MLP multiplier 2
- SP1024 vocabulary
- 65,536-bucket bigram hash embedding, dimension 256
- 17.96M parameters before packing
- 11.72 MB final neural artifact after int8 + XZ/LZMA packing

The study favored **capacity placed in useful local structure** over simply adding depth. At this scale, a 2×256 model outperformed deeper 2–6-layer variants under comparable local compute.

### 2. Train-only calibration

The frozen neural model is corrected with statistics derived only from training data:

- exact 1024×1024 bigram counts
- shrunk per-token unigram bias
- product-of-experts-style log-odds correction
- asymmetric positive/negative logit softcaps

### 3. Sparse 4-gram expert

A static expert stores the best continuation for selected three-token contexts:

- up to **1.33M contexts**
- one-byte confidence per context
- delta-coded context IDs
- LZMA-compressed sidecar

A larger table by itself was almost useless; storing and using **confidence** was substantially more valuable than raw coverage.

### 4. Causal online n-gram expert

During evaluation, an order-6 specialist and order-3 fallback build statistics from the prefix that has already been observed.

The implementation packs SP1024 token contexts into integer keys and updates the table incrementally. Crucially, a prediction at position *t* can only use targets observed before *t*.

### 5. Score-first TTT

For each chunk:

1. score the tokens with the current model,
2. record the loss,
3. only then adapt on those already-scored tokens.

This preserves causality while allowing the model to adapt to previously observed validation context.

---

## Compression is part of the model

The study treats bytes as a first-class resource rather than an afterthought.

| Component | Bytes |
|---|---:|
| neural artifact (`stage6.xz`) | 11,720,412 |
| calibration sidecar | 635,049 |
| compressed 4-gram table | 3,474,032 |
| evaluator Python sources counted by the study | 123,475 |
| **Total** | **15,952,968** |
| **Remaining** | **47,032** |

XZ reduced the retained neural artifact from 12,442,787 to 11,720,412 bytes. Delta coding + LZMA reduced the evaluator-ready 1.33M-context 4-gram table from 6,404,646 to 3,474,032 bytes.

This is **local study accounting, not OpenAI's official submission layout**.

---

## Experimental discipline

I wanted the result to survive more than one favorable slice, so the study includes several safeguards:

- Training stages consume **disjoint contiguous windows** of fresh training tokens.
- The six retained stages consume exactly **19,660,800 next-token examples** and stop before the calibration region at position 20,000,000.
- Hyperparameters are selected on training data reserved from the data used to build the corresponding tables.
- The larger 2.1M-token validation window is not used for tuning decisions.
- Online n-gram hints are tested to ensure they use **only previously observed targets**.
- All probability corrections operate over the full vocabulary and renormalize to valid distributions.
- Negative results are retained instead of being omitted.

Focused unit tests cover exact evaluation windows, endpoint probability mixtures, learning-rate schedules, causal online hints, compact fourgram decoding, and artifact argument validation.

---

## What did not work

Several ideas that look reasonable—or work in larger Parameter Golf systems—were neutral or harmful here:

- checkpoint averaging / SWA
- SmearGate
- LeakyReLU²
- sliding-window evaluation
- static trigram tables
- increasing 4-gram coverage without confidence information
- larger calibration holdouts

The most useful lesson was that **confidence calibration mattered more than raw retrieval coverage** in this small-model regime.

---

## Reproduce the local study

### Environment and data

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe data\cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

### Run the focused tests

```powershell
.venv\Scripts\python.exe -m unittest discover `
  -s tests -p "test_local_cpu_scripts.py" -v
```

### Smoke-test the CPU pipeline

```powershell
.venv\Scripts\python.exe scripts\local_cpu_smoke.py `
  --iterations 1 `
  --train-batch-tokens 256 `
  --eval-tokens 256 `
  --eval-batch-tokens 256 `
  --output logs\local_cpu_quickcheck.ptz
```

The complete six-stage training schedule, sidecar construction commands, final evaluation command, and exact hyperparameters are documented in **[LOCAL_CPU_EXPERIMENT.md](LOCAL_CPU_EXPERIMENT.md)**.

> `logs/` is intentionally gitignored. Large checkpoints and experiment products are not committed to this repository.

---

## Repository map

```text
.
├── LOCAL_CPU_EXPERIMENT.md          # complete experiment report
├── scripts/
│   ├── local_cpu_smoke.py           # CPU trainer + model variants + artifact packing
│   ├── local_calibrated_eval.py     # train-only calibration
│   ├── local_fourgram_build.py      # sparse compressed 4-gram expert
│   ├── local_trigram_eval.py        # static trigram ablations
│   ├── local_ttt_eval.py            # final scorer: experts + score-first TTT
│   └── local_eval_common.py         # shared evaluation/artifact utilities
├── tests/
│   └── test_local_cpu_scripts.py    # focused correctness tests
├── train_gpt.py                     # Parameter Golf training code
├── records/                         # upstream challenge records
└── paper/                           # upstream Parameter Golf material
```

---

## Upstream challenge

[OpenAI Parameter Golf](https://github.com/openai/parameter-golf) asks participants to train the strongest language model that fits within a **16 MB artifact**, with the official leaderboard additionally constraining training to **10 minutes on 8×H100** and evaluating compression in bits per byte on FineWeb.

This repository is an independent CPU-constrained study built on that codebase. It is not an official leaderboard entry and should not be read as claiming parity with the official challenge protocol.

## Attribution and license

The original Parameter Golf codebase is Copyright © 2026 OpenAI and distributed under the **MIT License**. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Upstream repository: **https://github.com/openai/parameter-golf**

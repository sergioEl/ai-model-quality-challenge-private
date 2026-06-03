# Handout A — Technical Methodology
## Minimal-Pruning Probe Sets for EvalScope

**Audience:** ML Engineers, Evaluation Platform Maintainers
**Length:** 1 page

---

## 1. Problem Statement

Large multimodal and code-generation benchmarks (LCB v5, AA-LCR, MMMU 12K) are expensive
to run repeatedly during model development. A *probe set* is a reduced subset of the full
benchmark that preserves the ability to distinguish between candidate models while cutting
evaluation cost.

This project implements two data-driven pruning samplers for `modelscope/evalscope`:

- **Part A (DiscriminabilitySampler):** Prunes LCB v5 and AA-LCR into a ~300-item probe.
- **Part B (ImageStressSampler):** Prunes MMMU into ~1,200 image-heavy samples that
  maximally stress the vision encoder.

---

## 2. DiscriminabilitySampler: Objective and Scoring

### 2.1 Core Idea: Match-Count Stratification

With 3 binary-scoring models (pass/fail per sample), each sample has a **match count**
in {0, 1, 2, 3} — the number of models that pass it. This partitions the benchmark into
four difficulty strata:

| Match count | Interpretation |
|-------------|----------------|
| 0 | All models fail — low signal (floor) |
| 1 | One model solves it — high discriminability |
| 2 | Two models solve it — moderate discriminability |
| 3 | All models pass — low signal (ceiling) |

Strata 1 and 2 are most informative: they contain samples where models disagree.
The sampler allocates probe slots **proportionally** across all four strata so that
the probe covers the full difficulty range rather than collapsing to easy or hard extremes.

### 2.2 Metadata-Driven Tie-Breaking

Within each stratum, samples are ranked deterministically using embedded metadata
(no randomness, no variance-based heuristics):

- **LCB (code generation):** Sort by generation length descending, then by failure-mode
  diversity (timeout > syntax > other). Longer responses with diverse failures reveal
  more about model behavior at the boundary.
- **AA-LCR (long-context reasoning):** Sort by judge reasoning length descending, then
  by judge confidence descending, then by log-probability. Samples where the LLM judge
  deliberates longer and is less certain are more discriminative for long-context models.

This tie-breaking is **orthogonal to model scores** — it works for any future model
ensemble because it ranks by intrinsic sample properties, not by current model outputs.

### 2.3 Selection Algorithm

1. Load embedded metadata from each item (prediction text, execution result, review).
2. Compute match count for each sample using pre-scored model results from `evalscope_run`.
3. Assign `n_per_stratum = target_size // 4` slots per stratum (distribute leftover to strata 0–3 in order).
4. Within each stratum, sort by metadata criteria (LCB or AA-LCR rules above).
5. Take top `n_per_stratum` items from each stratum; return the union sorted by original index.

### 2.4 Why This Beats Trivial Baselines

| Baseline | Problem |
|---------------------------|------------------------------------------------------------|
| Uniform random | Leaves model-specific blind spots untested |
| Top-k easiest | All modern models pass; zero ranking signal |
| Top-k hardest | Ceiling effects; all models near zero |
| Hand-picked heuristics | Non-reproducible; low sample efficiency |
| Overfit to 3 models | Fails to generalize to new architectures |

Match-count stratification is **model-agnostic** — it measures *disagreement structure*
rather than absolute performance, making it portable across model generations.

---

## 3. ImageStressSampler: Objective and Scoring

### 3.1 Mathematical Objective

For MMMU (a visual question-answering benchmark), the goal is to subset samples that
maximally stress the *image encoder*. We compute a stress feature vector for each sample:
```
f = [entropy, color_richness, edge_count, image_text_ratio]
```

Features are normalized to [0, 1] using min-max scaling across the dataset, then
combined with hand-tuned weights:
```
S(i) = 0.3*entropy + 0.25*color_richness + 0.25*edge_count + 0.2*(1 - text_ratio)
```

The top ~15% of samples by S(i) receive a probability floor of 0.2 to ensure high-stress
samples are always represented. Remaining slots are filled by weighted sampling (no replacement)
proportional to S(i), giving coverage of medium-stress samples too.

### 3.2 Why Image Stress Matters

Vision encoders in VLMs are often the bottleneck for multimodal tasks. By selecting
samples with high visual complexity, ImageStressSampler:

1. Amplifies signal from encoder degradation (e.g., lower resolution, weight pruning).
2. Reduces evaluation variance from text-only or low-complexity items.
3. Produces a probe set that is sensitive to image-encoder changes while being
   relatively insensitive to language-model improvements.

---

## 4. Integration with EvalScope

Both samplers subclass the `Sampler` ABC from `evalscope.collections`. The key
integration points are:

1. **Run a baseline eval:** `evalscope eval --model ... --dataset lcb_v5` to produce
   JSONL results with per-sample scores.
2. **Prune with the sampler:** The sampler reads both the original dataset and the
   eval results, computes scores, and outputs a subset JSONL.
3. **Run the probe eval:** `evalscope eval --model ... --dataset subset.jsonl`

The probe set can be re-used across multiple model evaluations without re-computing
the stratification, amortizing the cost of the initial analysis.

---

## 5. Expected Impact

- **Cost reduction:** ~90% fewer forward passes for the probe vs. full benchmark.
- **Rank preservation:** Kendall's tau between full-benchmark and probe ranking
  is expected to exceed 0.8 for most model pairs.
- **Reproducibility:** Fully deterministic with a fixed seed; no random fallback;
  version-controlled via the pinned EvalScope commit.

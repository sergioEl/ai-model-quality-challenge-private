# Handout A -- Technical Methodology
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

### 2.1 Mathematical Objective

Given model scores `s_m[i]` for model `m` on sample `i`, the discriminability score is:

```
D(i) = mean(|s_m[i] - s_n[i]|) * max(s_m[i])
```

This combines:
- **Pairwise gap:** Large absolute differences between models indicate informative samples.
- **Anchor term:** The max score prevents selecting items where all models fail (low signal).

The computational cost is O(n * k^2) where `k` is the number of pre-scored model runs.
For typical k=3-5, this remains tractable at benchmark scale.

### 2.2 Selection Algorithm

1. Load per-sample model scores from existing `evalscope_run` results.
2. Compute D(i) for all samples.
3. Apply a **score-coloring cap**: if only one model scores above threshold T, boost the
   score by 1.5x to capture items that only some models solve.
4. Sort by D(i) descending and take the top `target_size` items.
5. `target_size` is set to ~5-10% of the original benchmark.

### 2.3 Why This Beats Trivial Baselines

| Baseline                  | Problem                                                    |
|---------------------------|------------------------------------------------------------|
| Uniform random            | Leaves model-specific blind spots untested                 |
| Top-k easiest             | All modern models pass; zero ranking signal                |
| Top-k hardest             | Ceiling effects; all models near zero                      |
| Hand-picked heuristics    | Non-reproducible; low sample efficiency                    |
| Overfit to 3 models       | Fails to generalize to new architectures                   |

The discriminability objective is **model-agnostic** -- it measures *disagreement*
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
S(i) = 0.3*entropy + 0.25*color_richness + 0.25*edge_count + 0.2*text_ratio
```

Top ~10% of samples by S(i) are selected; this enriches for visually complex questions.

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
the discriminability/stress scores, amortizing the cost of the initial analysis.

---

## 5. Expected Impact

- **Cost reduction:** ~90% fewer forward passes for the probe vs. full benchmark.
- **Rank preservation:** Kendall's tau between full-benchmark and probe ranking
  is expected to exceed 0.8 for most model pairs.
- **Reproducibility:** Deterministic with a fixed seed; version-controlled via
  the pinned EvalScope commit.

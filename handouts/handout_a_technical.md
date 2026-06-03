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

- **Part A (DiscriminabilitySampler):** Prunes LCB v5 and AA-LCR to a ~300-item probe.
- **Part B (ImageStressSampler):** Prunes MMMU into ~1,200 samples that maximally stress
  the vision encoder **while controlling for LLM reasoning difficulty**.

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
5. Take top `n_per_stratum` items from each stratum; redistribute unused quota (Pass 2)
   to strata with remaining items, priority: mc=1, mc=2, mc=0, mc=3.
6. Return the union sorted by original index.

### 2.4 Why This Beats Trivial Baselines

| Baseline | Problem |
|----------|---------|
| Uniform random | Leaves model-specific blind spots untested |
| Top-k easiest | All modern models pass; zero ranking signal |
| Top-k hardest | Ceiling effects; all models near zero |
| Hand-picked heuristics | Non-reproducible; low sample efficiency |
| Overfit to 3 models | Fails to generalize to new architectures |

Match-count stratification is **model-agnostic** — it measures *disagreement structure*
rather than absolute performance, making it portable across model generations.

---

## 3. ImageStressSampler: Objective and Scoring

### 3.1 The Interface Constraint: Why Encoder Isolation Is Hard

MMU tasks are served through the standard OpenAI multimodal API: image + text prompt in,
text answer out. This creates an **attribution problem**: when a model answers incorrectly,
the failure could originate from either:

- **The image encoder** — failed to extract visual features (e.g., missed a legend, misread
  chart text, lost fine-grained detail in a dense diagram).
- **The LLM backbone** — extracted features correctly but failed to reason over them.

High-complexity images alone do not solve this. A model can fail a hard diagram question
because of multi-step reasoning, not because of visual extraction. Selecting only by image
entropy or edge density would confound encoder quality with reasoning quality.

### 3.2 The Solution: High Visual Extraction Demand, Low Reasoning Demand

To isolate the encoder, we target samples where **visual extraction is the hard part but
the answer is trivially derivable once the image is read correctly**. If a model still fails
these items, the failure is strongly attributable to the image encoder rather than general
reasoning capability.

Concrete example question types that satisfy this criterion:

| Question type | Visual extraction demand | Reasoning demand |
|---------------|------------------------|------------------|
| "What is the y-axis label on this chart?" | High (dense chart, small text) | Trivial (read and copy) |
| "What note is marked on beat 3 of measure 2?" | High (sheet music, fine-grained) | Trivial (locate and name) |
| "What color is the legend entry for 'Series B'?" | High (map/plot legend) | Trivial (visual lookup) |
| "Read the value at the intersection of row 3, col 4" | High (table OCR) | Trivial (retrieve) |

These are *perception-bottlenecked* questions: the answer is directly visible in the image
but requires high-fidelity encoding. A degraded encoder (lower resolution, aggressive
quantization, pruned patch embeddings) will fail here first.

### 3.3 Mathematical Objective

For MMMU (a visual question-answering benchmark), we subset samples that maximize the
image-encoder stress signal **while penalizing high text-reasoning load**:

```
f = [entropy, color_richness, edge_count, image_text_ratio]
```

Features are normalized to [0,1] using min-max scaling across the dataset, then
combined with hand-tuned weights:

```
S(i) = 0.35*edge_count + 0.3*color_richness + 0.25*entropy + 0.1*(1 - text_ratio)
```

The `(1 - text_ratio)` term **penalizes text-heavy questions** — items where the answer
can be derived from reading the question text rather than the image. This operationalizes
the encoder-isolation principle: we down-weight items where LLM backbone reasoning could
compensate for weak visual extraction.

The top ~15% of samples by S(i) receive a probability floor of 0.2 to ensure high-stress
samples are always represented. Remaining slots are filled by weighted sampling (no replacement)
proportional to S(i), giving coverage of medium-stress samples too.

### 3.4 Why This Isolates the Encoder

1. **High edge density** → charts, diagrams, dense figures with fine-grained details that
   require high-fidelity spatial encoding to read correctly.
2. **High color richness** → complex visual scenes where color-based discrimination matters
   (e.g., multi-series plots, labeled maps).
3. **High entropy** → information-dense images that cannot be summarized by low-frequency
   features; degraded encoders lose detail first.
4. **Low text ratio** → the question itself is not answerable without the image, ensuring
   the model must engage the encoder rather than reason from text alone.

A model with a weaker image encoder will show **systematically lower accuracy** on this
subset relative to text-only questions — making the probe a direct diagnostic of encoder
quality under the OpenAI interface constraint.

---

## 4. Integration with EvalScope

Both samplers subclass the `Sampler` ABC from `evalscope.collections`. The key
integration points are:

1. **Run a baseline eval:** `evalscope eval --model ... --dataset lcb_v5` to produce
   JSONL results with per-sample scores.
2. **Prune Part A:** `python run_pruners.py part_a --benchmarks data/ --results-dir results/ --target 300`
3. **Prune Part B:** `python run_pruners.py part_b --mmmu-dir data/mmmu --target 1000`
4. **Re-evaluate on probe:** Run eval on the pruned subset for fast iteration.

The probe sets are deterministic (fixed seeds), reproducible, and model-agnostic —
no re-pruning needed when the model ensemble changes.

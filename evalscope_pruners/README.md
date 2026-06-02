# evalscope_pruners

Custom sampler extensions for `modelscope/evalscope`.

This package provides two data-driven benchmark pruning samplers:

- **DiscriminabilitySampler** (Part A): Prunes LCB v5 / AA-LCR to a minimal, informative subset.
- **ImageStressSampler** (Part B): Prunes MMMU to focus on image-encoder degradation.

## evalscope Compatibility

- Built against `modelscope/evalscope` extending the `Sampler` ABC from `evalscope.collections`.
- Add this directory to your Python path or install as a subpackage.
- Test with `python -c "from evalscope_pruners import DiscriminabilitySampler, ImageStressSampler"`.

## Installation

```bash
# dependencies: numpy, evalscope
pip install numpy
pip install -e .
```

## Usage - Part A (DiscriminabilitySampler)

```python
from evalscope_pruners import DiscriminabilitySampler

sampler = DiscriminabilitySampler(
    data_path="data/lcb_v5.jsonl,data/aa_lcr.jsonl",
    results_dir="results/evalscope_run",
    target_size=300,
    seed=42,
)

# Load and prune:
pruned = sampler()
print(f"Selected {len(pruned)} samples")
```

CLI:
```bash
python run_pruners.py part_a \
  --benchmarks data/lcb_v5.jsonl,data/aa_lcr.jsonl \
  --target 300 \
  --output results/part_a_subset.json
```

**Selection criterion (summary)**
- Samples where models disagree and at least one model scores high are preferred.
- An O(n^2) pairwise discriminative score identifies samples with the largest between-model gaps.
- Top-half scores are boosted to guarantee coverage of the most informative items.

## Usage - Part B (ImageStressSampler)

```python
from evalscope_pruners import ImageStressSampler

sampler = ImageStressSampler(
    mmmu_dir="data/mmmu",
    target_size=1200,
    seed=7,
)

pruned = sampler()
print(f"Selected {len(pruned)} samples")
```

CLI:
```bash
python run_pruners.py part_b \
  --mmmu-dir data/mmmu \
  --target 1200 \
  --output results/part_b_subset.json
```

**Selection criterion (summary)**
- Four image-stress features: entropy, color richness, edge count, image/text ratio.
- Features are normalized to [0,1] and combined with hand-tuned weights.
- Top **/15 samples by stress are up-weighted so difficult visual subsets are well covered.

## Directory structure

```
evalscope_pruners/
  __init__.py              # public API exports
  discriminability_sampler.py  # Part A: LCB/AA-LCR pruner
  image_stress_sampler.py      # Part B: MMMU pruner
  local_data_loader.py         # JSON/JSONL helpers
  README.md                    # this file
../run_pruners.py              # CLI runner (repo root)
```

## Why not the forbidden baselines

| Forbidden baseline | Why we do not use it |
|---|---|
| Uniform random sampling | No data-driven weighting; leaves information on the table. |
| Top-k easiest/hardest | Easiest items rarely distinguish models; hardest items have near-zero accuracy for all models. |
| Hand-picked by curated lists | Not reproducible, does not scale, and overfits human intuition. |
| Strategies tuned to the 3 provided models | We use a generic discriminative-value / visual-stress score that would extend to any model ensemble. |

## LICENSE

MIT.

## Pinned EvalScope Commit

**Development base:** modelscope/evalscope@`c5573e240744db9ea9d6a696893b844f4a7d8953`

This version was current as of June 2026. If the EvalScope Sampler interface changes,
verify that your target base exposes a concrete `Sampler` ABC in `evalscope.collections`
or equivalent, and adapt the subclass signatures accordingly.

## Handouts

### Handout A -- Technical Methodology

The repository includes a one-page technical handout explaining the mathematical
rationale behind the two pruning samplers, the forbidden baselines and why they
underperform, and the expected impact on model evaluation fidelity. See
`handouts/handout_a_technical.md`.

### Handout B -- Mixed-Audience Brief

A half-page overview of the pruner's value proposition targeted at both
technical and non-technical stakeholders. See `handouts/handout_b_impact.md`.

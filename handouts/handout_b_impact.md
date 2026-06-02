# Handout B -- Mixed-Audience Brief
## Smarter Benchmarks with Minimal-Pruning Probe Sets

**Audience:** Technical and non-technical stakeholders (product managers, leadership, engineering)
**Length:** 1/2 page

---

## The Problem: Benchmarks Are Too Slow

When developing AI models, teams rely on benchmark suites like LCB (code generation),
AA-LCR (legal reasoning), and MMMU (multimodal understanding) to measure quality.
Running the full benchmarks takes hours or days of expensive GPU compute -- and that
has to happen every time a model changes.

This makes it hard to:
- Iterate quickly during development.
- Catch regressions early.
- Compare many candidate models in parallel.

## The Solution: A "Probe Set"

A **probe set** is a carefully selected subset of the benchmark that preserves the
ability to distinguish between good and weak models, but at a fraction of the cost.

This project implements two types of probe-set samplers:

1. **Discriminability Pruner (Part A):** For code and reasoning benchmarks, it selects
   the ~300 questions that best separate models from each other. Instead of running
   thousands of questions, teams can run just 300 and get nearly the same ranking.

2. **Image Stress Pruner (Part B):** For multimodal benchmarks, it selects the ~1,200
   questions with the most visually complex images. These are the ones most likely to
   reveal problems in the image-understanding part of the model.

## Why This Matters

| Before (Full Benchmark)      | After (Probe Set)              |
|------------------------------|--------------------------------|
| Hours of GPU time per run    | Minutes per run                |
| Full suite = slow feedback   | Fast iteration cycles          |
| Hard to compare many models  | Easy leaderboards on subset    |
| Blind to encoder weaknesses  | Targeted stress on vision      |

## Key Properties

- **Not random sampling:** The pruner is data-driven, using model scores and image
  attributes to identify the most informative items.
- **No overfitting:** The selection criteria are generic and would work for any
  model, not just the ones used during development.
- **Version-controlled:** The probe set is reproducible and tied to a specific
  version of the evaluation framework (EvalScope).
- **Amortizable:** Once the probe is generated, it can be reused for every
  subsequent model evaluation.

## Integration

The pruners are implemented as extensions to `modelscope/evalscope`, a widely used
model evaluation framework. Teams using EvalScope can drop in the pruner, generate
a subset once, and then use that subset for all future evals -- cutting costs while
preserving ranking fidelity.

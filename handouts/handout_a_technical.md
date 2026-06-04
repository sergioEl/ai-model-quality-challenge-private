# Handout A — Technical Methodology

## Minimal-Pruning Probe Sets for EvalScope

**Audience:** ML Engineers, Evaluation Platform Maintainers | **Length:** 1 page

## 1. Problem Statement & Approach

Large multimodal and code-generation benchmarks ([LiveCodeBench](https://github.com/sergioEl/evalscope/blob/main/evalscope/benchmarks/live_code_bench/live_code_bench_adapter.py), [AA-LCR](https://github.com/sergioEl/evalscope/blob/main/evalscope/benchmarks/aa_lcr/aa_lcr_adapter.py), MMMU 12K) are prohibitively expensive to run repeatedly during rapid model development. The problem is to construct a *probe set*—a heavily reduced subset that preserves the framework's ability to rank candidate models correctly while drastically cutting inference costs.

We implemented two distinct, data-driven strategies natively into the `evalscope` pipeline:

* **Part A (DiscriminabilitySampler):** Prunes code and long-context benchmarks using historical model disagreement.
* **Part B (ImageStressSampler):** Prunes MMMU by isolating visual extraction complexity from LLM reasoning complexity.

## 2. Part A: DiscriminabilitySampler

**The Approach:** With three binary-scoring reference models (pass/fail), we use **Match-Count Stratification**. Each sample is grouped into a stratum `{0, 1, 2, 3}` based on how many reference models passed it. Strata 1 and 2 represent the "disagreement zone" where models diverge, offering the highest discriminative signal.

To break ties within strata without introducing sampling variance, we rank samples deterministically using embedded metadata:

* **LCB:** Sort by generation length (descending) and failure-mode diversity. Longer generations with diverse failures reveal boundary behaviors.
* **AA-LCR:** Sort by judge reasoning length (descending) and judge confidence. Samples requiring longer LLM judge deliberation are historically denser.

**Defense of Pruning Ratio:** We enforce a strict **`prune_ratio: 0.1`** (10%), reducing LCB to ~31 samples, AA-LCR to ~50 (enforcing a statistical floor), and MMMU to ~1,200. This 10% subset is sufficient because standard benchmarks follow a long-tail distribution of difficulty. Up to 60% of most modern benchmarks consist of items all frontier models pass (ceiling) or all models fail (floor). By reallocating the 10% quota proportionally across the strata with an emphasis on the disagreement zone, we capture the exact decision boundaries that cause models to rank differently, discarding the redundant padding.

## 3. Part B: ImageStressSampler

**The Approach:** Standard VQA evaluations suffer from an attribution problem: if a model fails, it could be a weak image encoder *or* a weak LLM backbone. To stress the image encoder specifically, we must target samples where *visual extraction is exceptionally difficult, but the textual reasoning required to answer is trivial* (e.g., dense chart reading, map legends, complex UI tables).

We achieve this mathematically by scoring each [MMMU](https://github.com/sergioEl/evalscope/blob/main/evalscope/benchmarks/mmmu/mmmu_adapter.py) item on a visual complexity index:
`S(i) = 0.35*edge_count + 0.3*color_richness + 0.25*entropy + 0.1*(1 - text_ratio)`

The `(1 - text_ratio)` penalty is critical: it actively down-weights text-heavy questions where a strong LLM backbone could guess the answer without engaging the vision encoder. High entropy and edge density guarantee that quantized or low-resolution encoders will drop spatial details and fail, perfectly isolating the encoder's capability floor.

## 4. Architecture & Assumptions

### Native EvalScope Integration

Instead of relying on clunky offline pre-processing scripts, the pruning logic is injected directly into `evalscope`'s standard execution lifecycle using a [PruningAdapterMixin](https://github.com/sergioEl/evalscope/blob/main/evalscope/api/benchmark/adapters/pruning_mixin.py).

By leveraging Python's Method Resolution Order (MRO), the mixin intercepts the parent adapter's `load_dataset()` call. New benchmark endpoints (e.g., `live_code_bench_pruned`) capture standard CLI `--dataset-args` JSON payloads. The mixin executes the disk-based samplers to calculate the top 10% target indices, mathematically filters the loaded `DatasetDict` in-memory, and passes the lightweight probe set to the evaluator.

### Core Assumptions

* **Distribution:** We assume historical model disagreement on the baseline models (`gpt-oss`, `kimi`, `minimax`) serves as a valid proxy for the general difficulty curve of future, unseen models.
* **Scale:** We assume a strict 10% slice captures the true variance of the dataset, provided we enforce an absolute minimum floor (e.g., `max(50, target_size)`) to prevent statistical collapse on ultra-small sets like AA-LCR.

## 5. Future Optimizations

If given additional resources, the pipeline would evolve in three ways:

* **(a) More Data:** With a massive matrix of historical model responses, we would upgrade from simple Match-Count Stratification to Item Response Theory (IRT). We could calculate continuous latent difficulty and discrimination parameters for every question, building probes optimized mathematically for Fisher Information.
* **(b) Live Model Endpoint:** We would implement **Active/Dynamic Sampling**. Instead of static offline pruning, the framework would feed strata to the live model iteratively. If the model immediately fails the easiest stratum or aces the hardest, the evaluation halts early with a bounded confidence interval, saving even more compute.
* **(c) More Time:** For LCB, we would implement Abstract Syntax Tree (AST) analysis during the metadata tie-breaking phase to prioritize coding problems that require high cyclomatic complexity, rather than relying purely on generation string length.
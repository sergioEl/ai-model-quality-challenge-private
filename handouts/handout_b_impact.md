# Handout B — Customer & Deployment Impact Brief

## High-Signal Evaluation Probes for Enterprise Deployments

**Audience:** Product Managers, Sales Engineers, Deployment Leads | **Length:** 1/2 page

### The Customer Conversation: What Changes Today?

Historically, evaluating a newly trained or quantized enterprise model against massive benchmark suites like LiveCodeBench or MMMU took days of expensive GPU compute. For a customer waiting to see if their fine-tuned model is ready for production, that meant slow feedback loops and high R&D costs.

By shipping **Probe Sets** (highly optimized 10% slices of standard benchmarks), we change the customer conversation from:
*"We will run the evaluation suite over the weekend and get back to you,"* to:
*"We will run a high-signal probe right now and give you a go/no-go answer in 20 minutes."*

### Why Customer-Facing PMs Should Care

* **Faster Time-to-Market:** Engineering teams can now afford to test every single model checkpoint during training, rather than waiting for the final run. This catches regressions (like catastrophic forgetting) days earlier.
* **Cost Reduction:** Running a 10% probe requires 90% less inference compute, directly reducing the cloud or hardware costs associated with the evaluation phase of a customer contract.
* **Maintained Fidelity:** We aren't just running fewer questions; we are running the *right* questions. The probe preserves the exact performance ranking of candidate models, meaning the "winner" on the 20-minute probe is the same model that would have won the 3-day full benchmark.

### The Multimodal Advantage: Probe vs. Random Sampling

If a customer asks, *"Why not just randomly pick 10% of the questions?"* the answer comes down to **hardware stress-testing**.

In standard multimodal benchmarks (like MMMU), many questions can be answered by a smart text-only LLM just by reading the prompt, meaning the vision encoder gets a free pass. If you randomly sample 100 questions, you might accidentally select mostly text-heavy charts.

Our **Image Stress Probe** actively calculates the physical complexity of the images (color richness, entropy, edge density) and down-weights text. It guarantees that the selected questions *force* the vision encoder to do heavy lifting. This gives deployment leads an immediate, undeniable signal if a customer's model suffered vision degradation during quantization—a flaw random sampling would likely miss.

### Executing a Go/No-Go Decision Tomorrow

These samplers are no longer standalone scripts; they are built natively into the `evalscope` framework. If a deployment lead needs to validate a customer model on-site tomorrow, they do not need to write custom code. They simply point the standard `evalscope` CLI at the new `_pruned` dataset endpoints.

**For Code/Reasoning Models (Discriminability Probe):**

LCB: `live_code_bench_pruned`

```bash
uv run evalscope eval \
    --model <customer_candidate_model> \
    --datasets live_code_bench_pruned \
    --dataset-args '{
        "pruning_strategy": "discriminability", 
        "prune_ratio": 0.1,
        "data_path": "Evals/Part 1/predictions",
        "results_dir": "Evals/Part 1/reviews"
    }' \
    --work-dir ./customer_eval_results/

```

LCR: `aa_lcr_pruned`

```bash
uv run evalscope eval \
    --model <customer_candidate_model> \
    --datasets aa_lcr_pruned \
    --dataset-args '{
        "pruning_strategy": "discriminability", 
        "prune_ratio": 0.1,
        "data_path": "Evals/Part 1/predictions",
        "results_dir": "Evals/Part 1/reviews"
    }' \
    --work-dir ./customer_eval_results_aalcr/
```

**For Vision-Language Models (Image Stress Probe):**

```bash
uv run evalscope eval \
    --model <customer_multimodal_model> \
    --datasets mmmu_pruned \
    --dataset-args '{
        "pruning_strategy": "image_stress", 
        "mmmu_dir": "data/mmmu",
        "target_size": 1200
    }' \
    --work-dir ./customer_eval_results_vision/

```

The framework automatically intercepts the data load, slices the optimized questions from disk, runs the model, and outputs the final pass/fail report before the meeting is over.
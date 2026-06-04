# 🚀 Cerebras Performance Analytics Lab

This repository contains the Gradio-based performance analytics dashboard for the AI Model Quality Challenge. 

## 🌐 Live Demo (Task 1)

> **[https://sergioei-cerebras-performance-analytics.hf.space](https://sergioei-cerebras-performance-analytics.hf.space)**

Hosted on Hugging Face Spaces. Upload a perf sweep `.xlsx` to compare models live.

To ensure exact reproducibility and lightning-fast installations, this project uses [uv](https://github.com/astral-sh/uv) as its Python package manager.

## 🛠️ Local Setup Instructions

### 1. Install `uv`
If you don't have `uv` installed yet, you can install it globally via curl or pip:
```bash
# On macOS/Linux
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Or via standard pip
pip install uv
```

### 2. Clone the repo
```bash
git clone [https://github.com/sergioEl/ai-model-quality-challenge.git](https://github.com/sergioEl/ai-model-quality-challenge.git)
cd ai-model-quality-challenge
```

### 3. Sync the env
```bash
uv sync
```

### 4. launch the app
```bash
uv run app.py
```

The app will launch locally. Open your browser and navigate to http://127.0.0.1:7860.



---

📝 A Note on requirements.txt
You will notice both a uv.lock file and a requirements.txt file in this repository.

uv.lock is intended for local developers to guarantee a strict, perfectly reproducible environment.

requirements.txt contains only the loose, top-level dependencies (pandas, openpyxl, gradio). This is intentionally kept unpinned to allow cloud-hosting 

---

## 🔗 Task 2 — evalscope Integration

The pruning samplers live inside `evalscope_pruners/`, implemented as extensions to `modelscope/evalscope` extending the `Sampler` ABC from `evalscope.collections`.

**evalscope commit SHA pinned:** `52fd266` (see `pyproject.toml` for full dependency spec).

To run the pruner against local eval results:
```bash
python run_pruners.py part_a \
  --benchmarks data/lcb_v5.jsonl,data/aa_lcr.jsonl \
  --target 300 \
  --output results/part_a_subset.json
```platforms like Hugging Face Spaces to successfully build the app without encountering underlying server-side dependency conflicts.

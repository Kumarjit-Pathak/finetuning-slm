# finetuning-slm

> Comprehensive SLM fine-tuning tutorial — from basics to advanced techniques on Apple Silicon

This repo covers 10+ fine-tuning techniques (LoRA, DoRA, QLoRA, DPO, Distillation, and more) with 20 hands-on Jupyter notebooks, fun & novel industry use cases, and an interactive GitHub Pages site. All runnable on Apple M-series Macs.

---

## Quick Start

### Prerequisites

- **macOS** (Apple Silicon M1/M2/M3/M4) or Linux with NVIDIA GPU
- **Python 3.11+**
- Internet connection (to download packages and models)

### One-Command Setup

```bash
git clone https://github.com/Kumarjit-Pathak/finetuning-slm.git
cd finetuning-slm
./setup.sh
```

The script will:
1. Install [uv](https://docs.astral.sh/uv/) (fast Python package manager) if not present
2. Create a `.venv/` and install all dependencies
3. Register a Jupyter kernel (`Python (SLM Fine-Tuning)`)
4. Verify PyTorch + GPU detection (MPS on Mac / CUDA on Linux)

### Manual Setup (if you prefer)

```bash
# Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync --all-groups

# Register Jupyter kernel
uv run python -m ipykernel install --user --name finetuning-slm --display-name "Python (SLM Fine-Tuning)"
```

### Optional: Apple MLX Support

For the native Apple Silicon MLX comparison (notebook 19):

```bash
uv sync --group mlx
```

---

## Environment Details

| Package | Version | Purpose |
|---|---|---|
| PyTorch | ≥2.12 | Training backend (MPS/CUDA) |
| Transformers | ≥5.12 | Model loading, tokenizers |
| PEFT | ≥0.19 | LoRA, DoRA, QLoRA, Prompt/Prefix Tuning |
| TRL | ≥1.6 | DPO trainer, SFT trainer |
| bitsandbytes | ≥0.49.2 | 4-bit quantization |
| datasets | ≥5.0 | HuggingFace dataset loading |
| accelerate | ≥1.14 | Distributed/mixed-precision training |
| evaluate | ≥0.4.6 | Metrics framework |
| matplotlib + plotly | latest | Visualization & interactive charts |
| llama-cpp-python | ≥0.3.31 | GGUF inference fallback |

Full dependency list in [`pyproject.toml`](pyproject.toml).

---

## Hardware Requirements

| Tier | RAM | What Works |
|---|---|---|
| **Minimum** | 16 GB | SmolLM2-135M (all techniques), SmolLM2-360M (QLoRA only) |
| **Recommended** | 36 GB | All notebooks including Qwen2.5-1.5B QLoRA |
| **Ideal** | 64+ GB | Qwen2.5-3B distillation teacher at full precision |

---

## Project Structure

See [`slm-finetuning-plan.md`](slm-finetuning-plan.md) for the full implementation plan.

---

## License

MIT

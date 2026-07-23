# finetuning-slm

> Comprehensive SLM fine-tuning tutorial — from basics to advanced techniques, designed to run on Apple Silicon, Google Colab (free), or any system with a 6GB+ GPU.

This repo covers 10+ fine-tuning techniques (LoRA, DoRA, QLoRA, DPO, Distillation, and more) with 20 hands-on Jupyter notebooks, fun & novel industry use cases, and an interactive GitHub Pages site.

---

## Quick Start

### Prerequisites

- **macOS** (Apple Silicon M1/M2/M3/M4), **Linux** with NVIDIA GPU, or **Google Colab** (free tier)
- **Python 3.11+**
- Internet connection (to download packages and models)

### One-Command Setup (Local)

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

### Google Colab Setup

Each notebook includes a Colab-compatible setup cell at the top:

```python
# Run this cell first on Google Colab
!pip install -q torch transformers peft trl datasets accelerate bitsandbytes evaluate rouge-score bert-score matplotlib plotly
```

No local install needed — just open the notebook in Colab and run.

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

## Supported Models

### Open Models (No Token Required)

These models download freely — no sign-up, no authentication:

| Model | Parameters | License | Best For |
|---|---|---|---|
| [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | 135M | Apache 2.0 | Quick demos, 6GB GPU friendly |
| [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 360M | Apache 2.0 | Main tutorials, Colab friendly |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1.5B | Apache 2.0 | Quality-critical demos |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 3B | Apache 2.0 | Distillation teacher |

### Gated Models (HuggingFace Token Required)

These models are free but require accepting the license on HuggingFace:

| Model | Parameters | License | Best For |
|---|---|---|---|
| [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 1B | Llama 3.2 Community | Alternative SLM, great quality |
| [Llama 3.2 3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 3B | Llama 3.2 Community | Strong reasoning for its size |
| [Gemma 2 2B](https://huggingface.co/google/gemma-2-2b-it) | 2B | Gemma License | Excellent instruction following |

---

## Setting Up HuggingFace Token (for Gated Models)

> **Required only if** you want to use Llama 3.2, Gemma 2, or push your fine-tuned adapters to HuggingFace Hub. Skip this if you're using only the open models above.

### Step 1: Create a HuggingFace Account

1. Go to [huggingface.co/join](https://huggingface.co/join)
2. Sign up with email or GitHub

### Step 2: Accept Model Licenses

Visit each model page and click **"Agree and access repository"**:
- Llama 3.2: [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- Gemma 2: [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)

(Approval is usually instant for Llama; Gemma may take a few minutes.)

### Step 3: Create an Access Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **"Create new token"**
3. Name: `finetuning-slm` (or anything)
4. Type: **Read** (sufficient for downloading models)
   - Choose **Write** if you also want to push fine-tuned adapters to HF Hub
5. Click **"Create"** and copy the token (starts with `hf_...`)

### Step 4: Configure Locally

```bash
# Option A: Interactive login (recommended)
uv run huggingface-cli login
# Paste your token when prompted

# Option B: Environment variable (good for Colab / CI)
export HF_TOKEN="hf_your_token_here"

# Option C: In a notebook cell (Colab)
from huggingface_hub import login
login(token="hf_your_token_here")
```

### Verify Token Works

```python
from transformers import AutoTokenizer
# This will only work if token is set AND license is accepted
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
print("✅ Llama 3.2 access confirmed!")
```

---

## Hardware Compatibility

### What Works Where

| Hardware | VRAM/RAM | Models That Work | Techniques That Work |
|---|---|---|---|
| **Colab Free** (T4) | 15 GB VRAM | All models up to 3B (QLoRA) | All techniques |
| **6 GB GPU** (RTX 3060, GTX 1660) | 6 GB VRAM | SmolLM2-135M/360M, Llama 3.2 1B (QLoRA), Qwen2.5-1.5B (QLoRA) | QLoRA, Prompt Tuning, Prefix Tuning, LoRA (135M only) |
| **Apple M1/M2** (16 GB) | 16 GB unified | SmolLM2-135M (all), SmolLM2-360M (LoRA/QLoRA) | All except Full FT on 360M+ |
| **Apple M3/M4** (36 GB) | 36 GB unified | All models including Qwen2.5-1.5B LoRA | All techniques |
| **Apple M-series** (64+ GB) | 64 GB unified | All models at full precision | Everything including Qwen2.5-3B full FT |
| **Cloud GPU** (A100 40GB) | 40 GB VRAM | Everything | Everything |

### Notebook Compatibility by Hardware Tier

| Notebook | 6 GB GPU | Colab Free (T4) | Mac 16 GB | Mac 36+ GB |
|---|---|---|---|---|
| 00 Environment Setup | ✅ | ✅ | ✅ | ✅ |
| 01 Dataset Design & Prep | ✅ | ✅ | ✅ | ✅ |
| 02 Full Fine-Tuning (135M) | ✅ | ✅ | ✅ | ✅ |
| 03 Feature Extraction | ✅ | ✅ | ✅ | ✅ |
| 04 LoRA & DoRA (360M) | ⚠️ QLoRA only | ✅ | ✅ | ✅ |
| 05 QLoRA (1.5B) | ✅ | ✅ | ⚠️ tight | ✅ |
| 06 Prompt Tuning | ✅ | ✅ | ✅ | ✅ |
| 07 Prefix Tuning | ✅ | ✅ | ✅ | ✅ |
| 08 DPO Alignment | ⚠️ 135M only | ✅ | ✅ | ✅ |
| 09 Knowledge Distillation | ❌ | ⚠️ tight | ⚠️ tight | ✅ |
| 10 Model Merging | ✅ | ✅ | ✅ | ✅ |
| 11 Failure Points | ✅ | ✅ | ✅ | ✅ |
| 12 Responsible Fine-Tuning | ✅ | ✅ | ✅ | ✅ |
| 13 Evaluation & Benchmarking | ✅ | ✅ | ✅ | ✅ |
| 14–16 Fun Use Cases | ⚠️ QLoRA variants | ✅ | ✅ | ✅ |
| 17–18 Industry Use Cases | ⚠️ limited | ✅ | ✅ | ✅ |
| 19 Apple Silicon Guide | ❌ | ❌ | ✅ | ✅ |

> **Legend**: ✅ Works fully · ⚠️ Works with smaller model/QLoRA fallback · ❌ Not applicable

### Tips for 6 GB GPU Users

- Always use **QLoRA** (4-bit quantization) for models ≥360M
- Set `batch_size=1` with `gradient_accumulation_steps=8`
- Enable `gradient_checkpointing=True`
- Stick to `max_seq_length=512` (not 1024+)
- SmolLM2-135M works for ALL techniques even on 6 GB

### Tips for Google Colab (Free Tier)

- Select **Runtime → Change runtime type → T4 GPU**
- Session limit: ~12 hours (save checkpoints frequently!)
- Use `!pip install` cell at top of each notebook
- Mount Google Drive to persist outputs: `from google.colab import drive; drive.mount('/content/drive')`
- If you hit OOM: restart runtime and reduce batch size

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

## Project Structure

See [`slm-finetuning-plan.md`](slm-finetuning-plan.md) for the full implementation plan.

---

## License

MIT

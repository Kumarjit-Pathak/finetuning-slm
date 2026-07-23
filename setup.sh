#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SLM Fine-Tuning Tutorial — One-Command Environment Setup
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# Prerequisites:
#   - macOS (Apple Silicon M1/M2/M3/M4) or Linux
#   - Python 3.11+ installed
#   - Internet connection (to download packages and models)
#
# This script will:
#   1. Install uv (if not present)
#   2. Create a virtual environment and install all dependencies
#   3. Register a Jupyter kernel so notebooks "just work"
#   4. Verify the setup (PyTorch + MPS check)
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "╔══════════════════════════════════════════════════╗"
echo "║   SLM Fine-Tuning Tutorial — Environment Setup  ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Install uv if not present ──
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "   ✅ uv installed"
else
    echo "✅ uv already installed ($(uv --version))"
fi

# ── Step 2: Create venv and install dependencies ──
echo ""
echo "📦 Creating virtual environment and installing packages..."
echo "   (This may take 2-5 minutes on first run)"
echo ""

uv sync --all-groups

echo ""
echo "   ✅ All packages installed"

# ── Step 3: Register Jupyter kernel ──
echo ""
echo "🔧 Registering Jupyter kernel..."
uv run python -m ipykernel install --user --name finetuning-slm --display-name "Python (SLM Fine-Tuning)"
echo "   ✅ Kernel 'Python (SLM Fine-Tuning)' registered"

# ── Step 4: Verify setup ──
echo ""
echo "🔍 Verifying installation..."
echo ""

uv run python -c "
import sys
import torch
import transformers
import peft
import trl
import datasets

print(f'   Python:       {sys.version.split()[0]}')
print(f'   PyTorch:      {torch.__version__}')
print(f'   Transformers: {transformers.__version__}')
print(f'   PEFT:         {peft.__version__}')
print(f'   TRL:          {trl.__version__}')
print(f'   Datasets:     {datasets.__version__}')
print()

# Check GPU
if torch.backends.mps.is_available():
    print(f'   🍎 MPS (Apple Silicon GPU): AVAILABLE')
    print(f'      → Fine-tuning will use GPU acceleration')
elif torch.cuda.is_available():
    print(f'   🟢 CUDA GPU: {torch.cuda.get_device_name(0)}')
    print(f'      → Fine-tuning will use GPU acceleration')
else:
    print(f'   ⚠️  No GPU detected — fine-tuning will run on CPU (slower)')
    print(f'      → Still works, just takes longer')
"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   ✅ Setup complete! You're ready to go.        ║"
echo "║                                                  ║"
echo "║   Next steps:                                    ║"
echo "║   1. Open VS Code in this folder                 ║"
echo "║   2. Select kernel: 'Python (SLM Fine-Tuning)'  ║"
echo "║   3. Start with notebooks/00_environment_setup   ║"
echo "╚══════════════════════════════════════════════════╝"

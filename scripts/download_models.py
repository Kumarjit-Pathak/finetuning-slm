#!/usr/bin/env python3
"""Pre-download all models used in the tutorial.

Run this once to cache models locally, then notebooks work offline.

Usage:
    python scripts/download_models.py          # Download open models only
    python scripts/download_models.py --all    # Include gated models (needs HF token)
"""

import argparse
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.progress import track

console = Console()

OPEN_MODELS = [
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

GATED_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it",
]


def download_model(model_id: str) -> bool:
    """Download a model from HuggingFace Hub."""
    try:
        console.print(f"  Downloading [bold]{model_id}[/bold]...")
        snapshot_download(model_id)
        console.print(f"  [green]✓[/green] {model_id}")
        return True
    except Exception as e:
        console.print(f"  [red]✗[/red] {model_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download tutorial models")
    parser.add_argument("--all", action="store_true", help="Include gated models (needs HF token)")
    args = parser.parse_args()

    models = OPEN_MODELS.copy()
    if args.all:
        models.extend(GATED_MODELS)
        console.print("[yellow]Including gated models — make sure you've run 'huggingface-cli login'[/yellow]\n")

    console.print(f"[bold]Downloading {len(models)} models...[/bold]\n")

    success = 0
    for model_id in models:
        if download_model(model_id):
            success += 1

    console.print(f"\n[bold]Done:[/bold] {success}/{len(models)} models downloaded successfully")
    if success < len(models):
        console.print("[yellow]Some downloads failed. For gated models, ensure you've accepted the license and logged in.[/yellow]")


if __name__ == "__main__":
    main()

"""Training helpers, callbacks, and device setup utilities."""

import torch
from transformers import TrainingArguments, TrainerCallback
from typing import Optional
import time


def setup_device() -> torch.device:
    """Detect and return the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info() -> dict:
    """Get detailed device information."""
    device = setup_device()
    info = {"device": str(device), "device_type": device.type}

    if device.type == "mps":
        info["backend"] = "Metal Performance Shaders (Apple Silicon)"
        info["unified_memory"] = True
    elif device.type == "cuda":
        info["backend"] = f"CUDA ({torch.cuda.get_device_name(0)})"
        info["vram_gb"] = torch.cuda.get_device_properties(0).total_mem / 1e9
    else:
        info["backend"] = "CPU"

    return info


def get_training_args(
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 4,
    fp16: bool = False,
    bf16: bool = False,
    low_resource: bool = False,
) -> TrainingArguments:
    """Create TrainingArguments with sensible defaults for SLM fine-tuning."""
    device = setup_device()

    # Auto-configure precision based on device
    if device.type == "mps":
        fp16 = False  # MPS prefers fp32 or bf16
        bf16 = True
    elif device.type == "cuda":
        fp16 = True

    # Low-resource overrides for 6GB GPU / Colab
    if low_resource:
        batch_size = 1
        gradient_accumulation_steps = 8

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=fp16,
        bf16=bf16,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        gradient_checkpointing=low_resource,
    )


class MemoryMonitorCallback(TrainerCallback):
    """Callback to monitor memory usage during training."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            if torch.backends.mps.is_available():
                allocated = torch.mps.current_allocated_memory() / 1e9
                print(f"  [Step {state.global_step}] MPS Memory: {allocated:.2f} GB")
            elif torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"  [Step {state.global_step}] CUDA: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


class TrainingMetricsCallback(TrainerCallback):
    """Callback to track training metrics for visualization."""

    def __init__(self):
        self.metrics = {"steps": [], "loss": [], "eval_loss": [], "timestamps": []}
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.metrics["steps"].append(state.global_step)
            self.metrics["loss"].append(logs["loss"])
            self.metrics["timestamps"].append(time.time() - self.start_time)
        if logs and "eval_loss" in logs:
            self.metrics["eval_loss"].append(logs["eval_loss"])

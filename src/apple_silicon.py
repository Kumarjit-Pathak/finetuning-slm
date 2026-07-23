"""Apple Silicon / MPS utilities: device management, memory monitoring, workarounds."""

import torch
import platform
import subprocess


def check_mps_availability() -> dict:
    """Check MPS availability and system info."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
    }

    # Get system memory
    try:
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        if result.returncode == 0:
            info["system_memory_gb"] = int(result.stdout.strip()) / (1024**3)
    except FileNotFoundError:
        pass

    # Get chip info
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        if result.returncode == 0:
            info["chip"] = result.stdout.strip()
    except FileNotFoundError:
        pass

    return info


def get_memory_stats() -> dict:
    """Get current MPS/CUDA memory statistics."""
    if torch.backends.mps.is_available():
        return {
            "backend": "MPS",
            "allocated_gb": torch.mps.current_allocated_memory() / 1e9,
            "driver_allocated_gb": torch.mps.driver_allocated_memory() / 1e9,
        }
    elif torch.cuda.is_available():
        return {
            "backend": "CUDA",
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
    return {"backend": "CPU", "note": "No GPU memory tracking available"}


def optimize_for_mps(model):
    """Apply MPS-specific optimizations to a model."""
    # Move to MPS
    if torch.backends.mps.is_available():
        model = model.to("mps")

    # Disable features that may cause issues on MPS
    if hasattr(model, "config"):
        # Some attention implementations don't work on MPS
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"

    return model


def mps_safe_generate(model, inputs, **kwargs):
    """Generate text with MPS-safe fallbacks."""
    try:
        with torch.no_grad():
            return model.generate(**inputs, **kwargs)
    except RuntimeError as e:
        if "MPS" in str(e) or "mps" in str(e):
            # Fallback to CPU for unsupported ops
            print("  ⚠️ MPS error, falling back to CPU for generation...")
            model_cpu = model.to("cpu")
            inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}
            output = model_cpu.generate(**inputs_cpu, **kwargs)
            model.to("mps")  # Move back
            return output.to("mps")
        raise

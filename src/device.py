import torch


def get_device():
    """NVIDIA GPU (Colab) first, then Apple GPU (Mac), then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def gpu_memory_mb(device):
    """GPU memory in use right now, in MB (0 on CPU)."""
    if device == "cuda":
        return torch.cuda.memory_allocated() / 1e6
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1e6
    return 0.0

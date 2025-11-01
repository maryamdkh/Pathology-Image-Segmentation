import torch
import gc
import psutil
import GPUtil

def serialize_transform(transform):
    """
    Convert an Albumentations Compose transform into a serializable dictionary.
    """
    if hasattr(transform, "to_dict"):
        return transform.to_dict()
    else:
        return str(transform)
    
def free_gpu_memory():
    """
    Frees up GPU memory in PyTorch / Colab.
    """
    # Clear cache
    torch.cuda.empty_cache()

    # Delete any lingering variables
    gc.collect()

    # Optional: check remaining memory
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    cached = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"GPU memory cleared. Allocated: {allocated:.1f} MB, Cached: {cached:.1f} MB")


def check_system_resources(model=None):
    """
    Utility to check current system and GPU usage, and model size.

    Args:
        model: (optional) torch model to report parameter count and size
    """

    print("\n" + "="*60)
    print("🧠 SYSTEM & GPU RESOURCE REPORT")
    print("="*60)

    # ---- Model Parameters ----
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

        print(f"Model Parameters:     {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Estimated Model Size: {model_size_mb:.2f} MB")

    # ---- GPU ----
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"\nGPU: {gpu.name}")
            print(f"  Load: {gpu.load*100:.1f}%")
            print(f"  Free Memory:  {gpu.memoryFree:.1f} MB")
            print(f"  Used Memory:  {gpu.memoryUsed:.1f} MB")
            print(f"  Total Memory: {gpu.memoryTotal:.1f} MB")
    else:
        print("⚠️ No GPU detected. Using CPU only.")

    # ---- CPU and RAM ----
    ram = psutil.virtual_memory()
    print("\nCPU Usage: {:.1f}%".format(psutil.cpu_percent()))
    print("RAM Usage: {:.1f}% ({:.1f}/{:.1f} GB)".format(
        ram.percent, ram.used / 1e9, ram.total / 1e9))
    print("="*60)

def get_device(prefer: str = "auto") -> torch.device:
    """Get available device."""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")

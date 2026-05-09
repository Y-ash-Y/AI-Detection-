import torch
import os


def get_device(force_device: str = None):
    """
    Returns best available device.
    
    Priority (auto):
        1. MPS (Apple Silicon M1/M2/M3+)
        2. CUDA (NVIDIA GPU)
        3. CPU (fallback)
    
    Args:
        force_device: Override with "mps", "cuda", or "cpu"
    
    Environment Variables:
        PYTORCH_DEVICE=mps|cuda|cpu   — Force specific device
    
    Returns:
        torch.device instance
    """
    # Check environment variable or function argument
    env_device = os.getenv("PYTORCH_DEVICE")
    device_name = force_device or env_device
    
    if device_name:
        device_name = device_name.lower()
        if device_name == "mps":
            if torch.backends.mps.is_available():
                print(f"✓ Using MPS (Metal Performance Shaders)")
                return torch.device("mps")
            else:
                print("⚠ MPS requested but not available. Falling back to CUDA/CPU.")
        elif device_name == "cuda":
            if torch.cuda.is_available():
                print(f"✓ Using CUDA (GPU {torch.cuda.get_device_name(0)})")
                return torch.device("cuda")
            else:
                print("⚠ CUDA requested but not available. Falling back to CPU.")
        elif device_name == "cpu":
            print("✓ Using CPU")
            return torch.device("cpu")
    
    # Auto-detect best available
    if torch.backends.mps.is_available():
        print(f"✓ Auto-detected MPS (Metal Performance Shaders)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ Auto-detected CUDA ({gpu_name})")
        return torch.device("cuda")
    else:
        print("✓ Using CPU (no GPU available)")
        return torch.device("cpu")

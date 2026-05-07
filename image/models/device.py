import torch


def get_device():
    """
    Returns best available device.
    Priority: MPS (Apple Silicon) > CUDA > CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

import torch
from torch.utils.data import Dataset
import numpy as np

class ToyAVDataset(Dataset):
    """
    Toy audio-video pairs for wiring the pipeline.
    Returns (audio_feat, frame, label) where label=0 (real) or 1 (synthetic).
    """
    def __init__(self, n=2048, feat_dim=128, frame_dim=(3, 64, 64), seed=42):
        rng = np.random.default_rng(seed)
        self.audio = rng.normal(size=(n, feat_dim)).astype("float32")
        self.frames = rng.normal(size=(n, *frame_dim)).astype("float32")
        # half real, half synthetic
        labels = np.concatenate([np.zeros(n//2, dtype=np.int64),
                                 np.ones(n - n//2, dtype=np.int64)])
        rng.shuffle(labels)
        self.labels = labels

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.audio[idx]),
                torch.from_numpy(self.frames[idx]),
                torch.tensor(self.labels[idx]))

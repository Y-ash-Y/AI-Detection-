import numpy as np
from math import sqrt
from scipy.stats import norm

class DiffusionWatermark:
    """
    Spread-spectrum watermark for diffusion noise latents.
    GLRT statistic ~ normalized correlation with the known codeword.
    """
    def __init__(self, m_bits: int = 128, seed: int = 1337, strength: float = 0.1):
        rng = np.random.default_rng(seed)
        self.code = rng.choice([-1.0, 1.0], size=(m_bits,)).astype(np.float32)
        self.strength = float(strength)

    def embed(self, noise_latent: np.ndarray) -> np.ndarray:
        z = noise_latent.astype(np.float32).copy()
        L = z.size
        reps = int(np.ceil(L / self.code.size))
        w = np.tile(self.code, reps)[:L].reshape(z.shape)
        return z + self.strength * w

    def glrt(self, observed_latent: np.ndarray) -> float:
        z = observed_latent.astype(np.float32).ravel()
        reps = int(np.ceil(z.size / self.code.size))
        w = np.tile(self.code, reps)[:z.size]
        num = float((z * w).sum())
        den = float(np.linalg.norm(z) * np.linalg.norm(w) + 1e-8)
        return num / den

    @staticmethod
    def threshold_alpha(alpha: float, n: int) -> float:
        """
        α-level threshold under H0 ≈ N(0, 1/sqrt(n)): τ ≈ Φ^{-1}(1-α) / sqrt(n)
        """
        return float(norm.ppf(1 - alpha) / sqrt(n))

import numpy as np
from pmsa.models.watermark import DiffusionWatermark

def main():
    wm = DiffusionWatermark(m_bits=128, strength=0.25)
    z = np.random.normal(size=(4096,)).astype("float32")
    z_emb = wm.embed(z)
    stat_clean = wm.glrt(z)
    stat_emb = wm.glrt(z_emb)
    tau = wm.threshold_alpha(alpha=0.01, n=z.size)
    print(f"GLRT stat clean={stat_clean:.4f}, embedded={stat_emb:.4f}, tau@1%={tau:.4f}")
    assert stat_emb > tau, "Embedded sample should pass α=1% detection on average."

if __name__ == "__main__":
    main()

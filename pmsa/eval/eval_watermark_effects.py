# pmsa/eval/eval_watermark_attacks.py

import numpy as np
import torch
from pmsa.models.watermark import DiffusionWatermark
from pmsa.utils.attacks import jpeg_compress, add_gaussian_noise, random_crop


def eval_watermark(
    n_trials=500,
    latent_shape=(1, 64, 64),
    alpha=0.01
):
    wm = DiffusionWatermark(m_bits=128, strength=0.25)
    tau = wm.threshold_alpha(alpha, np.prod(latent_shape))
    print(f"GLRT threshold τ={tau:.4f}")

    def run(name, attack_fn):
        hits = 0
        for _ in range(n_trials):
            z = np.random.randn(*latent_shape).astype("float32")
            z_w = wm.embed(z)
            z_t = torch.tensor(z_w).unsqueeze(0)

            if attack_fn:
                z_t = attack_fn(z_t).squeeze(0).numpy()
            else:
                z_t = z_w

            stat = wm.glrt(z_t)
            if stat > tau:
                hits += 1

        print(f"{name:12s} TPR@α={hits/n_trials:.3f}")

    run("Clean", None)
    run("JPEG-30", lambda x: jpeg_compress(x, quality=30))
    run("Noise-0.1", lambda x: add_gaussian_noise(x, sigma=0.1))
    run("Crop-85%", lambda x: random_crop(x, crop_frac=0.85))


if __name__ == "__main__":
    eval_watermark()

# pmsa/eval/eval_explanations.py

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from pmsa.data.structured_dataset import StructuredAVDataset
from pmsa.models.detector import LRTSurrogate
from pmsa.models.explain import explain_image


def save_overlay(img, heat, path):
    img = ((img.transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
    cmap = plt.get_cmap("jet")
    heat_rgb = (cmap(heat)[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.6 * img + 0.4 * heat_rgb).astype(np.uint8)
    Image.fromarray(overlay).save(path)


def main(out_dir="explain_report", n_samples=25):
    os.makedirs(out_dir, exist_ok=True)

    ds = StructuredAVDataset(n=2000)
    model = LRTSurrogate().eval()

    rows = []
    for i in tqdm(range(n_samples)):
        _, frame, label = ds[i]
        frame = frame.unsqueeze(0)

        heat, text = explain_image(model, frame)
        img_np = frame.squeeze().numpy()

        img_path = f"{out_dir}/img_{i}.png"
        heat_path = f"{out_dir}/heat_{i}.png"

        Image.fromarray(((img_np.transpose(1,2,0)+1)*127.5).astype(np.uint8)).save(img_path)
        if heat is not None:
            save_overlay(img_np, heat, heat_path)

        rows.append(f"""
        <div>
          <img src="{img_path}" width="256">
          <img src="{heat_path}" width="256">
          <p>{text}</p>
        </div>
        """)

    html = "<html><body>" + "\n".join(rows) + "</body></html>"
    with open(f"{out_dir}/report.html", "w") as f:
        f.write(html)

    print("Saved explanation report:", out_dir)


if __name__ == "__main__":
    main()

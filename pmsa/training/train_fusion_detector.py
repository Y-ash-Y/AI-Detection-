import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from pmsa.models.fusion_detector import FusionDetector
from pmsa.models.detector import calibrate_threshold, roc_auc
from pmsa.utils.logging import set_seed

def generate_dummy_cache(
    n=6000,
    clip_dim=768,
    dino_dim=1024,
    forensic_dim=16,
    seed=42,
):
    rng = np.random.default_rng(seed)
    clip = rng.normal(size=(n, clip_dim)).astype("float32")
    dino = rng.normal(size=(n, dino_dim)).astype("float32")
    forensic = rng.normal(size=(n, forensic_dim)).astype("float32")
    labels = np.concatenate([np.zeros(n//2), np.ones(n - n//2)]).astype("int64")
    rng.shuffle(labels)
    return clip, dino, forensic, labels

def train(
    alpha=0.01,
    epochs=5,
    lr=1e-3,
    seed=42,
    save_path="fusion_detector.pt"
):
    set_seed(seed)

    # ---- Dummy cached features ----
    clip, dino, forensic, y = generate_dummy_cache()

    X = np.concatenate([clip, dino, forensic], axis=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train).float()
    X_val = torch.from_numpy(X_val)
    y_val = torch.from_numpy(y_val)

    model = FusionDetector(
        clip_dim=768,
        dino_dim=1024,
        forensic_dim=16
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # ---- Training ----
    model.train()
    for ep in range(epochs):
        logits = model(
            X_train[:, :768],
            X_train[:, 768:768+1024],
            X_train[:, -16:]
        )
        loss = loss_fn(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"Epoch {ep+1}/{epochs} | Loss={loss.item():.4f}")

    # ---- Validation & NP calibration ----
    model.eval()
    with torch.no_grad():
        scores = model(
            X_val[:, :768],
            X_val[:, 768:768+1024],
            X_val[:, -16:]
        )
        auc = roc_auc(scores, y_val)
        tau = calibrate_threshold(scores, y_val, alpha)

    print(f"[VAL] AUC={auc:.3f} | τ@α={alpha} → {tau:.4f}")

    torch.save(
        {"state_dict": model.state_dict(), "tau": tau},
        save_path
    )
    print("Saved:", save_path)

if __name__ == "__main__":
    train()

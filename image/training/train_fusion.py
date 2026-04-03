import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pmsa.image.models.fusion_model import FusionDetector
from pmsa.shared.utils import calibrate_threshold, set_seed


def load_cached_features(path="feature_cache/features.npz"):
    data = np.load(path)
    return (
        data["clip"].astype("float32"),
        data["dino"].astype("float32"),
        data["forensic"].astype("float32"),
        data["label"].astype("int64"),
    )


def train(
    alpha=0.01,
    epochs=15,
    lr=1e-3,
    seed=42,
    save_path="fusion_detector.pt",
):
    set_seed(seed)

    # Load cached features
    clip_f, dino_f, forensic_f, y = load_cached_features()

    X = np.concatenate([clip_f, dino_f, forensic_f], axis=1)
    y = y.astype("float32")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_val = torch.from_numpy(X_val)
    y_val = torch.from_numpy(y_val)

    model = FusionDetector(
        clip_dim=clip_f.shape[1],
        dino_dim=dino_f.shape[1],
        forensic_dim=forensic_f.shape[1],
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Training
    model.train()
    for ep in range(epochs):
        logits = model(
            X_train[:, :clip_f.shape[1]],
            X_train[:, clip_f.shape[1] : clip_f.shape[1] + dino_f.shape[1]],
            X_train[:, -forensic_f.shape[1] :],
        )
        loss = loss_fn(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Epoch {ep+1}/{epochs} | Loss={loss.item():.4f}")

    # Validation + NP calibration
    model.eval()
    with torch.no_grad():
        scores = model(
            X_val[:, :clip_f.shape[1]],
            X_val[:, clip_f.shape[1] : clip_f.shape[1] + dino_f.shape[1]],
            X_val[:, -forensic_f.shape[1] :],
        ).numpy()

    auc = roc_auc_score(y_val.numpy(), scores)
    tau = calibrate_threshold(torch.tensor(scores), y_val, alpha)

    print(f"[VAL] AUC={auc:.3f} | τ@α={alpha} → {tau:.4f}")

    torch.save(
        {"state_dict": model.state_dict(), "tau": tau},
        save_path,
    )
    print("Saved trained fusion detector →", save_path)


if __name__ == "__main__":
    train()

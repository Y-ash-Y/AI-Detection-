import numpy as np
import cv2
from PIL import Image


# ── Constants ──────────────────────────────────────────────
DCT_DIM    = 512   # high-freq DCT coefficients
NOISE_DIM  = 512   # Laplacian noise residual
FORENSIC_DIM = DCT_DIM + NOISE_DIM   # = 1024


# ── Individual feature extractors ──────────────────────────

def extract_dct_features(image_path: str) -> np.ndarray:
    """
    DCT spectrum features.

    Real cameras have natural 1/f^2 power decay.
    GAN/diffusion upsampling introduces periodic spectral spikes.
    We capture the high-frequency tail (coefficients 256:768 after
    flattening) which is most discriminative.

    Returns: float32 array of shape (DCT_DIM,)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # fallback: PIL load → numpy
        img = np.array(Image.open(image_path).convert("L"))

    img = cv2.resize(img, (256, 256)).astype(np.float32)

    # block DCT: process 8×8 blocks like JPEG
    h, w = img.shape
    dct_map = np.zeros_like(img)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = img[i:i+8, j:j+8]
            if block.shape == (8, 8):
                dct_map[i:i+8, j:j+8] = cv2.dct(block)

    # take log magnitude + flatten → slice high-freq region
    log_dct = np.log1p(np.abs(dct_map)).flatten()

    # pick 512 coefficients from the high-frequency region
    total = log_dct.shape[0]
    start = total // 4   # skip DC component heavy region
    feat = log_dct[start: start + DCT_DIM]

    if feat.shape[0] < DCT_DIM:
        feat = np.pad(feat, (0, DCT_DIM - feat.shape[0]))

    return feat.astype(np.float32)


def extract_noise_residual(image_path: str) -> np.ndarray:
    """
    Noise residual features.

    Real cameras: I = Signal + Sensor_noise (photon shot, PRNU, etc.)
    AI images: I = Generator_output (no camera noise fingerprint)

    We extract noise via Laplacian high-pass filter, then compute
    statistical descriptors over spatial blocks.

    Returns: float32 array of shape (NOISE_DIM,)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.array(Image.open(image_path).convert("L"))

    img = cv2.resize(img, (256, 256)).astype(np.float32)

    # Laplacian = high-pass = noise residual
    laplacian = cv2.Laplacian(img, cv2.CV_32F)
    residual = np.abs(laplacian).astype(np.float32)

    # Compute block statistics (mean + std over 8×8 blocks)
    stats = []
    for i in range(0, 256, 8):
        for j in range(0, 256, 8):
            block = residual[i:i+8, j:j+8]
            stats.append(block.mean())
            stats.append(block.std())

    feat = np.array(stats, dtype=np.float32)

    # resize to fixed NOISE_DIM
    if feat.shape[0] >= NOISE_DIM:
        feat = feat[:NOISE_DIM]
    else:
        feat = np.pad(feat, (0, NOISE_DIM - feat.shape[0]))

    return feat


def extract_forensic_features(image_path: str) -> np.ndarray:
    """
    Combined forensic feature vector.
    Returns: float32 array of shape (FORENSIC_DIM,) = (1024,)
    """
    dct   = extract_dct_features(image_path)
    noise = extract_noise_residual(image_path)
    return np.concatenate([dct, noise])

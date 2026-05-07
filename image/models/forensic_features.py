import numpy as np
import cv2
from PIL import Image
from scipy import stats as scipy_stats

# ── Constants ──────────────────────────────────────────────────────────────
DCT_DIM      = 512   # global 2D FFT radial spectrum + block DCT stats
NOISE_DIM    = 512   # improved Gaussian-denoised residual statistics
PRNU_DIM     = 256   # Photo Response Non-Uniformity features (NEW)
FORENSIC_DIM = DCT_DIM + NOISE_DIM + PRNU_DIM  # = 1280


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_gray(image_path: str, size: int = 256) -> np.ndarray:
    """Load image as float32 grayscale, resized to size×size."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.array(Image.open(image_path).convert("L"))
    return cv2.resize(img, (size, size)).astype(np.float32)


def _load_rgb(image_path: str, size: int = 128) -> np.ndarray:
    """Load image as float32 RGB (H, W, 3), resized to size×size."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        img = np.array(Image.open(image_path).convert("RGB"))
        img = cv2.resize(img, (size, size)).astype(np.float32)
        return img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size)).astype(np.float32)


def _safe_stats(arr: np.ndarray) -> np.ndarray:
    """
    Return [mean, std, skewness, kurtosis] of a flat array.
    Safe against constant arrays (zero std).
    """
    flat = arr.flatten().astype(np.float64)
    mean = float(np.mean(flat))
    std  = float(np.std(flat))
    if std < 1e-8:
        return np.array([mean, std, 0.0, 0.0], dtype=np.float32)
    skew = float(scipy_stats.skew(flat))
    kurt = float(scipy_stats.kurtosis(flat))
    return np.array([mean, std, skew, kurt], dtype=np.float32)


def _pad_or_trim(feat: np.ndarray, target_dim: int) -> np.ndarray:
    """Ensure feature vector is exactly target_dim long."""
    if feat.shape[0] >= target_dim:
        return feat[:target_dim].astype(np.float32)
    return np.pad(feat, (0, target_dim - feat.shape[0])).astype(np.float32)


# ── Feature extractor 1: DCT + FFT spectrum ────────────────────────────────

def extract_dct_features(image_path: str) -> np.ndarray:
    """
    Spectral fingerprint features combining:

    (A) Global 2D FFT radial power spectrum
        Real images follow 1/f^2 power law.
        GAN/diffusion upsampling introduces periodic grid artifacts
        visible as spikes in the frequency domain.
        We compute azimuthally-averaged radial spectrum (64 bands)
        which is rotation-invariant and captures these spikes cleanly.

    (B) Block DCT statistics
        8×8 block DCT mirrors JPEG compression basis.
        AI generators leave systematic residuals in AC coefficient
        distributions. We capture mean + std per AC coefficient
        position (63 AC coefficients × 2 stats = 126D).

    Returns: float32 array of shape (DCT_DIM,) = (512,)
    """
    img = _load_gray(image_path, size=256)

    feats = []

    # (A) Global FFT radial spectrum — 128D
    fft_mag = np.abs(np.fft.fft2(img))
    fft_mag = np.fft.fftshift(fft_mag)
    fft_log = np.log1p(fft_mag)

    h, w = fft_log.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.indices((h, w))
    radius = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2).astype(int)

    n_bands = 128
    max_r = min(cy, cx)
    band_size = max_r // n_bands
    if band_size < 1:
        band_size = 1

    radial_spectrum = []
    for b in range(n_bands):
        r_min = b * band_size
        r_max = (b + 1) * band_size
        mask = (radius >= r_min) & (radius < r_max)
        band_vals = fft_log[mask]
        radial_spectrum.append(float(band_vals.mean()) if band_vals.size > 0 else 0.0)

    feats.append(np.array(radial_spectrum, dtype=np.float32))  # 128D

    # (B) Block DCT AC coefficient statistics — 126×2 = 252D
    ac_accum = np.zeros((63,), dtype=np.float64)   # mean accumulator
    ac_sq    = np.zeros((63,), dtype=np.float64)   # for std computation
    n_blocks = 0

    for i in range(0, 256, 8):
        for j in range(0, 256, 8):
            block = img[i:i+8, j:j+8]
            if block.shape != (8, 8):
                continue
            dct_block = cv2.dct(block).flatten()
            ac = dct_block[1:]  # skip DC (index 0)
            ac_accum += ac
            ac_sq    += ac ** 2
            n_blocks += 1

    if n_blocks > 0:
        ac_mean = (ac_accum / n_blocks).astype(np.float32)
        ac_var  = (ac_sq / n_blocks - (ac_accum / n_blocks) ** 2)
        ac_std  = np.sqrt(np.maximum(ac_var, 0)).astype(np.float32)
    else:
        ac_mean = np.zeros(63, dtype=np.float32)
        ac_std  = np.zeros(63, dtype=np.float32)

    feats.append(ac_mean)   # 63D
    feats.append(ac_std)    # 63D
    # total so far: 128 + 63 + 63 = 254D

    # (C) High-frequency energy ratio — 2D
    low_energy  = float(fft_log[cy-32:cy+32, cx-32:cx+32].mean())
    high_energy = float(fft_log.mean())
    feats.append(np.array([low_energy, high_energy], dtype=np.float32))
    # total: 256D — pad/trim to DCT_DIM=512

    combined = np.concatenate(feats)  # 256D
    # expand with element-wise square (nonlinear feature expansion)
    combined = np.concatenate([combined, combined ** 2])  # 512D

    return _pad_or_trim(combined, DCT_DIM)


# ── Feature extractor 2: Noise residual (improved) ─────────────────────────

def extract_noise_residual(image_path: str) -> np.ndarray:
    """
    Gaussian-denoised noise residual features.

    Improvement over Laplacian approach:
      - Laplacian responds to edges (real image structure), not just noise
      - Gaussian denoiser: noise = image - GaussianBlur(image, sigma)
        produces a cleaner noise estimate closer to true sensor noise

    We compute:
      (A) 8×8 spatial block statistics (mean + std) — 256D
      (B) Global noise statistics per frequency band — 128D
      (C) Noise histogram (normalised, 128 bins) — 128D
    Total: 512D

    Returns: float32 array of shape (NOISE_DIM,) = (512,)
    """
    img = _load_gray(image_path, size=256)
    img_norm = img / 255.0

    feats = []

    # (A) Multi-scale Gaussian noise residuals
    #     sigma=1: fine noise; sigma=2: coarser structured noise
    for sigma in [1.0, 2.0]:
        blurred  = cv2.GaussianBlur(img_norm,
                                    (0, 0),        # kernel auto-sized
                                    sigma)
        residual = np.abs(img_norm - blurred).astype(np.float32)

        # 8×8 block stats: mean + std → (32×32 blocks) × 2 = 2048 values
        # → summarise as 4×4 macro-block statistics = 32 × 2 = 64D per sigma
        macro_means = []
        macro_stds  = []
        for i in range(0, 256, 64):
            for j in range(0, 256, 64):
                patch = residual[i:i+64, j:j+64]
                macro_means.append(float(patch.mean()))
                macro_stds.append(float(patch.std()))
        feats.append(np.array(macro_means + macro_stds, dtype=np.float32))  # 32D each
    # (A) total: 64D

    # (B) Fine-grained block stats (8×8 blocks, single sigma=1)
    blurred  = cv2.GaussianBlur(img_norm, (0, 0), 1.0)
    residual = np.abs(img_norm - blurred).astype(np.float32)

    block_stats = []
    for i in range(0, 256, 8):
        for j in range(0, 256, 8):
            block = residual[i:i+8, j:j+8]
            block_stats.append(float(block.mean()))
            block_stats.append(float(block.std()))
    feats.append(np.array(block_stats, dtype=np.float32))  # (32×32×2) = 2048D → trim

    # (C) Noise histogram — 128D
    hist, _ = np.histogram(residual.flatten(), bins=128, range=(0, 0.2),
                           density=True)
    feats.append(hist.astype(np.float32))  # 128D

    combined = np.concatenate(feats)
    return _pad_or_trim(combined, NOISE_DIM)


# ── Feature extractor 3: PRNU features (NEW) ───────────────────────────────

def extract_prnu_features(image_path: str) -> np.ndarray:
    """
    Photo Response Non-Uniformity (PRNU) features.

    Physical basis:
      Real cameras: each pixel has a unique sensitivity offset
        due to manufacturing variance → consistent per-pixel
        noise pattern (the camera's "fingerprint").
      AI generators: no physical sensor → PRNU fingerprint is absent.
        Generated noise patterns are statistically different:
        more spatially correlated, generation-model dependent.

    We estimate PRNU noise per channel as:
        N_c = I_c - Denoise(I_c)    [using Gaussian denoiser]

    Then extract:
      (A) Per-channel statistical moments: 4 stats × 3 channels = 12D
      (B) Spatial autocorrelation (lags 1–8, horizontal + vertical)
          per channel: 8 × 2 × 3 = 48D
          PRNU has high spatial correlation; AI noise is uncorrelated.
      (C) Cross-channel noise correlation: ρ(R,G), ρ(R,B), ρ(G,B) = 3D
          Camera noise: correlated across channels (same sensor).
          AI noise: may be channel-independent.
      (D) Frequency-domain noise energy in 8 bands × 3 channels = 24D
      (E) Spatial grid noise energy: 4×4 grid × 3 channels = 48D

    Total raw: 12+48+3+24+48 = 135D → pad to PRNU_DIM=256

    Returns: float32 array of shape (PRNU_DIM,) = (256,)
    """
    img_rgb = _load_rgb(image_path, size=128)  # (128, 128, 3)
    img_rgb = img_rgb / 255.0

    feats = []

    # Extract per-channel noise residual
    noise_channels = []
    for c in range(3):
        chan    = img_rgb[:, :, c].astype(np.float32)
        blurred = cv2.GaussianBlur(chan, (0, 0), 1.0)
        noise_channels.append(chan - blurred)  # signed residual
    noise = np.stack(noise_channels, axis=-1)  # (128, 128, 3)

    # (A) Per-channel moments — 12D
    moment_feats = []
    for c in range(3):
        moment_feats.extend(_safe_stats(noise[:, :, c]).tolist())
    feats.append(np.array(moment_feats, dtype=np.float32))

    # (B) Spatial autocorrelation — 48D
    autocorr_feats = []
    for c in range(3):
        n_c = noise[:, :, c].flatten()
        std = float(n_c.std())
        if std < 1e-8:
            autocorr_feats.extend([0.0] * 16)
            continue
        n_norm = (n_c - n_c.mean()) / std

        # horizontal autocorrelation (lags 1–8)
        for lag in range(1, 9):
            x = noise[:, lag:, c].flatten()
            y = noise[:, :-lag, c].flatten()
            if x.std() < 1e-8 or y.std() < 1e-8:
                autocorr_feats.append(0.0)
            else:
                corr = float(np.corrcoef(x, y)[0, 1])
                autocorr_feats.append(0.0 if np.isnan(corr) else corr)

        # vertical autocorrelation (lags 1–8)
        for lag in range(1, 9):
            x = noise[lag:, :, c].flatten()
            y = noise[:-lag, :, c].flatten()
            if x.std() < 1e-8 or y.std() < 1e-8:
                autocorr_feats.append(0.0)
            else:
                corr = float(np.corrcoef(x, y)[0, 1])
                autocorr_feats.append(0.0 if np.isnan(corr) else corr)

    feats.append(np.array(autocorr_feats, dtype=np.float32))  # 48D

    # (C) Cross-channel noise correlation — 3D
    cross_feats = []
    pairs = [(0, 1), (0, 2), (1, 2)]
    for c1, c2 in pairs:
        n1 = noise[:, :, c1].flatten()
        n2 = noise[:, :, c2].flatten()
        if n1.std() < 1e-8 or n2.std() < 1e-8:
            cross_feats.append(0.0)
        else:
            corr = float(np.corrcoef(n1, n2)[0, 1])
            cross_feats.append(0.0 if np.isnan(corr) else corr)
    feats.append(np.array(cross_feats, dtype=np.float32))  # 3D

    # (D) Frequency-domain noise energy in 8 bands — 24D
    freq_feats = []
    for c in range(3):
        fft_noise = np.abs(np.fft.fft2(noise[:, :, c]))
        fft_noise = np.fft.fftshift(fft_noise)
        h, w = fft_noise.shape
        cy, cx = h // 2, w // 2
        y_idx, x_idx = np.indices((h, w))
        radius = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2).astype(int)
        max_r = min(cy, cx)
        band_size = max(1, max_r // 8)
        for b in range(8):
            mask = (radius >= b * band_size) & (radius < (b + 1) * band_size)
            band_vals = fft_noise[mask]
            freq_feats.append(float(band_vals.mean()) if band_vals.size > 0 else 0.0)
    feats.append(np.array(freq_feats, dtype=np.float32))  # 24D

    # (E) Spatial grid energy — 48D
    grid_feats = []
    grid_size  = 128 // 4  # 4×4 grid → 32px per cell
    for c in range(3):
        for i in range(4):
            for j in range(4):
                cell = noise[i*grid_size:(i+1)*grid_size,
                             j*grid_size:(j+1)*grid_size, c]
                grid_feats.append(float(np.abs(cell).mean()))
    feats.append(np.array(grid_feats, dtype=np.float32))  # 48D

    combined = np.concatenate(feats)  # ~135D
    return _pad_or_trim(combined, PRNU_DIM)


# ── Combined extractor ─────────────────────────────────────────────────────

def extract_forensic_features(image_path: str) -> np.ndarray:
    """
    Full forensic feature vector combining:
      - DCT + FFT spectral fingerprint  (512D)
      - Gaussian noise residual stats   (512D)
      - PRNU camera fingerprint         (256D)

    Total: 1280D float32

    The three streams capture complementary signals:
      DCT/FFT  → generator upsampling artifacts, spectral anomalies
      Noise    → texture statistics, compression residuals
      PRNU     → absence of physical camera sensor fingerprint

    Returns: float32 array of shape (FORENSIC_DIM,) = (1280,)
    """
    dct   = extract_dct_features(image_path)    # 512D
    noise = extract_noise_residual(image_path)  # 512D
    prnu  = extract_prnu_features(image_path)   # 256D
    return np.concatenate([dct, noise, prnu])   # 1280D
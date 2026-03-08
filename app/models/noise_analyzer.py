"""
Texture & Noise Analyzer

Real photographs produced by CMOS/CCD sensors exhibit characteristic noise
patterns that differ markedly from the smooth, latent-diffusion-model-generated
noise in synthetic images.  This module applies several statistical tests:

  1. Local-variance analysis  – camera sensors show spatially-uniform micro
     variance; diffusion models produce structured variance gradients.
  2. High-frequency energy ratio – the amount of fine-grained high-frequency
     detail relative to low-frequency content.
  3. Noise floor estimation   – measures the standard deviation of the
     estimated noise residual (after median-filter subtraction).
  4. Colour-channel noise correlation – independent sensor channels in a real
     camera are only weakly correlated in noise space.  AI outputs tend to
     produce highly correlated inter-channel noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from skimage.filters import sobel


@dataclass
class NoiseResult:
    local_variance_score: float     # 0-1, higher = more camera-like
    high_freq_energy_ratio: float   # higher = more real-world texture
    noise_floor_std: float          # std-dev of noise residual
    channel_noise_correlation: float  # 0 = uncorrelated (real), 1 = correlated (synthetic)
    score: float                    # overall 0-1 (1 = camera-like)
    signals: List[str] = field(default_factory=list)


def analyze(img: Image.Image) -> NoiseResult:
    rgb = np.array(img.convert("RGB"), dtype=np.float32)
    gray = rgb.mean(axis=2)

    lv_score = _local_variance_score(gray)
    hf_ratio = _high_frequency_ratio(gray)
    nf_std = _noise_floor_std(gray)
    ch_corr = _channel_noise_correlation(rgb)

    score, signals = _score(lv_score, hf_ratio, nf_std, ch_corr)

    return NoiseResult(
        local_variance_score=round(lv_score, 4),
        high_freq_energy_ratio=round(hf_ratio, 4),
        noise_floor_std=round(float(nf_std), 4),
        channel_noise_correlation=round(float(ch_corr), 4),
        score=round(score, 4),
        signals=signals,
    )


# ---------------------------------------------------------------------------
# sub-analyses
# ---------------------------------------------------------------------------

def _local_variance_score(gray: np.ndarray) -> float:
    """
    Compute the coefficient-of-variation of block-level variance.
    Camera images show high spatial uniformity in noise variance (low CoV),
    while AI images may show structured variance (also low or very high CoV).
    We map this to a [0,1] score biased toward camera characteristics.
    """
    h, w = gray.shape
    block = 16
    variances = []
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            patch = gray[y:y + block, x:x + block]
            variances.append(float(np.var(patch)))

    if not variances:
        return 0.5

    var_arr = np.array(variances)
    mean_var = var_arr.mean()
    if mean_var < 1e-6:
        return 0.3  # very flat – likely synthetic or over-smoothed

    cov = var_arr.std() / mean_var  # coefficient of variation
    # Low CoV and moderate mean variance → camera-like
    # We score high if mean_var is in a "sensor noise" range and CoV is low-moderate
    mean_score = float(np.clip(mean_var / 500.0, 0.0, 1.0))
    cov_score = float(np.clip(1.0 - cov / 3.0, 0.0, 1.0))
    return (mean_score + cov_score) / 2.0


def _high_frequency_ratio(gray: np.ndarray) -> float:
    """
    Ratio of high-frequency Sobel energy to total image energy.
    Real photographs generally exhibit more heterogeneous fine detail.
    """
    edges = sobel(gray / 255.0)
    total_energy = float(np.mean(gray ** 2)) + 1e-9
    hf_energy = float(np.mean(edges ** 2))
    ratio = hf_energy / total_energy
    return float(np.clip(ratio * 20.0, 0.0, 1.0))  # normalise to 0-1


def _noise_floor_std(gray: np.ndarray) -> float:
    """
    Estimate noise residual via median-filter subtraction.
    Camera sensor noise produces a characteristic residual std-dev.
    """
    smooth = median_filter(gray, size=3).astype(np.float32)
    residual = gray - smooth
    return float(np.std(residual))


def _channel_noise_correlation(rgb: np.ndarray) -> float:
    """
    Compute mean absolute Pearson correlation of noise residuals across
    R, G, B channels.  Values close to 0 suggest real camera noise;
    values close to 1 suggest synthetic correlated generation.
    """
    channels = []
    for c in range(3):
        channel = rgb[:, :, c]
        smooth = median_filter(channel, size=3).astype(np.float32)
        residual = (channel - smooth).flatten()
        channels.append(residual)

    correlations = []
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = channels[i], channels[j]
            std_a, std_b = a.std(), b.std()
            if std_a < 1e-9 or std_b < 1e-9:
                correlations.append(0.0)
                continue
            corr_val = np.corrcoef(a, b)[0, 1]
            if np.isnan(corr_val):
                correlations.append(0.0)
            else:
                correlations.append(abs(float(corr_val)))

    return float(np.mean(correlations)) if correlations else 0.5


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _score(
    lv: float, hf: float, nf: float, ch_corr: float
) -> tuple[float, List[str]]:
    signals: List[str] = []

    # Noise floor: real cameras typically 2-15 in 0-255 float space
    if nf < 1.0:
        nf_score = 0.2
        signals.append("Very low noise floor – image may be synthetically over-smoothed.")
    elif nf < 3.0:
        nf_score = 0.5
        signals.append("Low noise floor – possible AI generation or heavy post-processing.")
    elif nf <= 20.0:
        nf_score = 1.0
        signals.append(f"Noise floor ({nf:.2f}) consistent with real camera sensor.")
    else:
        nf_score = 0.6
        signals.append(f"Elevated noise floor ({nf:.2f}) – may indicate compression or film grain.")

    # Channel correlation: real = low, synthetic = high
    if ch_corr > 0.8:
        corr_score = 0.1
        signals.append("High inter-channel noise correlation – characteristic of AI generation.")
    elif ch_corr > 0.5:
        corr_score = 0.5
        signals.append("Moderate inter-channel noise correlation.")
    else:
        corr_score = 1.0
        signals.append("Low inter-channel noise correlation – consistent with camera sensor.")

    if hf < 0.05:
        signals.append("Very low high-frequency detail – possible AI over-smoothing.")
    elif hf > 0.3:
        signals.append("Rich high-frequency texture – consistent with real photography.")

    weights = [0.25, 0.25, 0.25, 0.25]
    raw = weights[0] * lv + weights[1] * hf + weights[2] * nf_score + weights[3] * corr_score
    return round(float(np.clip(raw, 0.0, 1.0)), 4), signals

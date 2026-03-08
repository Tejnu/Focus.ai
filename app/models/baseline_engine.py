"""
Comparative Baseline Engine

Estimates the probability of synthesis by analysing frequency-domain
characteristics and performing Error Level Analysis (ELA).

Techniques:
  1. DCT energy distribution – genuine JPEG images exhibit a natural fall-off
     of energy from low to high frequencies.  AI-generated images often show
     anomalous high-frequency energy or abnormally flat DCT spectra.
  2. FFT radial spectrum      – the power spectral density along radial
     frequencies reveals diffusion-model "fingerprints" in certain bands.
  3. Error Level Analysis     – re-saves the image at a known quality and
     compares the difference.  Uniform ELA maps are characteristic of
     AI-generated images; heterogeneous maps suggest authentic photography.
  4. Colour-histogram entropy – natural photographs have high colour entropy
     due to optical complexity; smooth AI outputs can appear lower entropy.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List

import numpy as np
from PIL import Image


@dataclass
class BaselineResult:
    dct_high_freq_ratio: float     # 0-1, lower = more natural fall-off
    fft_spectral_flatness: float   # 0-1, higher = flatter (more synthetic)
    ela_uniformity: float          # 0-1, higher = more uniform ELA (more synthetic)
    colour_entropy: float          # bits; higher = richer natural palette
    score: float                   # 0-1, 1 = looks real
    signals: List[str] = field(default_factory=list)


def analyze(img: Image.Image) -> BaselineResult:
    gray = np.array(img.convert("L"), dtype=np.float32)
    rgb = np.array(img.convert("RGB"), dtype=np.float32)

    dct_hf = _dct_high_freq_ratio(gray)
    fft_flat = _fft_spectral_flatness(gray)
    ela_uni = _ela_uniformity(img)
    entropy = _colour_entropy(rgb)

    score, signals = _score(dct_hf, fft_flat, ela_uni, entropy)

    return BaselineResult(
        dct_high_freq_ratio=round(dct_hf, 4),
        fft_spectral_flatness=round(fft_flat, 4),
        ela_uniformity=round(ela_uni, 4),
        colour_entropy=round(entropy, 4),
        score=round(score, 4),
        signals=signals,
    )


# ---------------------------------------------------------------------------
# sub-analyses
# ---------------------------------------------------------------------------

def _dct_high_freq_ratio(gray: np.ndarray) -> float:
    """
    Approximate DCT energy analysis via 2D FFT (equivalent for this purpose).
    Returns the ratio of energy in high-frequency bins to total energy.
    """
    h, w = gray.shape
    f = np.fft.fft2(gray / 255.0)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift) ** 2

    cy, cx = h // 2, w // 2
    # Low-frequency region: inner 10% of the spectrum
    radius_low = int(min(h, w) * 0.10)
    radius_high = int(min(h, w) * 0.45)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

    low_mask = dist <= radius_low
    high_mask = (dist > radius_low) & (dist <= radius_high)

    low_energy = float(magnitude[low_mask].sum()) + 1e-12
    high_energy = float(magnitude[high_mask].sum()) + 1e-12
    total = low_energy + high_energy

    return float(np.clip(high_energy / total, 0.0, 1.0))


def _fft_spectral_flatness(gray: np.ndarray) -> float:
    """
    Wiener entropy (spectral flatness) of the radial power spectrum.
    Flat spectrum → synthetic; peaked natural spectrum → real.
    """
    h, w = gray.shape
    f = np.fft.fft2(gray / 255.0)
    magnitude = np.abs(np.fft.fftshift(f))
    power = magnitude ** 2

    cy, cx = h // 2, w // 2
    max_r = int(min(h, w) * 0.5)
    radial_profile = []
    for r in range(1, max_r, 2):
        Y, X = np.ogrid[:h, :w]
        mask = (np.sqrt((Y - cy) ** 2 + (X - cx) ** 2).astype(int) == r)
        if mask.any():
            radial_profile.append(float(power[mask].mean()))

    if len(radial_profile) < 4:
        return 0.5

    rp = np.array(radial_profile, dtype=np.float64) + 1e-12
    geom_mean = float(np.exp(np.mean(np.log(rp))))
    arith_mean = float(rp.mean())
    flatness = geom_mean / arith_mean  # 0 = very peaked, 1 = flat

    return float(np.clip(flatness, 0.0, 1.0))


def _ela_uniformity(img: Image.Image) -> float:
    """
    Error Level Analysis: re-compress at JPEG quality 90 and measure the
    uniformity of the difference map.
    Uniform difference → AI-generated; heterogeneous → authentic JPEG.
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    buf.seek(0)
    recompressed = Image.open(buf)
    recompressed.load()

    orig = np.array(img.convert("RGB"), dtype=np.float32)
    recomp = np.array(recompressed, dtype=np.float32)

    diff = np.abs(orig - recomp)
    mean_diff = diff.mean()
    std_diff = diff.std()

    if mean_diff < 1e-6:
        return 1.0  # perfectly uniform → very suspicious

    cov = std_diff / (mean_diff + 1e-9)
    # Low coefficient of variation → uniform ELA → synthetic
    uniformity = float(np.clip(1.0 - cov / 5.0, 0.0, 1.0))
    return uniformity


def _colour_entropy(rgb: np.ndarray) -> float:
    """
    Compute the Shannon entropy of the colour histogram across all 3 channels.
    Real photographs tend to have higher colour entropy.
    Returns bits.
    """
    entropies = []
    for c in range(3):
        hist, _ = np.histogram(rgb[:, :, c], bins=64, range=(0, 256))
        prob = hist / (hist.sum() + 1e-9)
        prob = prob[prob > 0]
        entropies.append(float(-np.sum(prob * np.log2(prob))))
    return float(np.mean(entropies))


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _score(dct_hf: float, fft_flat: float, ela_uni: float, entropy: float) -> tuple[float, List[str]]:
    signals: List[str] = []

    # DCT high-frequency ratio: natural ~0.1-0.4; very low or very high → synthetic
    if dct_hf < 0.05:
        dct_s = 0.3
        signals.append("Abnormally low high-frequency DCT energy – image may be AI-generated.")
    elif dct_hf < 0.5:
        dct_s = 1.0
        signals.append("DCT frequency distribution is natural.")
    else:
        dct_s = 0.4
        signals.append("Unusually high DCT high-frequency energy – possible synthetic sharpening.")

    # Spectral flatness: natural images < 0.3; flat spectrum > 0.5 → synthetic
    if fft_flat > 0.6:
        fft_s = 0.2
        signals.append("Flat power spectrum detected – consistent with AI diffusion models.")
    elif fft_flat < 0.3:
        fft_s = 1.0
        signals.append("Power spectrum shows natural 1/f roll-off.")
    else:
        fft_s = 1.0 - fft_flat
        signals.append("Power spectrum is mildly flat.")

    # ELA uniformity: high = synthetic
    if ela_uni > 0.85:
        ela_s = 0.1
        signals.append("Highly uniform Error Level Analysis map – strong indicator of AI generation.")
    elif ela_uni > 0.60:
        ela_s = 0.5
        signals.append("Moderately uniform ELA map – possible AI generation or heavy smoothing.")
    else:
        ela_s = 1.0
        signals.append("Heterogeneous ELA map – consistent with authentic camera image.")

    # Colour entropy: natural images typically 4-6 bits
    if entropy < 2.0:
        ent_s = 0.2
        signals.append(f"Low colour entropy ({entropy:.2f} bits) – image may lack natural complexity.")
    elif entropy >= 4.0:
        ent_s = 1.0
        signals.append(f"Rich colour entropy ({entropy:.2f} bits) – consistent with real photography.")
    else:
        ent_s = entropy / 4.0

    weights = [0.25, 0.25, 0.30, 0.20]
    raw = weights[0] * dct_s + weights[1] * fft_s + weights[2] * ela_s + weights[3] * ent_s
    return round(float(np.clip(raw, 0.0, 1.0)), 4), signals

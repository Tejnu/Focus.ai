"""
Artifact Detector

Scans for structural and biological anomalies that are common in AI-generated
images but rarely appear in real photographs:

  1. Repetitive-texture detection  – AI models often tile or repeat texture
     patches (e.g., crowd duplication, bark repeats).
  2. Edge-consistency check        – real images have organic, physics-driven
     edges; diffusion outputs may show unnatural sharpness discontinuities.
  3. Local-colour-blending score   – diffusion models blend colours at object
     boundaries in ways that differ from optical lens bokeh.
  4. Symmetry anomaly detection    – detects unnatural bilateral symmetry that
     sometimes appears in AI-generated nature or crowd shots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from PIL import Image
from skimage.filters import sobel
from skimage.feature import local_binary_pattern


@dataclass
class ArtifactResult:
    repetitive_texture_score: float  # 0-1, higher = more repetition detected
    edge_consistency_score: float    # 0-1, higher = more natural edges
    colour_blend_score: float        # 0-1, higher = organic blending (real)
    symmetry_anomaly_score: float    # 0-1, higher = more symmetric (suspicious)
    score: float                     # overall 0-1 (1 = looks real, 0 = looks synthetic)
    signals: List[str] = field(default_factory=list)


def analyze(img: Image.Image) -> ArtifactResult:
    rgb = np.array(img.convert("RGB"), dtype=np.float32)
    gray = rgb.mean(axis=2)

    rep = _repetitive_texture(gray)
    edge = _edge_consistency(gray)
    blend = _colour_blend(rgb)
    sym = _symmetry_anomaly(gray)

    score, signals = _score(rep, edge, blend, sym)

    return ArtifactResult(
        repetitive_texture_score=round(rep, 4),
        edge_consistency_score=round(edge, 4),
        colour_blend_score=round(blend, 4),
        symmetry_anomaly_score=round(sym, 4),
        score=round(score, 4),
        signals=signals,
    )


# ---------------------------------------------------------------------------
# sub-analyses
# ---------------------------------------------------------------------------

def _repetitive_texture(gray: np.ndarray) -> float:
    """
    Use LBP histogram self-similarity across non-overlapping blocks to detect
    repeated texture patterns.
    """
    h, w = gray.shape
    block = min(64, h // 4, w // 4)
    if block < 8:
        return 0.5

    lbp_radius = 1
    lbp_points = 8
    histograms = []

    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            patch = gray[y:y + block, x:x + block].astype(np.uint8)
            lbp = local_binary_pattern(patch, lbp_points, lbp_radius, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=lbp_points + 2, range=(0, lbp_points + 2))
            hist = hist.astype(float)
            total = hist.sum()
            if total > 0:
                hist /= total
            histograms.append(hist)

    if len(histograms) < 2:
        return 0.5

    # Compute pairwise cosine similarity between non-adjacent histograms
    similarities = []
    n = len(histograms)
    step = max(1, n // 8)
    for i in range(0, n - step, step):
        a, b = histograms[i], histograms[i + step]
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom > 1e-9:
            sim = float(np.dot(a, b) / denom)
            similarities.append(sim)

    if not similarities:
        return 0.5

    mean_sim = float(np.mean(similarities))
    # High similarity across blocks → likely repetitive AI texture
    return float(np.clip(mean_sim, 0.0, 1.0))


def _edge_consistency(gray: np.ndarray) -> float:
    """
    Score how consistent / organic the edge distribution is.
    Real images have a smooth, roughly exponential distribution of edge magnitudes.
    AI images may show unnatural spikes (hyper-crisp edges) or lack of weak edges.
    Returns higher value for more natural (real) edge distribution.
    """
    edges = sobel(gray / 255.0).ravel()
    if edges.max() < 1e-6:
        return 0.5

    # Check ratio of very strong vs moderate edges
    strong = float(np.mean(edges > 0.15))
    weak = float(np.mean((edges > 0.01) & (edges <= 0.15)))

    if weak < 1e-6:
        return 0.3  # no gradual edges → synthetic
    ratio = strong / (weak + 1e-9)

    # Natural images: ratio typically 0.01 – 0.3
    if ratio < 0.5:
        score = 1.0 - ratio * 2.0
        return float(np.clip(score, 0.5, 1.0))
    else:
        return float(np.clip(1.0 - ratio / 5.0, 0.0, 0.5))


def _colour_blend(rgb: np.ndarray) -> float:
    """
    Measure how smoothly colours transition at edges.
    Real-world lens optics produce smooth chromatic aberration and bokeh gradients.
    AI models often produce over-sharp or unnatural colour boundaries.
    Returns higher value for smoother (more organic) blending.
    """
    # Compute gradient magnitude per channel, then measure their correlation
    grads = []
    for c in range(3):
        ch = rgb[:, :, c] / 255.0
        g = sobel(ch)
        grads.append(g.ravel())

    grads = np.array(grads)  # (3, N)

    # Guard: if all gradients are zero (flat image) return neutral score
    if np.all(grads == 0):
        return 0.5

    # Guard per-channel: replace zero-std channels with small noise to avoid NaN
    for i in range(grads.shape[0]):
        if grads[i].std() < 1e-9:
            grads[i] = grads[i] + np.finfo(float).eps

    # Coherence: all channels should change together at edges
    corr_matrix = np.corrcoef(grads)
    off_diag = [
        corr_matrix[i, j]
        for i in range(3) for j in range(3) if i != j
    ]
    # Replace any remaining NaN values with 0 (neutral)
    off_diag_clean = [v if not np.isnan(v) else 0.0 for v in off_diag]
    mean_corr = float(np.mean(off_diag_clean))

    # High channel-gradient correlation = coherent edges = real camera optics
    return float(np.clip((mean_corr + 1.0) / 2.0, 0.0, 1.0))


def _symmetry_anomaly(gray: np.ndarray) -> float:
    """
    Detect unusual bilateral symmetry.  Natural scenes are rarely symmetric;
    AI generators sometimes produce unnaturally symmetric images.
    Returns higher value for more symmetry (more suspicious).
    """
    # Flip and compare
    flipped_h = np.fliplr(gray)
    diff_h = np.abs(gray - flipped_h).mean()
    flipped_v = np.flipud(gray)
    diff_v = np.abs(gray - flipped_v).mean()

    # Normalise to image dynamic range
    dyn = gray.max() - gray.min()
    if dyn < 1e-6:
        return 0.5

    sym_h = 1.0 - float(np.clip(diff_h / dyn, 0.0, 1.0))
    sym_v = 1.0 - float(np.clip(diff_v / dyn, 0.0, 1.0))
    return float(max(sym_h, sym_v))


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _score(rep: float, edge: float, blend: float, sym: float) -> tuple[float, List[str]]:
    signals: List[str] = []

    if rep > 0.85:
        rep_s = 0.1
        signals.append("High texture repetition detected – common in AI-generated crowd or nature scenes.")
    elif rep > 0.70:
        rep_s = 0.5
        signals.append("Moderate texture repetition detected.")
    else:
        rep_s = 1.0
        signals.append("Texture variation is organic – consistent with real photography.")

    if edge < 0.4:
        edge_s = 0.3
        signals.append("Unnatural edge distribution detected – possible synthetic sharpening.")
    elif edge > 0.7:
        edge_s = 1.0
        signals.append("Edge distribution appears natural.")
    else:
        edge_s = edge

    if blend < 0.4:
        blend_s = 0.3
        signals.append("Poor colour blending at boundaries – may indicate AI generation.")
    else:
        blend_s = blend

    if sym > 0.92:
        signals.append("Near-perfect bilateral symmetry detected – suspicious for real photography.")
    elif sym > 0.80:
        signals.append("Above-average bilateral symmetry – check for mirrored AI generation.")

    weights = [0.30, 0.30, 0.20, 0.20]
    raw = weights[0] * rep_s + weights[1] * edge_s + weights[2] * blend_s + weights[3] * (1.0 - sym * 0.5)
    return round(float(np.clip(raw, 0.0, 1.0)), 4), signals

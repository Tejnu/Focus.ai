"""
Tests for the artifact detector module.
"""
import numpy as np
import pytest
from PIL import Image

from app.models.artifact_detector import analyze, ArtifactResult


def _natural_image(seed=7, size=(128, 128)) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(40, 220, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _tiled_image(size=(128, 128)) -> Image.Image:
    """Image made of repeated 16×16 tiles – high repetition."""
    tile = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    arr = np.tile(tile, (size[0] // 16, size[1] // 16, 1))
    return Image.fromarray(arr[:size[0], :size[1]], "RGB")


def _symmetric_image(size=(128, 128)) -> Image.Image:
    """Perfectly horizontally symmetric image."""
    half = np.random.default_rng(1).integers(0, 255, (size[0], size[1] // 2, 3), dtype=np.uint8)
    arr = np.concatenate([half, half[:, ::-1, :]], axis=1)
    return Image.fromarray(arr, "RGB")


def test_returns_artifact_result():
    result = analyze(_natural_image())
    assert isinstance(result, ArtifactResult)


def test_scores_in_range():
    for img in [_natural_image(), _tiled_image(), _symmetric_image()]:
        result = analyze(img)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.repetitive_texture_score <= 1.0
        assert 0.0 <= result.edge_consistency_score <= 1.0
        assert 0.0 <= result.colour_blend_score <= 1.0
        assert 0.0 <= result.symmetry_anomaly_score <= 1.0


def test_symmetric_image_has_higher_symmetry_score():
    sym = analyze(_symmetric_image())
    nat = analyze(_natural_image())
    assert sym.symmetry_anomaly_score > nat.symmetry_anomaly_score


def test_signals_present():
    result = analyze(_natural_image())
    assert len(result.signals) > 0
    assert all(isinstance(s, str) for s in result.signals)

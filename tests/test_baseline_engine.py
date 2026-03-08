"""
Tests for the comparative baseline engine module.
"""
import io
import numpy as np
import pytest
from PIL import Image

from app.models.baseline_engine import analyze, BaselineResult


def _random_image(seed=99, size=(128, 128)) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _smooth_image(size=(128, 128)) -> Image.Image:
    """Smooth gradient image simulating an over-smoothed output."""
    arr = np.zeros((*size, 3), dtype=np.uint8)
    for i in range(size[0]):
        val = int((i / size[0]) * 255)
        arr[i, :] = [val, 128, 255 - val]
    return Image.fromarray(arr, "RGB")


def test_returns_baseline_result():
    result = analyze(_random_image())
    assert isinstance(result, BaselineResult)


def test_scores_in_range():
    for img in [_random_image(), _smooth_image()]:
        result = analyze(img)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.dct_high_freq_ratio <= 1.0
        assert 0.0 <= result.fft_spectral_flatness <= 1.0
        assert 0.0 <= result.ela_uniformity <= 1.0
        assert result.colour_entropy >= 0.0


def test_colour_entropy_nonzero_for_varied_image():
    result = analyze(_random_image())
    assert result.colour_entropy > 0.0


def test_signals_are_strings():
    result = analyze(_random_image())
    assert all(isinstance(s, str) for s in result.signals)

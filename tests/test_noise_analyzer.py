"""
Tests for the noise analyzer module.
"""
import numpy as np
import pytest
from PIL import Image

from app.models.noise_analyzer import analyze, NoiseResult


def _uniform_image(color=(128, 128, 128), size=(64, 64)) -> Image.Image:
    """Completely flat – no noise at all."""
    arr = np.full((*size, 3), color, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _noisy_image(seed=42, size=(128, 128)) -> Image.Image:
    """Random pixel noise – simulates sensor noise."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _natural_gradient_image(size=(128, 128)) -> Image.Image:
    """Smooth gradient – simulates over-smoothed / AI image."""
    arr = np.zeros((*size, 3), dtype=np.uint8)
    for i in range(size[0]):
        for j in range(size[1]):
            val = int((i / size[0]) * 255)
            arr[i, j] = [val, val // 2, 255 - val]
    return Image.fromarray(arr, "RGB")


def test_returns_noise_result():
    img = _noisy_image()
    result = analyze(img)
    assert isinstance(result, NoiseResult)


def test_scores_in_range():
    for img in [_uniform_image(), _noisy_image(), _natural_gradient_image()]:
        result = analyze(img)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.local_variance_score <= 1.0
        assert 0.0 <= result.high_freq_energy_ratio <= 1.0
        assert 0.0 <= result.channel_noise_correlation <= 1.0
        assert result.noise_floor_std >= 0.0


def test_noisy_image_has_higher_noise_floor():
    noisy = analyze(_noisy_image())
    flat = analyze(_uniform_image())
    assert noisy.noise_floor_std > flat.noise_floor_std


def test_signals_are_non_empty():
    img = _noisy_image()
    result = analyze(img)
    assert len(result.signals) > 0
    assert all(isinstance(s, str) for s in result.signals)

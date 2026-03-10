"""
Tests for the metadata analyzer module.
"""
import io
import pytest
from PIL import Image

from app.models.metadata_analyzer import analyze, MetadataResult


def _make_plain_image() -> Image.Image:
    """Create a simple 64x64 RGB image with no EXIF data."""
    return Image.new("RGB", (64, 64), color=(100, 150, 200))


def _make_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_plain_image_no_exif():
    img = _make_plain_image()
    result = analyze(img)

    assert isinstance(result, MetadataResult)
    assert result.has_exif is False
    assert result.camera_make is None
    assert result.camera_model is None
    assert result.score == 0.0
    assert len(result.signals) > 0
    assert any("no exif" in s.lower() or "no" in s.lower() for s in result.signals)


def test_result_score_in_range():
    img = _make_plain_image()
    result = analyze(img)
    assert 0.0 <= result.score <= 1.0


def test_jpeg_round_trip_no_exif():
    """A freshly created JPEG also has no EXIF."""
    buf = io.BytesIO()
    img = _make_plain_image()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    loaded = Image.open(buf)
    loaded.load()

    result = analyze(loaded)
    assert result.has_exif is False
    assert result.score == 0.0


def test_raw_exif_is_serialisable():
    """raw_exif values must be JSON-serialisable types."""
    img = _make_plain_image()
    result = analyze(img)

    import json
    # Should not raise
    json.dumps(result.raw_exif)


def test_camera_fields_found_is_list():
    img = _make_plain_image()
    result = analyze(img)
    assert isinstance(result.camera_fields_found, list)

"""
Integration tests for the verification engine and the FastAPI endpoints.
"""
import io
import json
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.services.verification_engine import verify_image, VerificationResult


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(size=(128, 128), seed=42) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_png_bytes(size=(128, 128)) -> bytes:
    img = Image.new("RGB", size, color=(50, 100, 150))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# verification engine unit tests
# ---------------------------------------------------------------------------

def test_verify_image_returns_result():
    img = Image.fromarray(
        np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8),
        "RGB",
    )
    result = verify_image(img)
    assert isinstance(result, VerificationResult)


def test_result_score_in_range():
    img = Image.fromarray(
        np.random.default_rng(2).integers(0, 256, (64, 64, 3), dtype=np.uint8),
        "RGB",
    )
    result = verify_image(img)
    assert 0.0 <= result.authenticity_score <= 1.0


def test_result_tier_valid():
    img = Image.new("RGB", (64, 64))
    result = verify_image(img)
    valid = {"AUTHENTIC", "LIKELY_AUTHENTIC", "UNCERTAIN", "LIKELY_SYNTHETIC", "SYNTHETIC"}
    assert result.tier in valid


def test_result_confidence_valid():
    img = Image.new("RGB", (64, 64))
    result = verify_image(img)
    assert result.confidence in {"HIGH", "MEDIUM", "LOW"}


def test_result_has_module_scores():
    img = Image.new("RGB", (64, 64))
    result = verify_image(img)
    assert set(result.module_scores.keys()) == {"metadata", "noise", "artifact", "baseline"}
    for v in result.module_scores.values():
        assert 0.0 <= v <= 1.0


def test_result_signals_not_empty():
    img = Image.fromarray(
        np.random.default_rng(3).integers(0, 256, (64, 64, 3), dtype=np.uint8),
        "RGB",
    )
    result = verify_image(img)
    assert len(result.all_signals) > 0


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_verify_jpeg_upload():
    data = _make_jpeg_bytes()
    resp = client.post(
        "/api/v1/verify",
        files={"file": ("test.jpg", data, "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "authenticity_score" in body
    assert "tier" in body
    assert "recommendation" in body
    assert 0.0 <= body["authenticity_score"] <= 1.0


def test_verify_png_upload():
    data = _make_png_bytes()
    resp = client.post(
        "/api/v1/verify",
        files={"file": ("test.png", data, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "authenticity_score" in body


def test_verify_empty_file_returns_400():
    resp = client.post(
        "/api/v1/verify",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400


def test_verify_invalid_file_returns_422():
    resp = client.post(
        "/api/v1/verify",
        files={"file": ("bad.jpg", b"this is not an image", "image/jpeg")},
    )
    assert resp.status_code == 422


def test_verify_non_image_content_type_returns_415():
    resp = client.post(
        "/api/v1/verify",
        files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert resp.status_code == 415


def test_response_is_json_serialisable():
    data = _make_jpeg_bytes()
    resp = client.post(
        "/api/v1/verify",
        files={"file": ("test.jpg", data, "image/jpeg")},
    )
    # Should not raise
    body = json.loads(resp.text)
    assert isinstance(body, dict)

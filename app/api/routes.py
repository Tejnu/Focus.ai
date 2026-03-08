"""
API routes for the Focus.ai verification platform.
"""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.services.verification_engine import verify_image_bytes

router = APIRouter(prefix="/api/v1", tags=["verification"])


@router.get("/health")
async def health_check() -> dict:
    """Simple health-check endpoint."""
    return {"status": "ok", "service": "Focus.ai Verification Engine"}


@router.post("/verify")
async def verify_image(file: UploadFile = File(...)) -> JSONResponse:
    """
    Analyse an uploaded image and return a comprehensive authenticity report.

    The response includes:
    - **authenticity_score** – composite 0–1 score (1 = definitely real)
    - **tier** – AUTHENTIC / LIKELY_AUTHENTIC / UNCERTAIN / LIKELY_SYNTHETIC / SYNTHETIC
    - **confidence** – HIGH / MEDIUM / LOW
    - **recommendation** – actionable guidance for moderators
    - **module_scores** – individual scores per analysis module
    - **all_signals** – detailed findings from every module
    - **metadata / noise / artifact / baseline** – raw sub-module results
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Only image files are accepted (image/jpeg, image/png, etc.).",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = verify_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return JSONResponse(content=_serialise(result))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _serialise(obj) -> dict:
    """Convert a VerificationResult dataclass to a plain dict."""
    from dataclasses import asdict
    return asdict(obj)

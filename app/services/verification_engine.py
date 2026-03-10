"""
Core Generative Verification Engine

Orchestrates all analysis modules to produce a comprehensive authenticity
report for a single image.  This is the central service called by the API.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from PIL import Image

from app.models import (
    artifact_detector,
    authenticity_advisor,
    baseline_engine,
    metadata_analyzer,
    noise_analyzer,
)
from app.utils.image_utils import load_image, resize_for_analysis


@dataclass
class VerificationResult:
    """Full verification result returned by the engine."""
    authenticity_score: float
    tier: str
    confidence: str
    recommendation: str
    summary: str
    module_scores: Dict[str, float]
    all_signals: list
    metadata: Dict[str, Any]
    noise: Dict[str, Any]
    artifact: Dict[str, Any]
    baseline: Dict[str, Any]


def verify_image_bytes(image_data: bytes) -> VerificationResult:
    """
    Load *image_data* and run the full verification pipeline.

    Parameters
    ----------
    image_data:
        Raw bytes of the uploaded image file.

    Returns
    -------
    VerificationResult
        Comprehensive authenticity assessment.
    """
    img = load_image(image_data)
    return verify_image(img)


def verify_image(img: Image.Image) -> VerificationResult:
    """Run the full verification pipeline on an already-loaded PIL image."""
    # Resize for analysis modules that are resolution-sensitive
    img_small = resize_for_analysis(img, max_side=512)

    # Run all analysis modules
    meta_result = metadata_analyzer.analyze(img)          # uses original for EXIF
    noise_result = noise_analyzer.analyze(img_small)
    artifact_result = artifact_detector.analyze(img_small)
    baseline_result = baseline_engine.analyze(img_small)

    # Aggregate into a final report
    report = authenticity_advisor.advise(
        meta_result, noise_result, artifact_result, baseline_result
    )

    return VerificationResult(
        authenticity_score=report.authenticity_score,
        tier=report.tier,
        confidence=report.confidence,
        recommendation=report.recommendation,
        summary=report.summary,
        module_scores=report.module_scores,
        all_signals=report.all_signals,
        metadata=asdict(meta_result),
        noise=asdict(noise_result),
        artifact=asdict(artifact_result),
        baseline=asdict(baseline_result),
    )

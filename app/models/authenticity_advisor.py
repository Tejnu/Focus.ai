"""
AI Authenticity Advisor

Aggregates the scores from all analysis modules, computes a composite
authenticity score, assigns a confidence tier, and produces human-readable
recommendations for platform moderators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .metadata_analyzer import MetadataResult
from .noise_analyzer import NoiseResult
from .artifact_detector import ArtifactResult
from .baseline_engine import BaselineResult


# Weight of each module in the composite score
_WEIGHTS: Dict[str, float] = {
    "metadata": 0.20,
    "noise": 0.30,
    "artifact": 0.25,
    "baseline": 0.25,
}

# Thresholds for authenticity tiers
_TIER_THRESHOLDS = {
    "AUTHENTIC": 0.75,
    "LIKELY_AUTHENTIC": 0.55,
    "UNCERTAIN": 0.40,
    "LIKELY_SYNTHETIC": 0.25,
    # Below 0.25 → SYNTHETIC
}


@dataclass
class AuthenticityReport:
    authenticity_score: float       # 0-1 composite (1 = definitely real)
    tier: str                       # one of AUTHENTIC / LIKELY_AUTHENTIC / UNCERTAIN / LIKELY_SYNTHETIC / SYNTHETIC
    confidence: str                 # HIGH / MEDIUM / LOW
    recommendation: str             # action for platform moderators
    module_scores: Dict[str, float]
    all_signals: List[str]
    summary: str
    metadata_weight: float = _WEIGHTS["metadata"]
    noise_weight: float = _WEIGHTS["noise"]
    artifact_weight: float = _WEIGHTS["artifact"]
    baseline_weight: float = _WEIGHTS["baseline"]


def advise(
    metadata: MetadataResult,
    noise: NoiseResult,
    artifact: ArtifactResult,
    baseline: BaselineResult,
) -> AuthenticityReport:
    """Combine sub-module results into a full :class:`AuthenticityReport`."""
    module_scores = {
        "metadata": metadata.score,
        "noise": noise.score,
        "artifact": artifact.score,
        "baseline": baseline.score,
    }
    # Replace any NaN scores (degenerate image) with 0.0
    module_scores = {
        k: (v if v == v else 0.0) for k, v in module_scores.items()
    }

    composite = sum(
        _WEIGHTS[k] * v for k, v in module_scores.items()
    )
    # Guard against NaN propagation from degenerate images (e.g., solid colour)
    if not isinstance(composite, float) or composite != composite:  # NaN check
        composite = 0.0
    composite = round(float(composite), 4)

    tier = _tier(composite)
    confidence = _confidence(module_scores)
    recommendation = _recommendation(tier, metadata, noise, artifact, baseline)
    all_signals = (
        metadata.signals
        + noise.signals
        + artifact.signals
        + baseline.signals
    )
    summary = _summary(tier, composite, confidence, metadata)

    return AuthenticityReport(
        authenticity_score=composite,
        tier=tier,
        confidence=confidence,
        recommendation=recommendation,
        module_scores=module_scores,
        all_signals=all_signals,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tier(score: float) -> str:
    if score >= _TIER_THRESHOLDS["AUTHENTIC"]:
        return "AUTHENTIC"
    if score >= _TIER_THRESHOLDS["LIKELY_AUTHENTIC"]:
        return "LIKELY_AUTHENTIC"
    if score >= _TIER_THRESHOLDS["UNCERTAIN"]:
        return "UNCERTAIN"
    if score >= _TIER_THRESHOLDS["LIKELY_SYNTHETIC"]:
        return "LIKELY_SYNTHETIC"
    return "SYNTHETIC"


def _confidence(scores: Dict[str, float]) -> str:
    """Confidence is HIGH when modules agree, LOW when they disagree."""
    values = list(scores.values())
    spread = max(values) - min(values)
    if spread < 0.20:
        return "HIGH"
    if spread < 0.40:
        return "MEDIUM"
    return "LOW"


def _recommendation(
    tier: str,
    metadata: MetadataResult,
    noise: NoiseResult,
    artifact: ArtifactResult,
    baseline: BaselineResult,
) -> str:
    if tier == "AUTHENTIC":
        return (
            "Image passes all authenticity checks. "
            "Approve upload and mark as verified."
        )
    if tier == "LIKELY_AUTHENTIC":
        return (
            "Image is likely authentic. "
            "Recommend a quick manual spot-check before final approval."
        )
    if tier == "UNCERTAIN":
        return (
            "Authenticity could not be determined with confidence. "
            "Escalate to a human reviewer for manual inspection. "
            "Consider requesting the original RAW file from the uploader."
        )
    if tier == "LIKELY_SYNTHETIC":
        return (
            "Image shows multiple indicators of AI generation. "
            "Flag this upload and notify the uploader that the content "
            "may violate synthetic-media policy. Do not approve until verified."
        )
    # SYNTHETIC
    return (
        "Image is very likely AI-generated or artificially modified. "
        "Reject the upload, apply a synthetic-media label, and log "
        "the incident for compliance reporting."
    )


def _summary(
    tier: str,
    score: float,
    confidence: str,
    metadata: MetadataResult,
) -> str:
    pct = int(score * 100)
    cam = (
        f"from a {metadata.camera_make} {metadata.camera_model}"
        if metadata.camera_make and metadata.camera_model
        else ("with camera metadata present" if metadata.has_exif else "with no camera metadata")
    )
    tier_label = {
        "AUTHENTIC": "authentic",
        "LIKELY_AUTHENTIC": "likely authentic",
        "UNCERTAIN": "of uncertain origin",
        "LIKELY_SYNTHETIC": "likely synthetic",
        "SYNTHETIC": "synthetic",
    }[tier]
    return (
        f"This image is assessed as {tier_label} (score {pct}/100, {confidence.lower()} confidence). "
        f"It appears to originate {cam}."
    )

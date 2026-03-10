"""
Metadata Analyzer – inspects EXIF / image metadata to look for real-camera
fingerprints (camera model, lens, GPS, timestamps, etc.).

A genuine photograph captured by a physical camera typically carries rich EXIF
data.  AI-generated images almost never embed EXIF at all, and simple image
editors/diffusion-model outputs usually strip or omit it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image
from PIL.ExifTags import TAGS


# EXIF tag ids for the fields we care about most
_CAMERA_TAGS = {
    "Make", "Model", "LensModel", "LensMake",
    "FocalLength", "ExposureTime", "FNumber", "ISOSpeedRatings",
    "Flash", "WhiteBalance", "MeteringMode",
}
_TIMESTAMP_TAGS = {"DateTime", "DateTimeOriginal", "DateTimeDigitized"}
_GPS_TAG_ID = 34853  # GPSInfo


@dataclass
class MetadataResult:
    has_exif: bool
    camera_make: Optional[str]
    camera_model: Optional[str]
    has_gps: bool
    has_timestamps: bool
    camera_fields_found: List[str]
    raw_exif: Dict[str, Any]
    score: float  # 0 = likely synthetic, 1 = strong camera evidence
    signals: List[str] = field(default_factory=list)


def analyze(img: Image.Image) -> MetadataResult:
    """Return a :class:`MetadataResult` for *img*."""
    exif_data: Dict[str, Any] = {}

    try:
        raw = img._getexif()  # type: ignore[attr-defined]
        if raw:
            exif_data = {
                TAGS.get(tag_id, str(tag_id)): value
                for tag_id, value in raw.items()
            }
    except (AttributeError, Exception):
        pass

    has_exif = bool(exif_data)
    camera_make = exif_data.get("Make")
    camera_model = exif_data.get("Model")
    has_gps = bool(exif_data.get("GPSInfo")) or ("GPSInfo" in exif_data)
    has_timestamps = bool(_TIMESTAMP_TAGS & exif_data.keys())
    camera_fields_found = [t for t in _CAMERA_TAGS if t in exif_data]

    score, signals = _score(
        has_exif, camera_fields_found, has_gps, has_timestamps,
        camera_make, camera_model,
    )

    return MetadataResult(
        has_exif=has_exif,
        camera_make=_clean_str(camera_make),
        camera_model=_clean_str(camera_model),
        has_gps=has_gps,
        has_timestamps=has_timestamps,
        camera_fields_found=camera_fields_found,
        raw_exif={k: _serialise(v) for k, v in exif_data.items()},
        score=score,
        signals=signals,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _score(
    has_exif: bool,
    camera_fields: List[str],
    has_gps: bool,
    has_timestamps: bool,
    make: Any,
    model: Any,
) -> tuple[float, List[str]]:
    signals: List[str] = []
    points = 0.0
    max_points = 6.0

    if not has_exif:
        signals.append("No EXIF data found – typical of AI-generated images.")
        return 0.0, signals

    signals.append("EXIF data is present.")
    points += 1.0

    if make:
        signals.append(f"Camera manufacturer detected: {_clean_str(make)}.")
        points += 1.5
    if model:
        signals.append(f"Camera model detected: {_clean_str(model)}.")
        points += 1.5
    if has_timestamps:
        signals.append("Capture timestamps are embedded.")
        points += 1.0
    if has_gps:
        signals.append("GPS location data is embedded.")
        points += 1.0

    if len(camera_fields) >= 5:
        signals.append(f"{len(camera_fields)} optical/exposure fields found (strong camera fingerprint).")
    elif camera_fields:
        signals.append(f"{len(camera_fields)} optical/exposure fields found.")

    return round(min(points / max_points, 1.0), 4), signals


def _clean_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().strip("\x00")
    return s if s else None


def _serialise(value: Any) -> Any:
    """Make EXIF values JSON-serialisable."""
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, tuple):
        return list(value)
    return str(value) if not isinstance(value, (int, float, str, bool, type(None))) else value

"""ML geo-estimation — Track 2, the no-metadata workhorse (Phase 2).

Uses **GeoCLIP** (https://github.com/Vicente-Vivanco/GeoCLIP) — a CLIP-based
model that predicts GPS coordinates directly from an image, with no training or
reference gallery of your own required. This is what actually handles photos
that have their EXIF stripped (i.e. most social-media images).

Design principles:
  * **Lazy + optional.** torch and the model weights are heavy, so nothing is
    imported or downloaded until this stage actually runs. If the ML extras
    aren't installed, the stage degrades gracefully (like OCR does) — the core
    tool stays lightweight.
  * **Honest confidence.** A model guess is never GPS. We derive confidence and
    precision from how *geographically concentrated* the top-k predictions are:
    tightly clustered predictions → city-level and more trusted; scattered ones
    → country-level at best. Confidence is capped well below a GPS fix.

Install the extras with:  pip install -r requirements-ml.txt
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .models import Coordinates, Precision

# Module-level model cache — loading GeoCLIP is expensive; do it once.
_model = None
_load_error: Optional[str] = None


@dataclass
class GeoPrediction:
    coordinates: Coordinates
    probability: float


@dataclass
class GeoEstimateResult:
    available: bool
    predictions: list[GeoPrediction] = field(default_factory=list)
    reason: Optional[str] = None

    @property
    def top(self) -> Optional[GeoPrediction]:
        return self.predictions[0] if self.predictions else None


def ml_available() -> bool:
    """True if the ML stack can be imported (does not download weights)."""
    try:
        import torch  # noqa: F401
        import geoclip  # noqa: F401

        return True
    except Exception:
        return False


def _load_model():
    """Load (and cache) the GeoCLIP model. Weights download on first call."""
    global _model, _load_error
    if _model is not None or _load_error is not None:
        return _model
    try:
        from geoclip import GeoCLIP

        _model = GeoCLIP()
    except Exception as exc:  # download failure, torch issue, etc.
        _load_error = f"failed to load GeoCLIP model: {exc}"
        _model = None
    return _model


def _haversine_km(a: Coordinates, b: Coordinates) -> float:
    """Great-circle distance between two points, in kilometres."""
    r = 6371.0
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dphi = math.radians(b.lat - a.lat)
    dlmb = math.radians(b.lon - a.lon)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


def predict(path: str, top_k: int = 5) -> GeoEstimateResult:
    """Estimate coordinates for an image. Never raises on the normal
    'not installed' path."""
    if not ml_available():
        return GeoEstimateResult(
            available=False,
            reason="ML extras not installed (pip install -r requirements-ml.txt)",
        )

    model = _load_model()
    if model is None:
        return GeoEstimateResult(available=False, reason=_load_error or "model unavailable")

    try:
        top_gps, top_prob = model.predict(path, top_k=top_k)
    except Exception as exc:
        return GeoEstimateResult(available=False, reason=f"GeoCLIP inference failed: {exc}")

    preds: list[GeoPrediction] = []
    try:
        for i in range(len(top_gps)):
            lat = float(top_gps[i][0])
            lon = float(top_gps[i][1])
            prob = float(top_prob[i])
            preds.append(GeoPrediction(Coordinates(lat, lon), prob))
    except (TypeError, IndexError, ValueError) as exc:
        return GeoEstimateResult(available=False, reason=f"unexpected model output: {exc}")

    return GeoEstimateResult(available=True, predictions=preds)


def assess(result: GeoEstimateResult) -> tuple[Precision, float, float]:
    """Translate a raw prediction set into (precision, confidence, spread_km).

    Confidence reflects agreement among the top predictions, not the model's raw
    probability (which is tiny by construction — GeoCLIP scores against a huge
    coordinate gallery). Tight clustering ⇒ higher precision and trust.
    """
    if not result.predictions:
        return Precision.UNKNOWN, 0.0, float("inf")

    top = result.predictions[0].coordinates
    # Spread = furthest of the other top-k predictions from the #1 pick.
    spread = 0.0
    for p in result.predictions[1:]:
        spread = max(spread, _haversine_km(top, p.coordinates))

    # Concentration -> precision + a capped, deliberately conservative confidence.
    # ML alone must never rival a GPS fix (0.95); we cap at 0.6.
    if spread < 25:
        return Precision.CITY, 0.60, spread
    if spread < 250:
        return Precision.REGION, 0.45, spread
    if spread < 1500:
        return Precision.COUNTRY, 0.35, spread
    return Precision.COUNTRY, 0.25, spread

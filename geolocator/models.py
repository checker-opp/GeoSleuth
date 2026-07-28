"""Shared data structures for the geolocation pipeline.

Every stage of the pipeline emits ``Signal`` objects. A signal is one piece
of evidence about where a photo was taken, together with how much we trust it.
The final ``GeoResult`` aggregates all signals into a single best guess plus a
transparent evidence trail — we never return a bare location without showing
the reasoning that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class Precision(str, Enum):
    """How specific a location signal is — coarse hints must never masquerade
    as a pinpoint."""

    EXACT = "exact"        # precise coordinates (e.g. EXIF GPS)
    CITY = "city"          # narrowed to a city / locality
    REGION = "region"      # state / province / broad area
    COUNTRY = "country"    # country-level only
    UNKNOWN = "unknown"    # a clue with no geographic resolution yet


# Rough ordering so the aggregator can compare "how specific" two signals are.
_PRECISION_RANK = {
    Precision.EXACT: 4,
    Precision.CITY: 3,
    Precision.REGION: 2,
    Precision.COUNTRY: 1,
    Precision.UNKNOWN: 0,
}


@dataclass
class Coordinates:
    lat: float
    lon: float

    def __str__(self) -> str:
        return f"{self.lat:.6f}, {self.lon:.6f}"


@dataclass
class Signal:
    """One piece of location evidence from a single pipeline stage."""

    source: str                          # which stage produced it, e.g. "exif"
    description: str                     # human-readable summary of the finding
    confidence: float                   # 0.0–1.0 trust in THIS signal
    precision: Precision = Precision.UNKNOWN
    coordinates: Optional[Coordinates] = None
    place: Optional[str] = None          # resolved place name, if any
    # True only for *independent locating* evidence — a signal that on its own
    # points at WHERE the photo was taken (EXIF GPS, an ML estimate, an OCR
    # place/language, a plate country). These may boost confidence when they
    # agree. False for *enrichment* signals (nearby OSM features, "street
    # imagery exists", climate zone) — they confirm a coordinate is a real,
    # photographable place, NOT that this image was taken there, so they add
    # evidence but must never inflate confidence.
    corroborating: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)  # raw supporting data

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["precision"] = self.precision.value
        if self.coordinates is not None:
            d["coordinates"] = {"lat": self.coordinates.lat, "lon": self.coordinates.lon}
        return d


@dataclass
class GeoResult:
    """The final answer: best-guess location, overall confidence, and every
    signal that contributed."""

    image_path: str
    signals: list[Signal] = field(default_factory=list)
    best_coordinates: Optional[Coordinates] = None
    best_place: Optional[str] = None
    best_precision: Precision = Precision.UNKNOWN
    overall_confidence: float = 0.0
    notes: list[str] = field(default_factory=list)
    # Scratch space for cross-stage handoff (e.g. EXIF timestamp -> solar stage).
    # Never serialized in the public result.
    meta: dict[str, Any] = field(default_factory=dict)

    def add(self, signal: Optional[Signal]) -> None:
        if signal is not None:
            self.signals.append(signal)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "best_guess": {
                "place": self.best_place,
                "coordinates": (
                    {"lat": self.best_coordinates.lat, "lon": self.best_coordinates.lon}
                    if self.best_coordinates
                    else None
                ),
                "precision": self.best_precision.value,
                "confidence": round(self.overall_confidence, 3),
            },
            "signals": [s.to_dict() for s in self.signals],
            "notes": self.notes,
            "pivots": self.meta.get("pivots", []),
        }


def precision_rank(p: Precision) -> int:
    return _PRECISION_RANK.get(p, 0)

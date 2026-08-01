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
    # A *locating* signal proposes WHERE the photo was taken (EXIF GPS, ML
    # estimate, OCR place/language, plate country) and is eligible to win the
    # best-location slot. Enrichment signals (nearby OSM features, "street
    # imagery exists", climate zone) set this False: they describe/corroborate a
    # candidate but must never become the chosen location — they carry no place
    # of their own and would hijack the slot from a real locator.
    locating: bool = True
    # Among locating signals, True marks *independent secondary* evidence that
    # may boost confidence when it agrees (OCR language, plate country). The
    # primary locators (EXIF/ML) leave this False — they set the base
    # confidence rather than adding a bonus to themselves. Enrichment signals
    # are always False here too.
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


@dataclass
class AnalyzeConfig:
    """Which signals to run + API keys — lets a query be customized.

    Defaults reproduce the classic pipeline. Toggle stages off (e.g. skip the
    heavy GeoCLIP model when relying on the GeoSeer API) or pass keys directly
    instead of via environment variables.
    """

    use_ocr: bool = True
    use_geoclip: bool = True          # local CLIP model (required)
    use_geoseer: bool = True          # AI API; self-skips if no key is available
    use_streetclip: bool = True       # 2nd model — ON by default (required setup)
    use_place_lookup: bool = True
    use_reverse_search: bool = True   # SerpAPI Google Lens; self-skips without a key
    use_street_match: bool = True
    use_osm: bool = True
    use_inaturalist: bool = True
    use_solar: bool = True

    # When GeoSeer returns a confident fix, skip the slow local GeoCLIP/StreetCLIP
    # models — GeoSeer already located it, so there's no need to also run them.
    short_circuit_on_geoseer: bool = True

    geoseer_key: Optional[str] = None       # overrides GEOSEER_API_KEY
    mapillary_token: Optional[str] = None   # overrides MAPILLARY_TOKEN
    serpapi_key: Optional[str] = None       # overrides SERPAPI_API_KEY

    @classmethod
    def from_env(cls) -> "AnalyzeConfig":
        """Default config. GeoCLIP + StreetCLIP are both on by default (the
        required local setup). GEOLOCATOR_STREETCLIP=0 can still disable it."""
        import os

        env = os.environ.get("GEOLOCATOR_STREETCLIP")
        cfg = cls()
        if env is not None and env in ("0", "false", "False", ""):
            cfg.use_streetclip = False
        return cfg

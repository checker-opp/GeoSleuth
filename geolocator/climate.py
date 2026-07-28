"""Coarse climate / latitude-zone descriptor — Phase 3 (enrichment).

The spec's intended signal is *flora in the image → climate zone → region*.
Extracting plant species from a photo needs a vision model (a future add), so
what we build now is the honest, buildable half: given a **candidate
coordinate**, describe its climate band. This doesn't locate anything — it
enriches the evidence and gives a sanity anchor ("a snowy scene shouldn't place
in the tropics").

Pure, offline, zero-dependency — a coarse latitude banding, not true Köppen
(which needs temperature/precipitation data).
"""

from __future__ import annotations

from .models import Coordinates


def hemisphere(coord: Coordinates) -> str:
    ns = "Northern" if coord.lat >= 0 else "Southern"
    ew = "Eastern" if coord.lon >= 0 else "Western"
    return f"{ns}/{ew} hemisphere"


def latitude_zone(lat: float) -> str:
    a = abs(lat)
    if a < 10:
        return "equatorial (tropical rainforest / monsoon)"
    if a < 23.5:
        return "tropical"
    if a < 35:
        return "subtropical"
    if a < 55:
        return "temperate"
    if a < 66.5:
        return "subpolar / boreal"
    return "polar"


def describe(coord: Coordinates) -> dict[str, str]:
    return {
        "latitude_zone": latitude_zone(coord.lat),
        "hemisphere": hemisphere(coord),
        "season_hint": _season_hint(coord.lat),
    }


def _season_hint(lat: float) -> str:
    """Which hemisphere's seasons apply — helps interpret vegetation/snow."""
    if lat >= 0:
        return "Northern seasons (summer ~Jun–Aug, winter ~Dec–Feb)"
    return "Southern seasons (summer ~Dec–Feb, winter ~Jun–Aug)"

"""Street-level imagery corroboration via Mapillary — Phase 4.

Given a candidate coordinate, ask Mapillary whether street-level photos exist
nearby. Coverage near the guess corroborates that it's a real, photographable
place and gives the investigator ready pivots to *visually* compare the target
photo against ground-level imagery — the manual step that actually confirms a
street-level match.

Requires a free Mapillary access token (https://www.mapillary.com/dashboard/developers)
supplied via the ``MAPILLARY_TOKEN`` environment variable. Without it, this
stage is skipped gracefully. KartaView (keyless) is a possible future addition.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import requests

from .models import Coordinates

GRAPH_URL = "https://graph.mapillary.com/images"
TOKEN_ENV = "MAPILLARY_TOKEN"


@dataclass
class StreetImage:
    id: str
    lat: float
    lon: float
    captured_at: Optional[int] = None

    @property
    def viewer_url(self) -> str:
        return f"https://www.mapillary.com/app/?pKey={self.id}&focus=photo"


@dataclass
class StreetMatchResult:
    available: bool
    images: list[StreetImage] = field(default_factory=list)
    reason: Optional[str] = None


def _bbox(coord: Coordinates, radius_m: float) -> tuple[float, float, float, float]:
    """Approximate lon/lat bounding box for a radius in metres."""
    import math

    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * max(0.01, math.cos(math.radians(coord.lat))))
    return (coord.lon - dlon, coord.lat - dlat, coord.lon + dlon, coord.lat + dlat)


def token_configured() -> bool:
    return bool(os.environ.get(TOKEN_ENV))


def nearby(
    coord: Coordinates,
    radius_m: float = 100.0,
    limit: int = 5,
    timeout: float = 20.0,
) -> StreetMatchResult:
    token = os.environ.get(TOKEN_ENV)
    if not token:
        return StreetMatchResult(
            available=False,
            reason=f"no {TOKEN_ENV} set (free Mapillary token enables street matching)",
        )

    west, south, east, north = _bbox(coord, radius_m)
    try:
        resp = requests.get(
            GRAPH_URL,
            params={
                "access_token": token,
                "fields": "id,computed_geometry,captured_at",
                "bbox": f"{west},{south},{east},{north}",
                "limit": limit,
            },
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return StreetMatchResult(available=False, reason=f"Mapillary request failed: {exc}")
    if resp.status_code != 200:
        return StreetMatchResult(available=False, reason=f"Mapillary HTTP {resp.status_code}")
    try:
        data = resp.json().get("data", [])
    except ValueError:
        return StreetMatchResult(available=False, reason="Mapillary returned invalid JSON")

    images: list[StreetImage] = []
    for item in data:
        geom = (item.get("computed_geometry") or {}).get("coordinates")
        lon, lat = (geom[0], geom[1]) if geom and len(geom) == 2 else (coord.lon, coord.lat)
        images.append(
            StreetImage(
                id=str(item.get("id")),
                lat=lat,
                lon=lon,
                captured_at=item.get("captured_at"),
            )
        )
    return StreetMatchResult(available=True, images=images)

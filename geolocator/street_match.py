"""Street-level imagery corroboration — Phase 4.

Given a candidate coordinate, check whether street-level photos exist nearby.
Coverage near the guess corroborates that it's a real, photographable place and
gives the investigator ready pivots to *visually* compare the target photo
against ground-level imagery — the manual step that actually confirms a match.

Two providers, tried in order:
  1. **Mapillary** — broad global coverage, needs a free access token
     (``MAPILLARY_TOKEN`` env var; https://www.mapillary.com/dashboard/developers).
  2. **KartaView** — keyless, community-run; strong in Europe, sparse elsewhere.
     Used as a fallback so street matching still works with no token at all.

Both fail gracefully (missing token, rate limit, downtime, no coverage).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import requests

from . import __version__
from .models import Coordinates

MAPILLARY_URL = "https://graph.mapillary.com/images"
KARTAVIEW_URL = "https://api.openstreetcam.org/1.0/list/nearby-photos/"
TOKEN_ENV = "MAPILLARY_TOKEN"
USER_AGENT = f"geolocator-osint/{__version__} (image geolocation research tool)"


@dataclass
class StreetImage:
    id: str
    lat: float
    lon: float
    provider: str
    viewer_url: str
    captured_at: Optional[str] = None


@dataclass
class StreetMatchResult:
    available: bool
    provider: Optional[str] = None
    images: list[StreetImage] = field(default_factory=list)
    reason: Optional[str] = None


def token_configured(token: Optional[str] = None) -> bool:
    return bool(token or os.environ.get(TOKEN_ENV))


def _bbox(coord: Coordinates, radius_m: float) -> tuple[float, float, float, float]:
    """Approximate lon/lat bounding box for a radius in metres."""
    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * max(0.01, math.cos(math.radians(coord.lat))))
    return (coord.lon - dlon, coord.lat - dlat, coord.lon + dlon, coord.lat + dlat)


def _mapillary_viewer_url(image_id: str) -> str:
    return f"https://www.mapillary.com/app/?pKey={image_id}&focus=photo"


def _kartaview_viewer_url(sequence_id, sequence_index) -> str:
    return f"https://kartaview.org/details/{sequence_id}/{sequence_index}"


# --------------------------------------------------------------------------- #
# Provider 1: Mapillary (token)
# --------------------------------------------------------------------------- #
def mapillary_nearby(
    coord: Coordinates,
    radius_m: float = 100.0,
    limit: int = 5,
    timeout: float = 20.0,
    token: Optional[str] = None,
) -> StreetMatchResult:
    token = token or os.environ.get(TOKEN_ENV)
    if not token:
        return StreetMatchResult(
            available=False,
            provider="mapillary",
            reason=f"no {TOKEN_ENV} set",
        )

    west, south, east, north = _bbox(coord, radius_m)
    try:
        resp = requests.get(
            MAPILLARY_URL,
            params={
                "access_token": token,
                "fields": "id,computed_geometry,captured_at",
                "bbox": f"{west},{south},{east},{north}",
                "limit": limit,
            },
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return StreetMatchResult(available=False, provider="mapillary",
                                 reason=f"Mapillary request failed: {exc}")
    if resp.status_code != 200:
        return StreetMatchResult(available=False, provider="mapillary",
                                 reason=f"Mapillary HTTP {resp.status_code}")
    try:
        data = resp.json().get("data", [])
    except ValueError:
        return StreetMatchResult(available=False, provider="mapillary",
                                 reason="Mapillary returned invalid JSON")

    images: list[StreetImage] = []
    for item in data:
        geom = (item.get("computed_geometry") or {}).get("coordinates")
        lon, lat = (geom[0], geom[1]) if geom and len(geom) == 2 else (coord.lon, coord.lat)
        iid = str(item.get("id"))
        images.append(
            StreetImage(
                id=iid, lat=lat, lon=lon, provider="Mapillary",
                viewer_url=_mapillary_viewer_url(iid),
                captured_at=str(item.get("captured_at")) if item.get("captured_at") else None,
            )
        )
    return StreetMatchResult(available=True, provider="mapillary", images=images)


# --------------------------------------------------------------------------- #
# Provider 2: KartaView (keyless)
# --------------------------------------------------------------------------- #
def kartaview_nearby(
    coord: Coordinates,
    radius_m: float = 100.0,
    limit: int = 5,
    timeout: float = 20.0,
) -> StreetMatchResult:
    try:
        resp = requests.post(
            KARTAVIEW_URL,
            data={"lat": coord.lat, "lng": coord.lon, "radius": int(radius_m)},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return StreetMatchResult(available=False, provider="kartaview",
                                 reason=f"KartaView request failed: {exc}")
    if resp.status_code != 200:
        return StreetMatchResult(available=False, provider="kartaview",
                                 reason=f"KartaView HTTP {resp.status_code}")
    try:
        items = resp.json().get("currentPageItems", [])
    except ValueError:
        return StreetMatchResult(available=False, provider="kartaview",
                                 reason="KartaView returned invalid JSON")

    images: list[StreetImage] = []
    for item in items[:limit]:
        try:
            lat = float(item.get("lat"))
            lon = float(item.get("lng"))
        except (TypeError, ValueError):
            continue
        images.append(
            StreetImage(
                id=str(item.get("id")),
                lat=lat,
                lon=lon,
                provider="KartaView",
                viewer_url=_kartaview_viewer_url(
                    item.get("sequence_id"), item.get("sequence_index")
                ),
                captured_at=item.get("shot_date"),
            )
        )
    return StreetMatchResult(available=True, provider="kartaview", images=images)


# --------------------------------------------------------------------------- #
# Unified: Mapillary first (if token), else / also KartaView
# --------------------------------------------------------------------------- #
def find_street_imagery(
    coord: Coordinates, radius_m: float = 100.0, limit: int = 5,
    token: Optional[str] = None,
) -> StreetMatchResult:
    """Return the first provider that yields imagery, preferring Mapillary when a
    token is configured and falling back to keyless KartaView otherwise."""
    reasons: list[str] = []

    if token_configured(token):
        res = mapillary_nearby(coord, radius_m, limit, token=token)
        if res.available and res.images:
            return res
        reasons.append(res.reason or "Mapillary: no imagery nearby")

    res = kartaview_nearby(coord, radius_m, limit)
    if res.available and res.images:
        return res
    reasons.append(res.reason or "KartaView: no imagery nearby")

    # Nothing found. Report available=False but explain both providers' outcomes.
    return StreetMatchResult(available=False, provider=None, reason="; ".join(reasons))

"""Overpass (OpenStreetMap) cross-reference — Phase 3 (corroboration).

Given a candidate coordinate, ask the free Overpass API what named features sit
nearby. This corroborates that the coordinate lands on a real, plausible place
and enriches the evidence with recognizable anchors (a named building, park, or
transit stop near the guess). It does not by itself locate an image.

Free and keyless, but a shared community resource — we keep the query small,
time-boxed, and fail silently on any error (rate limit, timeout, downtime).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import requests

from . import __version__
from .models import Coordinates

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = f"geolocator-osint/{__version__} (image geolocation research tool)"


@dataclass
class OsmFeature:
    name: str
    kind: str  # e.g. "amenity=cafe", "tourism=attraction"


@dataclass
class OsmResult:
    available: bool
    features: list[OsmFeature] = field(default_factory=list)
    reason: Optional[str] = None


def nearby(
    coord: Coordinates,
    radius_m: int = 300,
    limit: int = 8,
    timeout: float = 25.0,
) -> OsmResult:
    """Return notable named OSM features within ``radius_m`` of the coordinate."""
    # Named nodes/ways tagged with a handful of location-anchoring keys.
    query = f"""
    [out:json][timeout:20];
    (
      node(around:{radius_m},{coord.lat},{coord.lon})["name"]["amenity"];
      node(around:{radius_m},{coord.lat},{coord.lon})["name"]["tourism"];
      node(around:{radius_m},{coord.lat},{coord.lon})["name"]["shop"];
      way(around:{radius_m},{coord.lat},{coord.lon})["name"]["building"];
      node(around:{radius_m},{coord.lat},{coord.lon})["name"]["railway"="station"];
    );
    out center {limit * 3};
    """
    try:
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return OsmResult(available=False, reason=f"Overpass request failed: {exc}")
    if resp.status_code != 200:
        return OsmResult(available=False, reason=f"Overpass HTTP {resp.status_code}")
    try:
        elements = resp.json().get("elements", [])
    except ValueError:
        return OsmResult(available=False, reason="Overpass returned invalid JSON")

    seen: set[str] = set()
    features: list[OsmFeature] = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        kind = None
        for key in ("tourism", "amenity", "shop", "railway", "building"):
            if key in tags:
                kind = f"{key}={tags[key]}"
                break
        features.append(OsmFeature(name=name, kind=kind or "feature"))
        if len(features) >= limit:
            break

    return OsmResult(available=True, features=features)

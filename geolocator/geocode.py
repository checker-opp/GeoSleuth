"""Reverse geocoding via Nominatim (OpenStreetMap) — free, no API key.

Nominatim's usage policy is strict and we honour it:
  * a descriptive, identifying User-Agent (required — default/empty UAs are blocked)
  * at most 1 request/second (enforced here with a module-level throttle)

For any real volume you should self-host Nominatim or use a paid geocoder;
this client is fine for a one-image-at-a-time CLI.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import requests

from . import __version__
from .models import Coordinates

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = f"geolocator-osint/{__version__} (image geolocation research tool)"
_MIN_INTERVAL = 1.05  # seconds between requests — a little over the 1/sec limit

_last_request_at = 0.0
_throttle_lock = threading.Lock()


@dataclass
class Place:
    display_name: str
    country: Optional[str] = None
    country_code: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None


@dataclass
class SearchHit:
    display_name: str
    lat: float
    lon: float
    category: Optional[str] = None   # OSM class, e.g. "shop", "amenity"
    importance: float = 0.0          # Nominatim's relevance score (0–1)


def _throttle() -> None:
    global _last_request_at
    with _throttle_lock:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _last_request_at = time.monotonic()


def reverse(coords: Coordinates, timeout: float = 15.0) -> Optional[Place]:
    """Turn coordinates into a human-readable place, or None on failure."""
    _throttle()
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={
                "lat": coords.lat,
                "lon": coords.lon,
                "format": "json",
                "zoom": 18,
                "addressdetails": 1,
            },
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
            timeout=timeout,
        )
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if "error" in data:
        return None

    addr = data.get("address", {})
    city = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("municipality")
        or addr.get("hamlet")
    )
    return Place(
        display_name=data.get("display_name", str(coords)),
        country=addr.get("country"),
        country_code=(addr.get("country_code") or "").upper() or None,
        state=addr.get("state") or addr.get("region"),
        city=city,
    )


def search(
    query: str,
    limit: int = 5,
    country_codes: Optional[list[str]] = None,
    timeout: float = 15.0,
) -> list[SearchHit]:
    """Forward-geocode a free-text query (e.g. a business name) to candidate
    places. Optionally restrict to country codes to disambiguate. Rate-limited
    and UA-compliant like reverse(); returns [] on any failure."""
    if not query or not query.strip():
        return []
    _throttle()
    params = {
        "q": query,
        "format": "json",
        "addressdetails": 0,
        "limit": limit,
    }
    if country_codes:
        params["countrycodes"] = ",".join(c.lower() for c in country_codes)
    try:
        resp = requests.get(
            NOMINATIM_SEARCH_URL,
            params=params,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
            timeout=timeout,
        )
    except requests.RequestException:
        return []
    if resp.status_code != 200:
        return []
    try:
        rows = resp.json()
    except ValueError:
        return []

    hits: list[SearchHit] = []
    for row in rows if isinstance(rows, list) else []:
        try:
            hits.append(
                SearchHit(
                    display_name=row.get("display_name", query),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    category=row.get("class"),
                    importance=float(row.get("importance", 0.0) or 0.0),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return hits

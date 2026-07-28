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

"""GeoSeer — AI image geolocation API (a GeoSpy-style third locator).

GeoSeer (https://geoseeer.com) takes a photo and returns estimated coordinates
plus an address and a self-reported confidence. It's an *independent* locator
alongside GeoCLIP and StreetCLIP, and a strong one — so when its key is present
it participates directly in the location decision.

Gated on the ``GEOSEER_API_KEY`` environment variable. **Free tier is only ~10
requests/day**, so it's opt-in and best used on single important images, not
large batches — the response's remaining-quota counter is surfaced so you can
see how many calls you have left.

API: ``POST https://geoseeer.com/api/v1/analyze`` with header ``X-API-Key`` and a
multipart ``file`` upload.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import requests

API_URL = "https://geoseeer.com/api/v1/analyze"
API_KEY_ENV = "GEOSEER_API_KEY"


@dataclass
class GeoSeerLocation:
    lat: float
    lon: float
    address: Optional[str] = None
    confidence: float = 0.0
    reasoning: Optional[str] = None


@dataclass
class GeoSeerResult:
    available: bool
    locations: list[GeoSeerLocation] = field(default_factory=list)
    requests_remaining: Optional[int] = None
    reason: Optional[str] = None

    @property
    def top(self) -> Optional[GeoSeerLocation]:
        return self.locations[0] if self.locations else None


def api_key_configured(api_key: Optional[str] = None) -> bool:
    return bool(api_key or os.environ.get(API_KEY_ENV))


def predict(path: str, api_key: Optional[str] = None, mode: str = "fast",
            timeout: float = 90.0) -> GeoSeerResult:
    """Send an image to GeoSeer and parse the estimated location(s).

    ``api_key`` overrides the ``GEOSEER_API_KEY`` env var. Never raises on the
    normal 'not configured / quota / network' paths.
    """
    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        return GeoSeerResult(available=False,
                             reason=f"no {API_KEY_ENV} set (free ~10/day key enables GeoSeer)")

    try:
        with open(path, "rb") as fh:
            resp = requests.post(
                API_URL,
                headers={"X-API-Key": key},
                files={"file": fh},
                data={"analysis_mode": mode},
                timeout=timeout,
            )
    except (OSError, requests.RequestException) as exc:
        return GeoSeerResult(available=False, reason=f"GeoSeer request failed: {exc}")

    if resp.status_code == 402:
        return GeoSeerResult(available=False, reason="GeoSeer daily quota reached (402)")
    if resp.status_code == 401:
        return GeoSeerResult(available=False, reason="GeoSeer rejected the API key (401)")
    if resp.status_code != 200:
        return GeoSeerResult(available=False, reason=f"GeoSeer HTTP {resp.status_code}")

    try:
        data = resp.json()
    except ValueError:
        return GeoSeerResult(available=False, reason="GeoSeer returned invalid JSON")

    remaining = data.get("API_Requests_remaining")
    locations: list[GeoSeerLocation] = []
    for loc in data.get("locations", []) or []:
        try:
            locations.append(
                GeoSeerLocation(
                    lat=float(loc["latitude"]),
                    lon=float(loc["longitude"]),
                    address=loc.get("address"),
                    confidence=float(loc.get("confidence", 0.0) or 0.0),
                    reasoning=loc.get("reasoning"),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    if not locations:
        return GeoSeerResult(available=False, requests_remaining=remaining,
                             reason="GeoSeer returned no location")
    # Highest-confidence first.
    locations.sort(key=lambda l: l.confidence, reverse=True)
    return GeoSeerResult(available=True, locations=locations, requests_remaining=remaining)

"""Automated reverse image search via SerpAPI (Google Lens) — Phase 4.

For images pulled from the public web (reddit, Pinterest, blogs), the single
strongest OSINT signal is *finding the source*: the page it appears on usually
names the place. This runs that automatically and legitimately through SerpAPI's
Google Lens engine (no scraping, no CAPTCHAs), then extracts a location from the
match titles by geocoding them.

Requires a SerpAPI key (``SERPAPI_API_KEY``; free tier ~100 searches/month).

**Privacy note:** Google Lens needs a *public image URL*, and SerpAPI does not
accept file uploads — so the image is briefly uploaded to a public host
(litterbox, a temporary ~1 h host; catbox as fallback) to obtain a URL. Enable
this only for images you're comfortable sending to those services.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import requests

from . import __version__

SERPAPI_KEY_ENV = "SERPAPI_API_KEY"
SERPAPI_URL = "https://serpapi.com/search"
USER_AGENT = f"geolocator-osint/{__version__} (image geolocation research tool)"


@dataclass
class LensMatch:
    title: str
    link: Optional[str] = None
    source: Optional[str] = None


@dataclass
class ReverseSearchResult:
    available: bool
    matches: list[LensMatch] = field(default_factory=list)
    hosted_url: Optional[str] = None
    reason: Optional[str] = None


def api_key_configured(api_key: Optional[str] = None) -> bool:
    return bool(api_key or os.environ.get(SERPAPI_KEY_ENV))


# --------------------------------------------------------------------------- #
# Temporary public hosting (Lens needs a URL; SerpAPI takes no file upload)
# --------------------------------------------------------------------------- #
def _upload_litterbox(path: str, timeout: float = 60.0) -> Optional[str]:
    try:
        with open(path, "rb") as fh:
            r = requests.post(
                "https://litterbox.catbox.moe/resources/internals/api.php",
                data={"reqtype": "fileupload", "time": "1h"},
                files={"fileToUpload": fh},
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
            )
        if r.status_code == 200 and r.text.startswith("http"):
            return r.text.strip()
    except (OSError, requests.RequestException):
        pass
    return None


def _upload_catbox(path: str, timeout: float = 60.0) -> Optional[str]:
    try:
        with open(path, "rb") as fh:
            r = requests.post(
                "https://catbox.moe/user/api.php",
                data={"reqtype": "fileupload"},
                files={"fileToUpload": fh},
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
            )
        if r.status_code == 200 and r.text.startswith("http"):
            return r.text.strip()
    except (OSError, requests.RequestException):
        pass
    return None


def _host_image(path: str) -> Optional[str]:
    """Upload to a temporary host first, permanent as fallback. Returns a URL."""
    return _upload_litterbox(path) or _upload_catbox(path)


# --------------------------------------------------------------------------- #
# SerpAPI Google Lens
# --------------------------------------------------------------------------- #
def lens_search(image_url: str, api_key: str, timeout: float = 90.0) -> ReverseSearchResult:
    try:
        resp = requests.get(
            SERPAPI_URL,
            params={"engine": "google_lens", "url": image_url,
                    "api_key": api_key, "hl": "en"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return ReverseSearchResult(False, reason=f"SerpAPI request failed: {exc}")
    if resp.status_code != 200:
        return ReverseSearchResult(False, reason=f"SerpAPI HTTP {resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        return ReverseSearchResult(False, reason="SerpAPI returned invalid JSON")
    if data.get("error"):
        return ReverseSearchResult(False, reason=f"SerpAPI: {data['error']}")

    matches: list[LensMatch] = []
    for m in data.get("visual_matches", []) or []:
        title = (m.get("title") or "").strip()
        if title:
            matches.append(LensMatch(title=title, link=m.get("link"),
                                     source=m.get("source")))
    return ReverseSearchResult(available=True, matches=matches, hosted_url=image_url)


def search(path: str, api_key: Optional[str] = None) -> ReverseSearchResult:
    """Host the image, run Google Lens, return the visual matches."""
    key = api_key or os.environ.get(SERPAPI_KEY_ENV)
    if not key:
        return ReverseSearchResult(False, reason=f"no {SERPAPI_KEY_ENV} set")
    url = _host_image(path)
    if not url:
        return ReverseSearchResult(False, reason="could not upload image to a temporary host")
    return lens_search(url, key)

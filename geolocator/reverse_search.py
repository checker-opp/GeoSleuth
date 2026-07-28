"""Reverse-image-search pivots — Phase 4 (best-effort, manual).

Reverse image search is the single most useful OSINT pivot for a photo, but
doing it *automatically* means either scraping engines that actively fight bots
(Yandex, Google Lens) or paying for an API (TinEye commercial). This tool does
**not** scrape or attempt to defeat CAPTCHAs — that violates those services'
terms and gets you IP-banned regardless.

Instead we do the honest, reliable thing: emit ready-to-open **pivot links** to
each engine's upload page, so the investigator runs the search manually in a
browser. Yandex is listed first — it's widely regarded as the strongest engine
for location matching on landscapes and street scenes.

(Automated TinEye/Bing lookups could be added later behind an API key; see the
README. They're intentionally omitted here rather than shipped untested.)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Pivot:
    engine: str
    url: str
    note: str


# Ordered by usefulness for geolocation (Yandex first, per OSINT practice).
def build_pivots() -> list[Pivot]:
    return [
        Pivot(
            engine="Yandex Images",
            url="https://yandex.com/images/",
            note="best for landscapes/streets — click the camera icon, upload the photo",
        ),
        Pivot(
            engine="Google Lens",
            url="https://lens.google.com/",
            note="strong for landmarks and products — upload the photo",
        ),
        Pivot(
            engine="Bing Visual Search",
            url="https://www.bing.com/visualsearch",
            note="upload the photo via the camera icon",
        ),
        Pivot(
            engine="TinEye",
            url="https://tineye.com/",
            note="best for finding exact-copy reposts (where else the image appears)",
        ),
    ]

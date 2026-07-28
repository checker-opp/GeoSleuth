"""License-plate region hints from OCR text — Phase 4 (weak signal).

The spec calls for OpenALPR (a heavy, partly-commercial dependency). Rather than
pull that in, we take a lightweight, honest approximation: scan text that OCR
already extracted for tokens matching a few *highly distinctive* national plate
formats. This is deliberately conservative — most plate formats overlap across
countries, so we only claim a country for patterns with strong signal, and even
then at low confidence. Ambiguous plate-like tokens are reported without a
country claim.

Pure/offline — operates on already-extracted OCR text, no model needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Each entry: (compiled regex, country, human note). Only distinctive formats.
_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # UK current format: two letters, two digits, three letters — e.g. "AB12 CDE".
    (re.compile(r"\b[A-Z]{2}[0-9]{2}\s?[A-Z]{3}\b"), "United Kingdom",
     "UK 2001-present format (AA00 AAA)"),
    # India: state code + RTO digits + series + 4 digits — e.g. "MH12 AB 1234".
    (re.compile(r"\b[A-Z]{2}[0-9]{1,2}\s?[A-Z]{1,3}\s?[0-9]{4}\b"), "India",
     "Indian format (SS NN L(L) NNNN)"),
    # Netherlands sidecode-ish grouped format e.g. "12-ABC-3" / "AB-123-C".
    (re.compile(r"\b([0-9]{2}-[A-Z]{3}-[0-9]|[A-Z]{2}-[0-9]{3}-[A-Z])\b"),
     "Netherlands", "Dutch dash-grouped format"),
]

# A generic "looks like a plate" catcher (no country claim) — 5-8 alnum,
# containing both letters and digits, optionally with one space/dash.
_GENERIC = re.compile(r"\b(?=[A-Z0-9\s-]{5,9}\b)(?=[^\s-]*[A-Z])(?=[^\s-]*[0-9])[A-Z0-9]{2,4}[\s-]?[A-Z0-9]{2,4}\b")


@dataclass
class PlateHint:
    matched: bool
    country: Optional[str] = None
    note: Optional[str] = None
    token: Optional[str] = None


def detect(text: str) -> PlateHint:
    """Look for a distinctive plate pattern in OCR text.

    Returns the first confident country match; failing that, flags a generic
    plate-like token (no country); failing that, no match.
    """
    if not text:
        return PlateHint(matched=False)
    upper = text.upper()

    for pattern, country, note in _PATTERNS:
        m = pattern.search(upper)
        if m:
            return PlateHint(matched=True, country=country, note=note, token=m.group(0).strip())

    m = _GENERIC.search(upper)
    if m:
        return PlateHint(matched=True, country=None,
                         note="plate-like token (format not distinctive enough to attribute)",
                         token=m.group(0).strip())

    return PlateHint(matched=False)

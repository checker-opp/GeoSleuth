"""iNaturalist biodiversity cross-reference — Phase 3 (enrichment).

The spec's idea is "flora → climate zone → region". Going *image → species*
needs a plant/animal vision model we don't ship, so we do the honest, free half:
given a **candidate coordinate**, ask iNaturalist which species are actually
observed nearby. That corroborates the biome (a snowy-conifer guess shouldn't
sit in a tropical species list) and adds recognizable ecological context.

Keyless and free — the public iNaturalist API needs no auth. Enrichment only:
it describes/sanity-checks a candidate, it doesn't locate on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import requests

from . import __version__
from .models import Coordinates

SPECIES_COUNTS_URL = "https://api.inaturalist.org/v1/observations/species_counts"
USER_AGENT = f"geolocator-osint/{__version__} (image geolocation research tool)"


@dataclass
class Taxon:
    name: str                         # scientific name
    common_name: Optional[str] = None
    group: Optional[str] = None        # iconic taxon, e.g. "Plantae", "Aves"
    observations: int = 0


@dataclass
class InatResult:
    available: bool
    taxa: list[Taxon] = field(default_factory=list)
    reason: Optional[str] = None


def nearby_taxa(
    coord: Coordinates,
    radius_km: int = 50,
    per_page: int = 8,
    timeout: float = 20.0,
) -> InatResult:
    """Most-observed species near a coordinate (research-grade). [] on failure."""
    try:
        resp = requests.get(
            SPECIES_COUNTS_URL,
            params={
                "lat": coord.lat,
                "lng": coord.lon,
                "radius": radius_km,
                "quality_grade": "research",
                "per_page": per_page,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return InatResult(available=False, reason=f"iNaturalist request failed: {exc}")
    if resp.status_code != 200:
        return InatResult(available=False, reason=f"iNaturalist HTTP {resp.status_code}")
    try:
        results = resp.json().get("results", [])
    except ValueError:
        return InatResult(available=False, reason="iNaturalist returned invalid JSON")

    taxa: list[Taxon] = []
    for row in results:
        taxon = row.get("taxon") or {}
        name = taxon.get("name")
        if not name:
            continue
        taxa.append(
            Taxon(
                name=name,
                common_name=taxon.get("preferred_common_name"),
                group=taxon.get("iconic_taxon_name"),
                observations=row.get("count", 0),
            )
        )
    return InatResult(available=True, taxa=taxa)

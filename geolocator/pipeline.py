"""Pipeline orchestration + confidence aggregation.

Runs the location signals in the spec's priority order (cheap/deterministic
first, fuzzy/expensive later) and folds them into a single ``GeoResult``:

    EXIF  ->  [reverse image search]  ->  OCR + plates  ->  landmark model
          ->  street match  ->  shadow / flora

Phase 1 implements the EXIF and OCR stages. The later stages are declared as
placeholders so the ordering and result-merging logic is already in place —
adding a stage means writing one function and slotting it into ``STAGES``.
"""

from __future__ import annotations

import os

from . import climate as climate_mod
from . import exif as exif_mod
from . import geocode as geocode_mod
from . import geoestimate as geoestimate_mod
from . import geoseer as geoseer_mod
from . import inaturalist as inaturalist_mod
from . import ocr as ocr_mod
from . import osm as osm_mod
from . import plates as plates_mod
from . import reverse_search as reverse_search_mod
from . import reverse_search_api as reverse_search_api_mod
from . import secondmodel as secondmodel_mod
from . import solar as solar_mod
from . import street_match as street_match_mod
from .models import (
    AnalyzeConfig,
    Coordinates,
    GeoResult,
    Precision,
    Signal,
    precision_rank,
)

# Confidence assigned to a GPS fix. Not 1.0: EXIF GPS can be spoofed, stale,
# or reflect where a photo was edited rather than shot — high, not certain.
EXIF_GPS_CONFIDENCE = 0.95

# A GeoSeer result at or above this confidence short-circuits the local models.
GEOSEER_SHORT_CIRCUIT_MIN = 0.5


# --------------------------------------------------------------------------- #
# Stage: EXIF
# --------------------------------------------------------------------------- #
def stage_exif(result: GeoResult) -> None:
    data = exif_mod.extract(result.image_path)

    # Stash timestamp / timezone for the later solar-corroboration stage.
    if data.datetime:
        result.meta["exif_datetime"] = data.datetime
    if data.timezone_offset:
        result.meta["timezone_offset"] = data.timezone_offset

    if data.has_gps:
        coords = data.coordinates
        place_name = None
        precision = Precision.EXACT
        place = geocode_mod.reverse(coords)
        if place:
            place_name = place.display_name
        result.add(
            Signal(
                source="exif",
                description=(
                    f"GPS coordinates embedded in image metadata"
                    + (f" → {place_name}" if place_name else "")
                ),
                confidence=EXIF_GPS_CONFIDENCE,
                precision=precision,
                coordinates=coords,
                place=place_name,
                evidence={
                    "extractor": data.source,
                    "camera": _camera_str(data),
                    "datetime": data.datetime,
                    "reverse_geocoded": bool(place),
                },
            )
        )
        return

    # No GPS — still surface metadata clues as low-value context.
    clues: dict[str, str] = {}
    if data.camera_make or data.camera_model:
        clues["camera"] = _camera_str(data)
    if data.datetime:
        clues["datetime"] = data.datetime
    if data.timezone_offset:
        clues["timezone_offset"] = data.timezone_offset

    if data.timezone_offset:
        # UTC offset is a (very) coarse longitude band — worth noting only.
        result.add(
            Signal(
                source="exif",
                description=f"No GPS, but timezone offset {data.timezone_offset} "
                f"hints at a longitude band",
                confidence=0.10,
                precision=Precision.UNKNOWN,
                evidence=clues,
            )
        )
    if clues and not data.timezone_offset:
        result.note(
            "EXIF present but no GPS. Metadata clues: "
            + ", ".join(f"{k}={v}" for k, v in clues.items())
        )
    elif data.source == "none" or not data.raw:
        result.note("No EXIF metadata found (likely stripped, e.g. via social media).")


def _camera_str(data: exif_mod.ExifData) -> str:
    parts = [p for p in (data.camera_make, data.camera_model) if p]
    return " ".join(parts) if parts else "unknown"


# --------------------------------------------------------------------------- #
# Stage: OCR + language hint
# --------------------------------------------------------------------------- #
def stage_ocr(result: GeoResult) -> None:
    ocr = ocr_mod.run(result.image_path)

    if not ocr.available:
        result.note(f"OCR skipped: {ocr.reason}")
        return
    if not ocr.has_text:
        result.note("OCR ran but found no readable text in the image.")
        return

    # Stash candidate name lines for the later place-name lookup stage.
    if ocr.lines:
        result.meta["ocr_lines"] = ocr.lines

    snippet = _snippet(ocr.text)

    if ocr.language and ocr.language_countries:
        result.add(
            Signal(
                source="ocr",
                description=(
                    f"Detected {ocr.language!r} text → likely one of: "
                    + ", ".join(ocr.language_countries)
                ),
                confidence=0.30,
                precision=Precision.COUNTRY,
                corroborating=True,  # independent locating hint
                place=None,
                evidence={
                    "language": ocr.language,
                    "candidate_countries": ocr.language_countries,
                    "text_snippet": snippet,
                },
            )
        )
    else:
        result.add(
            Signal(
                source="ocr",
                description="Readable text found (no confident language match) — "
                "manual review may pinpoint a business or place name",
                confidence=0.15,
                precision=Precision.UNKNOWN,
                evidence={"text_snippet": snippet},
            )
        )

    # License-plate region hint (Phase 4) — reuses the text OCR already found.
    hint = plates_mod.detect(ocr.text)
    if hint.matched and hint.country:
        result.add(
            Signal(
                source="plate",
                description=f"License-plate pattern suggests {hint.country} "
                f"({hint.note})",
                confidence=0.25,
                precision=Precision.COUNTRY,
                corroborating=True,  # independent locating hint
                evidence={"token": hint.token, "note": hint.note,
                          "candidate_countries": [hint.country]},
            )
        )
    elif hint.matched:
        result.note(
            f"Possible license-plate token detected ('{hint.token}') but format "
            f"is not distinctive enough to attribute a country."
        )


def _snippet(text: str, limit: int = 240) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


# --------------------------------------------------------------------------- #
# Placeholder stages (later phases) — declared so ordering is explicit.
# --------------------------------------------------------------------------- #
import re as _re


def _reverse_extract(matches) -> Optional[tuple]:
    """From Lens match titles, geocode candidates and return the best-agreed
    location as (Coordinates, place, agree_count), or None."""
    def clean(t: str) -> str:
        t = _re.sub(r"^File:", "", t)
        t = _re.sub(r"\.(jpg|jpeg|png|webp).*$", "", t, flags=_re.I)
        t = _re.sub(r"\s*[-|–]\s*\S.{0,25}$", "", t)  # trailing " - Source"
        return t.strip()

    seen: set = set()
    hits = []
    for m in matches[:12]:
        c = clean(m.title)
        if len(c) < 4 or c.lower() in seen:
            continue
        seen.add(c.lower())
        found = geocode_mod.search(c, limit=1)
        if found:
            hits.append(found[0])
        if len(hits) >= 6:
            break
    if not hits:
        return None
    # Pick the hit with the most neighbours within ~75 km (agreement), tie-broken
    # by Nominatim importance.
    best, best_agree, best_imp = None, -1, -1.0
    for h in hits:
        agree = sum(
            1 for o in hits
            if geoestimate_mod._haversine_km(Coordinates(h.lat, h.lon),
                                             Coordinates(o.lat, o.lon)) <= 75
        )
        if (agree, h.importance) > (best_agree, best_imp):
            best, best_agree, best_imp = h, agree, h.importance
    return (Coordinates(best.lat, best.lon), best.display_name, best_agree)


def stage_reverse_image_search(result: GeoResult) -> None:  # Phase 4
    """Automated reverse image search (SerpAPI Google Lens) + manual pivot links.

    For web-sourced photos this is often the single strongest signal — it finds
    the page the image appears on, whose title usually names the place. Needs a
    SerpAPI key; without one it just attaches the manual pivot links.
    """
    # Manual pivot links are always available (no key, no scraping).
    result.meta["pivots"] = [
        {"engine": p.engine, "url": p.url, "note": p.note}
        for p in reverse_search_mod.build_pivots()
    ]

    key = result.meta.get("serpapi_key")
    cfg = result.meta.get("config")
    if not key or (cfg is not None and not getattr(cfg, "use_reverse_search", True)):
        return
    if any(s.precision == Precision.EXACT for s in result.signals):
        return  # already have an exact GPS fix

    res = reverse_search_api_mod.search(result.image_path, api_key=key)
    if not res.available:
        result.note(f"Reverse image search skipped: {res.reason}")
        return
    if not res.matches:
        result.note("Reverse image search ran but found no visual matches.")
        return

    extracted = _reverse_extract(res.matches)
    top_titles = [m.title for m in res.matches[:4]]
    if extracted is None:
        result.note("Reverse image search found matches but couldn't extract a "
                    "location — inspect them: " + " | ".join(top_titles))
        return

    coords, place, agree = extracted
    # Finding the source is strong evidence; more agreeing matches = more trust.
    confidence = min(0.78, 0.45 + 0.1 * agree)
    result.add(
        Signal(
            source="reverse_image",
            description=f"Reverse image search matched '{place}' "
            f"({agree} agreeing web match(es))",
            confidence=confidence,
            precision=Precision.CITY,
            coordinates=coords,
            place=place,
            corroborating=True,
            evidence={"matches": top_titles, "agree": agree},
        )
    )
    # A solid reverse-search hit is definitive — skip the slow local models.
    short = getattr(cfg, "short_circuit_on_geoseer", True) if cfg else True
    if short and agree >= 2:
        result.meta["skip_heavy_models"] = True
        result.note("Reverse image search located the source — skipping GeoSeer "
                    "and the local models.")


def stage_landmark_model(result: GeoResult) -> None:  # Phase 2 (GeoCLIP)
    """ML coordinate estimation for images with no usable metadata.

    Skipped when we already have an exact GPS fix (no point running an expensive
    model to second-guess real coordinates), when GeoSeer already produced a
    confident fix (short-circuit), or when the ML extras aren't installed.
    """
    if result.meta.get("skip_heavy_models"):
        return  # GeoSeer already located it confidently
    # If an exact fix already exists, ML estimation adds nothing — skip it.
    if any(s.precision == Precision.EXACT for s in result.signals):
        return

    est = geoestimate_mod.predict(result.image_path, top_k=5)
    if not est.available:
        result.note(f"ML geo-estimation skipped: {est.reason}")
        return
    if est.top is None:
        result.note("ML geo-estimation ran but returned no prediction.")
        return

    precision, confidence, spread = geoestimate_mod.assess(est)
    coords = est.top.coordinates

    place_name = None
    place = geocode_mod.reverse(coords)
    if place:
        place_name = place.display_name

    spread_desc = (
        f"top-5 predictions within ~{spread:.0f} km"
        if spread != float("inf")
        else "single prediction"
    )
    result.add(
        Signal(
            source="ml_geoclip",
            description=(
                f"GeoCLIP visual estimate ({spread_desc})"
                + (f" → {place_name}" if place_name else "")
            ),
            confidence=confidence,
            precision=precision,
            coordinates=coords,
            place=place_name,
            corroborating=True,  # independent locator; boosts an agreeing winner
            evidence={
                "model": "GeoCLIP",
                "top_probability": round(est.top.probability, 4),
                "spread_km": round(spread, 1) if spread != float("inf") else None,
                "top_k": [
                    {"lat": p.coordinates.lat, "lon": p.coordinates.lon,
                     "prob": round(p.probability, 4)}
                    for p in est.predictions
                ],
            },
        )
    )


def stage_geoseer(result: GeoResult) -> None:  # AI geolocation API (3rd locator)
    """GeoSeer AI geolocation — a strong independent locator when its key is set.

    Skipped without GEOSEER_API_KEY, and skipped when an exact GPS fix already
    exists (no point spending a scarce ~10/day API call). Emits a locating
    signal with the API's coordinates + confidence; it competes in the normal
    aggregation, so a confident GeoSeer result naturally wins over a weaker
    GeoCLIP guess and corroborates an agreeing one.
    """
    if result.meta.get("skip_heavy_models"):
        return  # reverse image search already located it
    key = result.meta.get("geoseer_key")
    if not key:
        return
    if any(s.precision == Precision.EXACT for s in result.signals):
        return

    res = geoseer_mod.predict(result.image_path, api_key=key)
    if res.requests_remaining is not None:
        result.meta["geoseer_remaining"] = res.requests_remaining
    if not res.available or res.top is None:
        result.note(f"GeoSeer skipped: {res.reason}")
        return

    loc = res.top
    coords = Coordinates(loc.lat, loc.lon)
    # Prefer a normalized Nominatim place; fall back to the API's address.
    place = geocode_mod.reverse(coords)
    place_name = place.display_name if place else loc.address

    api_conf = loc.confidence
    if api_conf >= 0.6:
        precision = Precision.CITY
    elif api_conf >= 0.35:
        precision = Precision.REGION
    else:
        precision = Precision.COUNTRY

    result.add(
        Signal(
            source="geoseer",
            description=(
                f"GeoSeer AI estimate (p={api_conf:.2f})"
                + (f" → {place_name}" if place_name else "")
            ),
            confidence=min(0.80, api_conf),  # strong, but never rivals a GPS fix
            precision=precision,
            coordinates=coords,
            place=place_name,
            corroborating=True,
            evidence={
                "api_confidence": api_conf,
                "api_address": loc.address,
                "reasoning": (loc.reasoning or "")[:280] or None,
                "requests_remaining": res.requests_remaining,
            },
        )
    )
    if res.requests_remaining is not None:
        result.note(f"GeoSeer: {res.requests_remaining} API request(s) remaining today.")

    # Short-circuit: a confident GeoSeer fix makes the slow local models redundant.
    cfg = result.meta.get("config")
    short = getattr(cfg, "short_circuit_on_geoseer", True) if cfg else True
    if short and api_conf >= GEOSEER_SHORT_CIRCUIT_MIN:
        result.meta["skip_heavy_models"] = True
        result.note("GeoSeer produced a confident fix — skipping the local "
                    "GeoCLIP/StreetCLIP models.")


# Opt-in: StreetCLIP is a heavy second model (~1.6 GB, slow on CPU). Off unless
# GEOLOCATOR_STREETCLIP is set, so normal runs stay fast.
STREETCLIP_ENV = "GEOLOCATOR_STREETCLIP"


def streetclip_enabled() -> bool:
    return bool(os.environ.get(STREETCLIP_ENV))


def stage_second_model(result: GeoResult) -> None:  # optional StreetCLIP cross-check
    """Run the StreetCLIP second opinion and stash its country verdict.

    Opt-in (GEOLOCATOR_STREETCLIP=1) because the model is large. This stage only
    *records* the verdict (as evidence + meta); the ``resolve_models`` stage
    later decides how it interacts with the GeoCLIP guess (boost / re-rank /
    override / down-weight).
    """
    if result.meta.get("skip_heavy_models"):
        return  # GeoSeer already located it confidently
    best = max(
        (s for s in result.signals if s.locating and s.place),
        key=lambda s: (precision_rank(s.precision), s.confidence),
        default=None,
    )
    geoclip_country = _country_of(best.place) if best and best.place else None

    pred = secondmodel_mod.predict_country(
        result.image_path,
        extra_countries=[geoclip_country] if geoclip_country else None,
    )
    if pred is None:
        result.note("StreetCLIP second opinion unavailable (ML extras/weights).")
        return

    result.meta["streetclip"] = {"country": pred.country, "score": round(pred.score, 3)}
    result.add(
        Signal(
            source="streetclip",
            description=f"StreetCLIP country estimate: {pred.country} (p={pred.score:.2f})",
            confidence=round(pred.score, 3),
            precision=Precision.COUNTRY,
            locating=False,          # evidence unless resolve_models promotes it
            corroborating=False,     # resolve_models flips this on agreement
            evidence={"country": pred.country, "score": round(pred.score, 3),
                      "candidate_countries": [pred.country]},
        )
    )


def _country_of(place: str) -> Optional[str]:
    """Last comma-separated component of a Nominatim display name is the country."""
    if not place:
        return None
    parts = [p.strip() for p in place.split(",") if p.strip()]
    return parts[-1] if parts else None


# StreetCLIP must be at least this confident to override GeoCLIP on disagreement.
_OVERRIDE_MIN_SCORE = 0.40


def stage_resolve_models(result: GeoResult) -> None:
    """Reconcile GeoCLIP and the StreetCLIP second opinion.

    On **agreement** the StreetCLIP signal becomes corroborating (boosts
    confidence). On **disagreement** — where GeoCLIP has been observed to be
    *confidently wrong* (e.g. a glass CBD → the wrong continent) — we, in order:

      1. **Re-rank:** if any of GeoCLIP's own top-5 predictions is in the country
         StreetCLIP votes for, promote that one (keeps coordinate precision).
      2. **Country-override:** otherwise report StreetCLIP's country at
         country-level, demoting GeoCLIP's contradicted coordinates to evidence.
      3. **Down-weight:** always cut confidence when the two models disagree.

    Never overrides an exact EXIF GPS fix.
    """
    sc = result.meta.get("streetclip")
    if not sc:
        return
    sc_country, sc_score = sc["country"], sc["score"]

    ml = max(
        (s for s in result.signals if s.locating and s.place),
        key=lambda s: (precision_rank(s.precision), s.confidence),
        default=None,
    )
    sc_signal = next((s for s in result.signals if s.source == "streetclip"), None)

    # No visual guess to compare against, or an exact GPS fix we trust — stand down.
    if ml is None:
        return
    if ml.precision == Precision.EXACT:
        return

    ml_country = _country_of(ml.place)
    agrees = ml_country and sc_country.lower() in ml_country.lower()

    if agrees:
        if sc_signal is not None:
            sc_signal.corroborating = True
            sc_signal.description = (
                f"StreetCLIP independently agrees on {sc_country} "
                f"(p={sc_score:.2f}) — cross-model corroboration"
            )
        return

    # --- Disagreement ---
    result.note(
        f"Model disagreement: GeoCLIP→{ml_country}, StreetCLIP→{sc_country} "
        f"(p={sc_score:.2f})."
    )
    result.meta["disagreement_penalty"] = True

    if sc_score < _OVERRIDE_MIN_SCORE:
        return  # too weak to override; the down-weight in _aggregate is enough

    # Only correct GeoCLIP — it's the locator prone to confident continental
    # errors. Stronger locators (GeoSeer, an OCR place match) are trusted; a
    # StreetCLIP disagreement with them only down-weights, never overrides.
    if ml.source != "ml_geoclip":
        return

    # 1) Re-rank: look for a top-5 GeoCLIP prediction in StreetCLIP's country.
    topk = (ml.evidence or {}).get("top_k", [])
    for p in topk[1:]:  # skip #1 (that's the contradicted ml pick)
        pl = geocode_mod.reverse(Coordinates(p["lat"], p["lon"]))
        if pl and pl.country and sc_country.lower() in pl.country.lower():
            ml.locating = False       # demote the contradicted top pick
            ml.corroborating = False  # ...and it must not corroborate its replacement
            result.add(
                Signal(
                    source="resolved",
                    description=f"Re-ranked to a StreetCLIP-consistent GeoCLIP "
                    f"prediction → {pl.display_name}",
                    confidence=0.40,
                    precision=Precision.REGION,
                    coordinates=Coordinates(p["lat"], p["lon"]),
                    place=pl.display_name,
                    locating=True,
                    evidence={"reason": "rerank", "agreed_country": sc_country},
                )
            )
            result.note(
                "Re-ranked: GeoCLIP's top pick contradicted StreetCLIP, so a "
                "lower-ranked GeoCLIP prediction in the agreed country was used."
            )
            return

    # 2) Country-override: no GeoCLIP prediction matches → trust StreetCLIP's country.
    ml.locating = False
    ml.corroborating = False
    result.add(
        Signal(
            source="resolved",
            description=f"Models disagree and no GeoCLIP prediction matches; "
            f"reporting StreetCLIP's country: {sc_country}",
            confidence=0.30,
            precision=Precision.COUNTRY,
            coordinates=None,
            place=sc_country,
            locating=True,
            evidence={"reason": "country_override", "geoclip_country": ml_country,
                      "streetclip_country": sc_country, "streetclip_score": sc_score},
        )
    )
    result.note(
        f"GeoCLIP's predictions all contradict StreetCLIP; falling back to "
        f"StreetCLIP's country ({sc_country}) at country-level."
    )


def stage_place_lookup(result: GeoResult) -> None:  # OCR text -> Nominatim search
    """Turn OCR'd business / place names into a location.

    Searches Nominatim for each candidate name and, when a hit lands near the
    visual (GeoCLIP) candidate, emits a strong *locating* signal — a named
    business that both OCR reads and the model points at is powerful evidence.
    Requiring geographic agreement with the ML candidate is what keeps noisy
    OCR tokens from producing false matches.
    """
    names = result.meta.get("ocr_lines") or []
    if not names:
        return
    anchor = _current_best_coord(result)  # GeoCLIP/EXIF candidate, if any
    if anchor is None:
        # Without a visual anchor an OCR name is geographically ambiguous
        # (a shop name exists in many cities) — skip rather than guess wrong.
        result.note(
            f"OCR found {len(names)} candidate name(s) but no visual anchor to "
            f"disambiguate them; use the reverse-image-search pivots."
        )
        return

    # Query the most distinctive candidates (longest = most specific), capped to
    # respect Nominatim's rate limit.
    ranked = sorted(names, key=len, reverse=True)[:4]
    best_hit = None
    best_dist = float("inf")
    best_name = None
    for name in ranked:
        for hit in geocode_mod.search(name, limit=5):
            dist = geoestimate_mod._haversine_km(
                anchor, Coordinates(hit.lat, hit.lon)
            )
            if dist < best_dist:
                best_dist, best_hit, best_name = dist, hit, name

    # Accept only if a hit is in the same metro area as the visual estimate.
    CONSISTENT_KM = 30.0
    if best_hit is None or best_dist > CONSISTENT_KM:
        if best_hit is not None:
            result.note(
                f"OCR name search found candidates but none near the visual "
                f"estimate (closest ~{best_dist:.0f} km) — not used."
            )
        return

    coords = Coordinates(best_hit.lat, best_hit.lon)
    result.add(
        Signal(
            source="ocr_place",
            description=(
                f"OCR read '{best_name}' → matches '{best_hit.display_name}' "
                f"~{best_dist:.0f} km from the visual estimate"
            ),
            confidence=0.55,
            precision=Precision.CITY,
            coordinates=coords,
            place=best_hit.display_name,
            corroborating=True,  # independent locator that agrees with the model
            evidence={
                "ocr_name": best_name,
                "matched": best_hit.display_name,
                "distance_km_from_visual": round(best_dist, 1),
                "category": best_hit.category,
            },
        )
    )


def stage_street_match(result: GeoResult) -> None:  # Phase 4 (Mapillary / KartaView)
    """Corroborate the candidate with nearby street-level imagery coverage.

    Enrichment only (corroborating=False): imagery existing near a coordinate
    doesn't prove *this* photo was taken there — a human must visually compare.
    """
    coord = _current_best_coord(result)
    if coord is None:
        return

    res = street_match_mod.find_street_imagery(
        coord, token=result.meta.get("mapillary_token")
    )
    if not res.available or not res.images:
        result.note(f"Street matching: no imagery near the candidate ({res.reason}).")
        return

    provider = res.images[0].provider
    result.add(
        Signal(
            source="street_match",
            description=(
                f"{len(res.images)} {provider} street-level image(s) near the "
                f"candidate — open to visually confirm the match"
            ),
            confidence=0.20,
            precision=Precision.CITY,
            coordinates=coord,
            locating=False,       # enrichment: echoes the candidate
            corroborating=False,  # imagery existing != this photo taken here
            evidence={
                "provider": provider,
                "count": len(res.images),
                "viewer_urls": [img.viewer_url for img in res.images],
            },
        )
    )


# --------------------------------------------------------------------------- #
# Corroboration helpers (Phase 3 stages operate on the best candidate so far)
# --------------------------------------------------------------------------- #
def _current_best_coord(result: GeoResult) -> Optional[Coordinates]:
    """Coordinates of the most precise *locating* signal so far (enrichment
    signals only echo a candidate, so they're excluded)."""
    coord_signals = [
        s for s in result.signals if s.coordinates is not None and s.locating
    ]
    if not coord_signals:
        return None
    best = max(coord_signals, key=lambda s: (precision_rank(s.precision), s.confidence))
    return best.coordinates


def stage_osm_crossref(result: GeoResult) -> None:  # Phase 3 (Overpass)
    """Enrich the candidate with nearby named OSM features (corroboration)."""
    coord = _current_best_coord(result)
    if coord is None:
        return  # nothing to corroborate

    res = osm_mod.nearby(coord)
    if not res.available:
        result.note(f"OSM cross-reference skipped: {res.reason}")
        return
    if not res.features:
        result.note("OSM cross-reference: no named features near the candidate.")
        return

    names = [f"{f.name} ({f.kind})" for f in res.features]
    result.add(
        Signal(
            source="osm",
            description="Nearby OSM features corroborate a real place: "
            + ", ".join(f.name for f in res.features[:5]),
            confidence=0.20,
            precision=Precision.CITY,
            coordinates=coord,
            locating=False,   # enrichment: echoes the candidate, not its own fix
            evidence={"features": names, "radius_m": 300},
        )
    )


def stage_inaturalist(result: GeoResult) -> None:  # Phase 3 (biodiversity enrichment)
    """Corroborate the candidate's biome with species actually observed nearby.

    Enrichment only (locating=False): it sanity-checks / describes a candidate
    coordinate with real biodiversity data, it doesn't locate on its own.
    """
    coord = _current_best_coord(result)
    if coord is None:
        return

    res = inaturalist_mod.nearby_taxa(coord)
    if not res.available:
        result.note(f"iNaturalist cross-reference skipped: {res.reason}")
        return
    if not res.taxa:
        result.note("iNaturalist: no research-grade observations near the candidate.")
        return

    def label(t):
        return t.common_name or t.name

    names = [label(t) for t in res.taxa[:5]]
    result.add(
        Signal(
            source="inaturalist",
            description="Species observed nearby corroborate the biome: "
            + ", ".join(names),
            confidence=0.08,
            precision=Precision.UNKNOWN,
            coordinates=coord,
            locating=False,
            evidence={
                "taxa": [
                    {"name": t.name, "common": t.common_name, "group": t.group,
                     "observations": t.observations}
                    for t in res.taxa
                ]
            },
        )
    )


def stage_shadow_flora(result: GeoResult) -> None:  # Phase 3 (solar + climate)
    """Solar consistency + climate-zone descriptor for the candidate."""
    coord = _current_best_coord(result)
    if coord is None:
        return

    # --- Climate / latitude-zone descriptor (always available) ---
    climate = climate_mod.describe(coord)

    # --- Solar corroboration (uses EXIF time / timezone if present) ---
    info = solar_mod.analyze(
        coord,
        exif_datetime=result.meta.get("exif_datetime"),
        utc_offset=result.meta.get("timezone_offset"),
    )

    parts: list[str] = [f"climate zone: {climate['latitude_zone']}"]
    evidence: dict = {"climate": climate}
    confidence = 0.08  # enrichment-level; corroborates, doesn't locate

    if info.longitude_consistent is not None:
        evidence["timezone_longitude_check"] = {
            "band_longitude": info.longitude_from_offset,
            "candidate_longitude": coord.lon,
            "delta_deg": info.longitude_delta_deg,
            "consistent": info.longitude_consistent,
        }
        if info.longitude_consistent:
            parts.append(
                f"timezone offset agrees with candidate longitude "
                f"(Δ{info.longitude_delta_deg}°)"
            )
            confidence = 0.18  # a real independent cross-check that passed
        else:
            parts.append(
                f"⚠ timezone offset disagrees with candidate longitude "
                f"(Δ{info.longitude_delta_deg}°)"
            )
            result.note(
                "Solar check: EXIF timezone offset is inconsistent with the "
                "candidate longitude — the location guess may be wrong."
            )

    if info.sun_elevation is not None:
        state = "daytime" if info.is_daytime else "nighttime"
        parts.append(f"sun {info.sun_elevation}° ({state}) at capture time")
        evidence["sun"] = {
            "elevation_deg": info.sun_elevation,
            "azimuth_deg": info.sun_azimuth,
            "is_daytime": info.is_daytime,
        }

    result.add(
        Signal(
            source="solar_climate",
            description="; ".join(parts),
            confidence=confidence,
            # Enrichment/consistency — never overrides the actual location fix.
            precision=Precision.UNKNOWN,
            locating=False,
            evidence=evidence,
        )
    )


# Ordered exactly per the spec. Only implemented stages do work today.
STAGES = [
    ("exif", stage_exif),
    ("reverse_image_search", stage_reverse_image_search),
    ("ocr", stage_ocr),
    ("geoseer", stage_geoseer),          # runs first so it can short-circuit GeoCLIP
    ("landmark_model", stage_landmark_model),
    ("second_model", stage_second_model),
    ("place_lookup", stage_place_lookup),
    ("resolve_models", stage_resolve_models),
    ("street_match", stage_street_match),
    ("osm_crossref", stage_osm_crossref),
    ("inaturalist", stage_inaturalist),
    ("shadow_flora", stage_shadow_flora),
]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _aggregate(result: GeoResult) -> None:
    """Fold all signals into one best guess + overall confidence.

    Rule: the most *precise* signal wins the location slot (an exact GPS fix
    always beats a country-level language hint). Overall confidence is that
    signal's confidence, nudged upward when independent signals corroborate.
    """
    if not result.signals:
        result.overall_confidence = 0.0
        result.best_precision = Precision.UNKNOWN
        return

    # Only *locating* signals that actually carry a location (coordinates or a
    # place name) may win the slot. This excludes both enrichment signals and
    # country-only votes (StreetCLIP agreement, OCR language) — those corroborate
    # but can't BE the answer, or they'd blank out the place while keeping a high
    # number. Fall back gracefully if nothing carries a location.
    located = [s for s in result.signals if s.locating and (s.coordinates or s.place)]
    pool = located or [s for s in result.signals if s.locating] or result.signals
    # Pick the winning signal: highest precision, then highest confidence.
    best = max(
        pool,
        key=lambda s: (precision_rank(s.precision), s.confidence),
    )
    result.best_coordinates = best.coordinates
    result.best_place = best.place
    result.best_precision = best.precision
    confidence = best.confidence

    # Corroboration bonus: only *independent locating* signals (corroborating=
    # True) count. Enrichment signals — nearby OSM features, "street imagery
    # exists", climate zone — are excluded on purpose: they'd fire for any real
    # place and would inflate even a wrong guess. A coordinate-bearing
    # corroborator must also be geographically consistent with the winner (else
    # it's evidence *against*, not for). Diminishing returns, capped so a pile
    # of weak hints never impersonates a GPS-grade certainty.
    others = []
    for s in result.signals:
        if s is best or not s.corroborating or s.precision == Precision.UNKNOWN:
            continue
        if s.coordinates and best.coordinates:
            if geoestimate_mod._haversine_km(s.coordinates, best.coordinates) > 200:
                continue  # points somewhere else — don't count as agreement
        elif not s.coordinates:
            # A country-level hint with no coordinates (OCR language, plate) only
            # corroborates if its candidate countries include the winner's.
            cands = s.evidence.get("candidate_countries")
            if cands and best.place:
                if not any(c.lower() in best.place.lower() for c in cands):
                    continue  # hint points at other countries — not agreement
        others.append(s)
    if others and best.precision != Precision.EXACT:
        bonus = min(0.15, 0.05 * len(others))
        confidence = min(0.85, confidence + bonus)

    # Down-weight when a second model disagreed but we didn't override (weak
    # disagreement) — the winner is contradicted, so it must not read confident.
    if result.meta.get("disagreement_penalty") and best.source != "resolved":
        confidence *= 0.6

    result.overall_confidence = confidence


def _build_stages(config: AnalyzeConfig):
    """Ordered (name, fn) stages selected by the config. Order matches STAGES."""
    stages = [
        ("exif", stage_exif),
        ("reverse_image_search", stage_reverse_image_search),
    ]
    if config.use_ocr:
        stages.append(("ocr", stage_ocr))
    if config.use_geoseer:
        stages.append(("geoseer", stage_geoseer))       # first: may short-circuit GeoCLIP
    if config.use_geoclip:
        stages.append(("landmark_model", stage_landmark_model))
    if config.use_streetclip:
        stages.append(("second_model", stage_second_model))
    if config.use_place_lookup:
        stages.append(("place_lookup", stage_place_lookup))
    stages.append(("resolve_models", stage_resolve_models))  # no-op without a 2nd model
    if config.use_street_match:
        stages.append(("street_match", stage_street_match))
    if config.use_osm:
        stages.append(("osm_crossref", stage_osm_crossref))
    if config.use_inaturalist:
        stages.append(("inaturalist", stage_inaturalist))
    if config.use_solar:
        stages.append(("shadow_flora", stage_shadow_flora))
    return stages


def analyze(image_path: str, config: Optional[AnalyzeConfig] = None) -> GeoResult:
    """Run the pipeline over one image and return the aggregated result.

    ``config`` selects which signals run and can carry API keys directly; it
    defaults to the classic env-driven behaviour.
    """
    config = config or AnalyzeConfig.from_env()
    result = GeoResult(image_path=image_path)
    result.meta["config"] = config

    # Keys: explicit config wins, else fall back to the environment.
    gk = config.geoseer_key or os.environ.get(geoseer_mod.API_KEY_ENV)
    if gk:
        result.meta["geoseer_key"] = gk
    mt = config.mapillary_token or os.environ.get(street_match_mod.TOKEN_ENV)
    if mt:
        result.meta["mapillary_token"] = mt
    sk = config.serpapi_key or os.environ.get(reverse_search_api_mod.SERPAPI_KEY_ENV)
    if sk:
        result.meta["serpapi_key"] = sk

    for _name, stage in _build_stages(config):
        stage(result)
    _aggregate(result)
    return result

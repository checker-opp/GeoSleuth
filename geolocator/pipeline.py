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
from . import ocr as ocr_mod
from . import osm as osm_mod
from . import plates as plates_mod
from . import reverse_search as reverse_search_mod
from . import secondmodel as secondmodel_mod
from . import solar as solar_mod
from . import street_match as street_match_mod
from .models import (
    Coordinates,
    GeoResult,
    Precision,
    Signal,
    precision_rank,
)

# Confidence assigned to a GPS fix. Not 1.0: EXIF GPS can be spoofed, stale,
# or reflect where a photo was edited rather than shot — high, not certain.
EXIF_GPS_CONFIDENCE = 0.95


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
def stage_reverse_image_search(result: GeoResult) -> None:  # Phase 4
    """Attach manual reverse-image-search pivot links (no scraping / no keys).

    These are actionable next-steps, not location evidence, so they're stored in
    ``result.meta`` and rendered separately rather than as a signal.
    """
    pivots = reverse_search_mod.build_pivots()
    result.meta["pivots"] = [
        {"engine": p.engine, "url": p.url, "note": p.note} for p in pivots
    ]


def stage_landmark_model(result: GeoResult) -> None:  # Phase 2 (GeoCLIP)
    """ML coordinate estimation for images with no usable metadata.

    Skipped when we already have an exact GPS fix (no point running an expensive
    model to second-guess real coordinates) or when the ML extras aren't
    installed.
    """
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


# Opt-in: StreetCLIP is a heavy second model (~1.6 GB, slow on CPU). Off unless
# GEOLOCATOR_STREETCLIP is set, so normal runs stay fast.
STREETCLIP_ENV = "GEOLOCATOR_STREETCLIP"


def streetclip_enabled() -> bool:
    return bool(os.environ.get(STREETCLIP_ENV))


def stage_second_model(result: GeoResult) -> None:  # optional StreetCLIP cross-check
    """Independent second opinion on the country via StreetCLIP.

    Opt-in (GEOLOCATOR_STREETCLIP=1) because the model is large and slow on CPU.
    When it AGREES with the visual (GeoCLIP) country, that's genuine
    cross-model corroboration and boosts confidence; a disagreement is surfaced
    as a caution rather than silently ignored.
    """
    if not streetclip_enabled():
        return

    # Compare against the current best locating guess's country, if we have one.
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

    agrees = (
        geoclip_country is not None
        and pred.country.lower() in geoclip_country.lower()
    )
    if agrees:
        result.add(
            Signal(
                source="streetclip",
                description=f"StreetCLIP independently agrees on {pred.country} "
                f"(p={pred.score:.2f}) — cross-model corroboration",
                confidence=0.40,
                precision=Precision.COUNTRY,
                corroborating=True,
                evidence={"country": pred.country, "score": round(pred.score, 3),
                          "candidate_countries": [pred.country]},
            )
        )
    else:
        result.add(
            Signal(
                source="streetclip",
                description=f"StreetCLIP's top country is {pred.country} "
                f"(p={pred.score:.2f})"
                + (f", differing from the visual estimate ({geoclip_country})"
                   if geoclip_country else ""),
                confidence=0.30,
                precision=Precision.COUNTRY,
                # An independent locator: if there's no GeoCLIP guess it can even
                # stand in; if it disagrees it's shown but not treated as agreement.
                corroborating=False,
                evidence={"country": pred.country, "score": round(pred.score, 3)},
            )
        )
        if geoclip_country:
            result.note(
                f"Model disagreement: GeoCLIP→{geoclip_country}, "
                f"StreetCLIP→{pred.country}. Treat the country as uncertain."
            )


def _country_of(place: str) -> Optional[str]:
    """Last comma-separated component of a Nominatim display name is the country."""
    if not place:
        return None
    parts = [p.strip() for p in place.split(",") if p.strip()]
    return parts[-1] if parts else None


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

    res = street_match_mod.find_street_imagery(coord)
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
    ("landmark_model", stage_landmark_model),
    ("second_model", stage_second_model),
    ("place_lookup", stage_place_lookup),
    ("street_match", stage_street_match),
    ("osm_crossref", stage_osm_crossref),
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

    result.overall_confidence = confidence


def analyze(image_path: str) -> GeoResult:
    """Run the full pipeline over one image and return the aggregated result."""
    result = GeoResult(image_path=image_path)
    for _name, stage in STAGES:
        stage(result)
    _aggregate(result)
    return result

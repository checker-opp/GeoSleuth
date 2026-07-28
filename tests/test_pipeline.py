"""Offline unit tests — no network, no external binaries required.

Covers the pure logic: coordinate conversion/validation, signal aggregation,
precision ranking, and the language->country hint map.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geolocator import exif as exif_mod
from geolocator import geoestimate as ge
from geolocator import solar as solar_mod
from geolocator import climate as climate_mod
from geolocator import plates as plates_mod
from geolocator import reverse_search as reverse_search_mod
from geolocator import street_match as street_match_mod
from geolocator.models import Coordinates, GeoResult, Precision, Signal, precision_rank
from geolocator import pipeline
from geolocator.ocr import LANG_TO_COUNTRIES


# --- coordinate helpers ---------------------------------------------------- #
def test_dms_to_decimal_north_east():
    val = exif_mod._dms_to_decimal((48, 51, 30.24), "N")
    assert abs(val - 48.8584) < 1e-4


def test_dms_to_decimal_south_west_is_negative():
    assert exif_mod._dms_to_decimal((33, 51, 0), "S") < 0
    assert exif_mod._dms_to_decimal((70, 0, 0), "W") < 0


def test_valid_coord_rejects_null_island():
    assert not exif_mod._valid_coord(0.0, 0.0)


def test_valid_coord_rejects_out_of_range():
    assert not exif_mod._valid_coord(200.0, 0.0)
    assert not exif_mod._valid_coord(0.0, 999.0)


def test_valid_coord_accepts_real_point():
    assert exif_mod._valid_coord(48.8584, 2.2945)


# --- precision ranking ----------------------------------------------------- #
def test_precision_rank_ordering():
    assert precision_rank(Precision.EXACT) > precision_rank(Precision.CITY)
    assert precision_rank(Precision.COUNTRY) > precision_rank(Precision.UNKNOWN)


# --- aggregation ----------------------------------------------------------- #
def test_aggregate_picks_most_precise_signal():
    r = GeoResult(image_path="x.jpg")
    r.add(Signal("ocr", "lang hint", 0.3, Precision.COUNTRY))
    r.add(
        Signal(
            "exif",
            "gps",
            0.95,
            Precision.EXACT,
            coordinates=Coordinates(48.85, 2.29),
            place="Paris",
        )
    )
    pipeline._aggregate(r)
    assert r.best_precision == Precision.EXACT
    assert r.best_place == "Paris"
    assert r.overall_confidence == 0.95


def test_aggregate_empty_is_zero_confidence():
    r = GeoResult(image_path="x.jpg")
    pipeline._aggregate(r)
    assert r.overall_confidence == 0.0
    assert r.best_precision == Precision.UNKNOWN


def test_corroboration_bonus_capped():
    r = GeoResult(image_path="x.jpg")
    # one country-level winner + several weak *independent locating* hints
    r.add(Signal("a", "hint", 0.30, Precision.COUNTRY, corroborating=True))
    for i in range(10):
        r.add(Signal(f"s{i}", "hint", 0.20, Precision.COUNTRY, corroborating=True))
    pipeline._aggregate(r)
    # bonus is capped; weak hints must never reach GPS-grade certainty
    assert r.overall_confidence <= 0.85
    assert r.overall_confidence > 0.30  # but corroboration did raise it


def test_enrichment_never_wins_location_slot():
    # Regression: a coarse (country-level) ML locator must keep the location
    # slot even when higher-precision *enrichment* signals (city-level OSM/
    # street, locating=False) are present — those carry no place of their own.
    r = GeoResult(image_path="x.jpg")
    r.add(Signal("ml_geoclip", "Karachi (spread out)", 0.35, Precision.COUNTRY,
                 coordinates=Coordinates(24.86, 67.01), place="Karachi, Pakistan"))
    r.add(Signal("osm", "nearby places", 0.20, Precision.CITY,
                 coordinates=Coordinates(24.86, 67.01), locating=False))
    r.add(Signal("street_match", "imagery exists", 0.20, Precision.CITY,
                 coordinates=Coordinates(24.86, 67.01), locating=False))
    pipeline._aggregate(r)
    assert r.best_place == "Karachi, Pakistan"      # locator won, not enrichment
    assert r.best_precision == Precision.COUNTRY
    assert r.overall_confidence == 0.35


def test_country_only_locator_cannot_win_slot():
    # Regression: a country-only locating signal (no coords/place, e.g. a
    # StreetCLIP agreement vote) with HIGHER confidence must not steal the
    # location slot from a coordinate-bearing locator — that produced a blank
    # "best guess" at an inflated confidence.
    r = GeoResult(image_path="x.jpg")
    r.add(Signal("ml_geoclip", "Karachi", 0.35, Precision.COUNTRY,
                 coordinates=Coordinates(24.86, 67.01), place="Karachi, Pakistan",
                 corroborating=True))
    r.add(Signal("streetclip", "Pakistan vote", 0.40, Precision.COUNTRY,
                 corroborating=True, evidence={"candidate_countries": ["Pakistan"]}))
    pipeline._aggregate(r)
    assert r.best_place == "Karachi, Pakistan"   # coord-bearing locator wins
    assert r.best_coordinates is not None
    assert round(r.overall_confidence, 2) == 0.40  # 0.35 + one agreeing corroborator


def test_enrichment_signals_do_not_boost_confidence():
    # An ML guess plus enrichment-only signals (OSM/street/climate, i.e.
    # corroborating=False) must NOT raise confidence — those fire for any real
    # place and would inflate even a wrong guess.
    r = GeoResult(image_path="x.jpg")
    r.add(Signal("ml", "est", 0.60, Precision.CITY, coordinates=Coordinates(1, 1)))
    r.add(Signal("osm", "nearby places", 0.20, Precision.CITY,
                 coordinates=Coordinates(1, 1), corroborating=False))
    r.add(Signal("street_match", "imagery exists", 0.20, Precision.CITY,
                 coordinates=Coordinates(1, 1), corroborating=False))
    pipeline._aggregate(r)
    assert r.overall_confidence == 0.60  # unchanged by enrichment


def test_exact_signal_gets_no_corroboration_inflation():
    r = GeoResult(image_path="x.jpg")
    r.add(Signal("exif", "gps", 0.95, Precision.EXACT, coordinates=Coordinates(1, 1)))
    r.add(Signal("ocr", "hint", 0.30, Precision.COUNTRY))
    pipeline._aggregate(r)
    assert r.overall_confidence == 0.95  # unchanged — GPS doesn't get a bonus


# --- language map ---------------------------------------------------------- #
def test_language_map_has_common_languages():
    for code in ("en", "fr", "ja", "ar", "ru"):
        assert code in LANG_TO_COUNTRIES
        assert LANG_TO_COUNTRIES[code]


# --- ML geo-estimation (pure logic, no model needed) ----------------------- #
def test_haversine_known_distance():
    # Paris -> London is ~344 km
    paris = Coordinates(48.8566, 2.3522)
    london = Coordinates(51.5074, -0.1278)
    d = ge._haversine_km(paris, london)
    assert 300 < d < 380


def test_assess_tight_cluster_is_city_level():
    top = Coordinates(48.8566, 2.3522)
    preds = [
        ge.GeoPrediction(top, 0.4),
        ge.GeoPrediction(Coordinates(48.86, 2.35), 0.3),  # ~5 km away
    ]
    precision, conf, spread = ge.assess(ge.GeoEstimateResult(True, preds))
    assert precision == Precision.CITY
    assert conf <= 0.6  # ML never rivals a GPS fix
    assert spread < 25


def test_assess_scattered_is_country_level_low_conf():
    preds = [
        ge.GeoPrediction(Coordinates(48.85, 2.35), 0.1),
        ge.GeoPrediction(Coordinates(-33.86, 151.2), 0.1),  # Sydney — far away
    ]
    precision, conf, spread = ge.assess(ge.GeoEstimateResult(True, preds))
    assert precision == Precision.COUNTRY
    assert conf <= 0.35
    assert spread > 1500


def test_assess_empty_predictions():
    precision, conf, spread = ge.assess(ge.GeoEstimateResult(True, []))
    assert precision == Precision.UNKNOWN
    assert conf == 0.0


def test_ml_stage_skipped_when_exact_gps_present():
    r = GeoResult(image_path="x.jpg")
    r.add(Signal("exif", "gps", 0.95, Precision.EXACT, coordinates=Coordinates(1, 1)))
    before = len(r.signals)
    pipeline.stage_landmark_model(r)  # must early-return, not run the model
    assert len(r.signals) == before


# --- Phase 3: solar corroboration ------------------------------------------ #
def test_parse_exif_datetime():
    dt = solar_mod.parse_exif_datetime("2024:06:15 14:30:00")
    assert dt is not None and dt.year == 2024 and dt.hour == 14


def test_parse_exif_datetime_bad_input():
    assert solar_mod.parse_exif_datetime("not a date") is None
    assert solar_mod.parse_exif_datetime(None) is None


def test_parse_utc_offset_signs():
    assert solar_mod.parse_utc_offset("+05:30").total_seconds() == 5.5 * 3600
    assert solar_mod.parse_utc_offset("-08:00").total_seconds() == -8 * 3600


def test_longitude_from_offset():
    from datetime import timedelta

    assert solar_mod.longitude_from_offset(timedelta(hours=2)) == 30.0
    assert solar_mod.longitude_from_offset(timedelta(hours=-5)) == -75.0


def test_solar_longitude_consistency_paris_dst():
    # Paris (2.29E) in summer reports +02:00 (DST). The DST-aware check should
    # judge this consistent, not flag it.
    info = solar_mod.analyze(
        Coordinates(48.8584, 2.2945), "2024:06:15 14:30:00", "+02:00"
    )
    assert info.longitude_consistent is True
    assert info.sun_elevation is not None and info.sun_elevation > 0  # daytime


def test_solar_longitude_inconsistency_flagged():
    # Claiming Tokyo's timezone (+09:00 -> 135E) but coordinates in the Atlantic.
    info = solar_mod.analyze(Coordinates(0.0, -30.0), None, "+09:00")
    assert info.longitude_consistent is False


# --- Phase 3: climate descriptor ------------------------------------------- #
def test_climate_latitude_zones():
    assert "tropical" in climate_mod.latitude_zone(5.0)
    assert climate_mod.latitude_zone(45.0) == "temperate"
    assert climate_mod.latitude_zone(80.0) == "polar"


def test_climate_hemisphere():
    assert "Southern" in climate_mod.hemisphere(Coordinates(-33.0, 151.0))
    assert "Northern" in climate_mod.hemisphere(Coordinates(48.0, 2.0))


# --- Phase 3: current-best-coord helper ------------------------------------ #
def test_current_best_coord_prefers_precise():
    r = GeoResult(image_path="x.jpg")
    r.add(Signal("ocr", "hint", 0.3, Precision.COUNTRY))  # no coords
    r.add(Signal("ml", "est", 0.6, Precision.CITY, coordinates=Coordinates(1, 2)))
    r.add(Signal("exif", "gps", 0.95, Precision.EXACT, coordinates=Coordinates(9, 9)))
    coord = pipeline._current_best_coord(r)
    assert coord.lat == 9 and coord.lon == 9


def test_current_best_coord_none_when_no_coords():
    r = GeoResult(image_path="x.jpg")
    r.add(Signal("ocr", "hint", 0.3, Precision.COUNTRY))
    assert pipeline._current_best_coord(r) is None


# --- Phase 4: license-plate hints ------------------------------------------ #
def test_plate_uk_format():
    hint = plates_mod.detect("some text AB12 CDE more")
    assert hint.matched and hint.country == "United Kingdom"


def test_plate_india_format():
    hint = plates_mod.detect("MH12 AB 1234")
    assert hint.matched and hint.country == "India"


def test_plate_plain_text_no_false_country():
    hint = plates_mod.detect("welcome to the temperate riverside cafe")
    assert hint.country is None  # no distinctive plate -> no country claim


def test_plate_empty_text():
    assert plates_mod.detect("").matched is False


# --- Phase 4: reverse-search pivots ---------------------------------------- #
def test_pivots_yandex_first():
    pivots = reverse_search_mod.build_pivots()
    assert pivots[0].engine.startswith("Yandex")
    assert all(p.url.startswith("http") for p in pivots)


# --- Phase 4: street-match bbox + token gate ------------------------------- #
def test_street_bbox_contains_point():
    c = Coordinates(48.8584, 2.2945)
    west, south, east, north = street_match_mod._bbox(c, 100.0)
    assert west < c.lon < east
    assert south < c.lat < north


def test_street_token_gate(monkeypatch):
    monkeypatch.delenv("MAPILLARY_TOKEN", raising=False)
    assert street_match_mod.token_configured() is False
    res = street_match_mod.mapillary_nearby(Coordinates(0, 0))
    assert res.available is False and "MAPILLARY_TOKEN" in res.reason


def test_kartaview_viewer_url():
    url = street_match_mod._kartaview_viewer_url(2061158, 3)
    assert url == "https://kartaview.org/details/2061158/3"


def test_find_street_imagery_falls_back_to_kartaview(monkeypatch):
    # No token -> Mapillary is skipped -> KartaView is consulted. Stub the
    # KartaView call so the test stays offline.
    monkeypatch.delenv("MAPILLARY_TOKEN", raising=False)
    stub = street_match_mod.StreetMatchResult(
        available=True, provider="kartaview",
        images=[street_match_mod.StreetImage(
            id="1", lat=44.4, lon=26.1, provider="KartaView",
            viewer_url="https://kartaview.org/details/1/0")],
    )
    monkeypatch.setattr(street_match_mod, "kartaview_nearby", lambda *a, **k: stub)
    res = street_match_mod.find_street_imagery(Coordinates(44.4, 26.1))
    assert res.available and res.images[0].provider == "KartaView"


# --- OCR candidate-line extraction ----------------------------------------- #
def test_ocr_candidate_lines_filters_noise():
    from geolocator import ocr as ocr_mod
    raw = "IMPERIAL WATCH CO\nbf x8\n!!\nK.B. SARKAR\ni a"
    lines = ocr_mod._candidate_lines(raw)
    assert any("IMPERIAL" in l for l in lines)
    assert any("SARKAR" in l for l in lines)
    assert "bf x8" not in lines          # no 4+ letter word
    assert "i a" not in lines            # too short


# --- confidence: country-overlap gate for coordinate-less hints ------------ #
def test_country_hint_boosts_only_when_country_matches():
    from geolocator.models import GeoResult, Signal, Coordinates, Precision
    # ML says Pakistan; an English-language hint (candidate countries lack
    # Pakistan) must NOT boost confidence.
    r = GeoResult(image_path="x.jpg")
    r.add(Signal("ml", "est", 0.35, Precision.COUNTRY,
                 coordinates=Coordinates(24.86, 67.01),
                 place="Karachi, Pakistan", corroborating=True))
    r.add(Signal("ocr", "english", 0.30, Precision.COUNTRY, corroborating=True,
                 evidence={"candidate_countries": ["US", "UK", "India"]}))
    pipeline._aggregate(r)
    assert r.overall_confidence == 0.35   # no boost — India != Pakistan

    # Now a hint whose countries include the winner's -> boosts.
    r2 = GeoResult(image_path="x.jpg")
    r2.add(Signal("ml", "est", 0.35, Precision.COUNTRY,
                  coordinates=Coordinates(24.86, 67.01),
                  place="Karachi, Pakistan", corroborating=True))
    r2.add(Signal("ocr", "urdu", 0.30, Precision.COUNTRY, corroborating=True,
                  evidence={"candidate_countries": ["Pakistan", "India"]}))
    pipeline._aggregate(r2)
    assert r2.overall_confidence > 0.35   # boosted — Pakistan matches


def test_inconsistent_coord_corroborator_excluded():
    from geolocator.models import GeoResult, Signal, Coordinates, Precision
    r = GeoResult(image_path="x.jpg")
    r.add(Signal("ml", "est", 0.35, Precision.COUNTRY,
                 coordinates=Coordinates(24.86, 67.01), place="Karachi, Pakistan",
                 corroborating=True))
    # A locating signal 8000 km away must not count as agreement.
    r.add(Signal("ocr_place", "far match", 0.55, Precision.CITY,
                 coordinates=Coordinates(48.85, 2.35), corroborating=True))
    # (ocr_place is CITY so it would win the slot; assert the ML stays uncorroborated
    #  when IT is the winner is not the case here — instead check the far signal wins
    #  but gets no bonus from the distant ML.)
    pipeline._aggregate(r)
    # ocr_place (CITY) wins; ML is >200km away so provides no bonus.
    assert r.best_precision == Precision.CITY
    assert r.overall_confidence == 0.55


# --- StreetCLIP helpers ---------------------------------------------------- #
def test_country_of_parses_display_name():
    assert pipeline._country_of("Princess St, Lyari, Karachi, Sindh, Pakistan") == "Pakistan"
    assert pipeline._country_of("") is None


def test_streetclip_off_by_default(monkeypatch):
    monkeypatch.delenv("GEOLOCATOR_STREETCLIP", raising=False)
    assert pipeline.streetclip_enabled() is False
    monkeypatch.setenv("GEOLOCATOR_STREETCLIP", "1")
    assert pipeline.streetclip_enabled() is True


# --- batch input expansion ------------------------------------------------- #
def test_expand_inputs_dir_and_missing(tmp_path):
    from geolocator import cli
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    images, errors = cli._expand_inputs([str(tmp_path), "does_not_exist.jpg"])
    assert len(images) == 2                      # jpg + png, not txt
    assert any("not found" in e for e in errors)


# --- model disagreement resolution (StreetCLIP vs GeoCLIP) ----------------- #
def _disagreement_result(topk, ml_place="Reforma, Mexico City, Mexico",
                         sc_country="Indonesia", sc_score=0.78):
    r = GeoResult(image_path="x.jpg")
    r.add(Signal("ml_geoclip", "geo", 0.60, Precision.CITY,
                 coordinates=Coordinates(topk[0]["lat"], topk[0]["lon"]),
                 place=ml_place, corroborating=True, evidence={"top_k": topk}))
    r.add(Signal("streetclip", "sc", sc_score, Precision.COUNTRY, locating=False,
                 evidence={"country": sc_country, "score": sc_score,
                           "candidate_countries": [sc_country]}))
    r.meta["streetclip"] = {"country": sc_country, "score": sc_score}
    return r


def test_resolve_country_override(monkeypatch):
    from geolocator import geocode as geocode_mod
    from geolocator.geocode import Place
    topk = [{"lat": 19.4, "lon": -99.1, "prob": 0.2},
            {"lat": 19.4, "lon": -99.2, "prob": 0.2}]
    r = _disagreement_result(topk)
    # every GeoCLIP point reverse-geocodes to Mexico -> no re-rank match
    monkeypatch.setattr(geocode_mod, "reverse",
                        lambda c, **k: Place(display_name="X, Mexico", country="Mexico"))
    pipeline.stage_resolve_models(r)
    pipeline._aggregate(r)
    assert r.best_place == "Indonesia"           # override to StreetCLIP country
    assert r.best_precision == Precision.COUNTRY
    assert r.best_coordinates is None
    assert round(r.overall_confidence, 2) == 0.30
    assert any(s.source == "ml_geoclip" and not s.locating for s in r.signals)  # demoted


def test_resolve_rerank_keeps_precision(monkeypatch):
    from geolocator import geocode as geocode_mod
    from geolocator.geocode import Place
    # top pick Mexico, but #2 is an Indonesian prediction
    topk = [{"lat": 19.4, "lon": -99.1, "prob": 0.2},
            {"lat": -6.2, "lon": 106.8, "prob": 0.18}]

    def fake_reverse(c, **k):
        if -11 <= c.lat <= 6:  # Indonesia latitudes
            return Place(display_name="Jakarta, Indonesia", country="Indonesia")
        return Place(display_name="X, Mexico", country="Mexico")

    r = _disagreement_result(topk)
    monkeypatch.setattr(geocode_mod, "reverse", fake_reverse)
    pipeline.stage_resolve_models(r)
    pipeline._aggregate(r)
    assert r.best_precision == Precision.REGION           # kept a real coordinate
    assert r.best_coordinates is not None
    assert "Indonesia" in (r.best_place or "")


def test_resolve_agreement_boosts(monkeypatch):
    from geolocator import geocode as geocode_mod
    topk = [{"lat": -6.2, "lon": 106.8, "prob": 0.3}]
    r = _disagreement_result(topk, ml_place="Jakarta, Indonesia", sc_country="Indonesia")
    pipeline.stage_resolve_models(r)   # agree path doesn't hit the network
    pipeline._aggregate(r)
    assert "Indonesia" in (r.best_place or "")
    assert r.best_precision == Precision.CITY             # GeoCLIP retained
    assert r.overall_confidence > 0.60                    # corroboration boost


def test_resolve_never_overrides_exif():
    r = GeoResult(image_path="x.jpg")
    r.add(Signal("exif", "gps", 0.95, Precision.EXACT,
                 coordinates=Coordinates(19.4, -99.1), place="Mexico City, Mexico"))
    r.add(Signal("streetclip", "sc", 0.9, Precision.COUNTRY, locating=False,
                 evidence={"country": "Indonesia", "score": 0.9}))
    r.meta["streetclip"] = {"country": "Indonesia", "score": 0.9}
    pipeline.stage_resolve_models(r)
    pipeline._aggregate(r)
    assert r.best_precision == Precision.EXACT
    assert r.overall_confidence == 0.95                   # GPS untouched


def test_resolve_weak_disagreement_downweights_only():
    topk = [{"lat": 19.4, "lon": -99.1, "prob": 0.2}]
    r = _disagreement_result(topk, sc_score=0.30)          # below override threshold
    pipeline.stage_resolve_models(r)
    pipeline._aggregate(r)
    assert "Mexico" in (r.best_place or "")                # GeoCLIP kept
    assert round(r.overall_confidence, 2) == 0.36          # 0.60 * 0.6 down-weight


# --- AnalyzeConfig / stage selection --------------------------------------- #
def test_build_stages_default_includes_core():
    from geolocator.models import AnalyzeConfig
    names = [n for n, _ in pipeline._build_stages(AnalyzeConfig())]
    assert "landmark_model" in names and "geoseer" in names
    assert "second_model" not in names   # StreetCLIP off by default


def test_build_stages_geoseer_only_skips_clip():
    from geolocator.models import AnalyzeConfig
    cfg = AnalyzeConfig(use_geoclip=False, use_streetclip=False, use_geoseer=True)
    names = [n for n, _ in pipeline._build_stages(cfg)]
    assert "geoseer" in names
    assert "landmark_model" not in names and "second_model" not in names


def test_build_stages_toggles_off():
    from geolocator.models import AnalyzeConfig
    cfg = AnalyzeConfig(use_ocr=False, use_osm=False, use_inaturalist=False,
                        use_street_match=False, use_solar=False)
    names = [n for n, _ in pipeline._build_stages(cfg)]
    for gone in ("ocr", "osm_crossref", "inaturalist", "street_match", "shadow_flora"):
        assert gone not in names


# --- GeoSeer API locator --------------------------------------------------- #
def test_geoseer_no_key(monkeypatch):
    from geolocator import geoseer as geoseer_mod
    monkeypatch.delenv("GEOSEER_API_KEY", raising=False)
    assert geoseer_mod.api_key_configured() is False
    res = geoseer_mod.predict("x.jpg")
    assert res.available is False and "GEOSEER_API_KEY" in res.reason


def test_geoseer_wins_and_survives_streetclip(monkeypatch):
    from geolocator import geoseer as geoseer_mod, geocode as geocode_mod
    from geolocator.geocode import Place
    r = GeoResult(image_path="x.jpg")
    # GeoCLIP wrongly says Mexico
    r.add(Signal("ml_geoclip", "geo", 0.60, Precision.CITY,
                 coordinates=Coordinates(19.4, -99.1), place="Reforma, Mexico City, Mexico",
                 corroborating=True, evidence={"top_k": [{"lat": 19.4, "lon": -99.1, "prob": 0.2}]}))
    fake = geoseer_mod.GeoSeerResult(
        available=True, requests_remaining=8,
        locations=[geoseer_mod.GeoSeerLocation(lat=-6.2, lon=106.8,
                   address="South Jakarta, Indonesia", confidence=0.95)])
    monkeypatch.setattr(geoseer_mod, "predict", lambda p, **k: fake)
    monkeypatch.setattr(geocode_mod, "reverse",
                        lambda c, **k: Place(display_name="WTC 2, South Jakarta, Indonesia",
                                             country="Indonesia"))
    r.meta["geoseer_key"] = "test-key"   # stage reads the key from meta
    pipeline.stage_geoseer(r)
    # A GeoSeer signal now outranks the GeoCLIP one.
    best = max((s for s in r.signals if s.locating and s.place),
               key=lambda s: (precision_rank(s.precision), s.confidence))
    assert best.source == "geoseer"
    # StreetCLIP disagrees (says Mexico) — must NOT override the stronger GeoSeer.
    r.add(Signal("streetclip", "sc", 0.9, Precision.COUNTRY, locating=False,
                 evidence={"country": "Mexico", "score": 0.9}))
    r.meta["streetclip"] = {"country": "Mexico", "score": 0.9}
    pipeline.stage_resolve_models(r)
    pipeline._aggregate(r)
    assert "Indonesia" in (r.best_place or "")     # GeoSeer stood its ground
    assert r.best_precision == Precision.CITY


def test_geoseer_skipped_when_exif_present(monkeypatch):
    from geolocator import geoseer as geoseer_mod
    called = {"n": 0}
    monkeypatch.setattr(geoseer_mod, "predict",
                        lambda p, **k: called.__setitem__("n", called["n"] + 1))
    r = GeoResult(image_path="x.jpg")
    r.meta["geoseer_key"] = "test-key"   # key IS available...
    r.add(Signal("exif", "gps", 0.95, Precision.EXACT, coordinates=Coordinates(1, 1)))
    pipeline.stage_geoseer(r)
    assert called["n"] == 0     # ...but no scarce API call spent when GPS already known


# --- iNaturalist enrichment ------------------------------------------------ #
def test_inaturalist_enrichment(monkeypatch):
    from geolocator import inaturalist as inat_mod
    r = GeoResult(image_path="x.jpg")
    r.add(Signal("ml_geoclip", "geo", 0.6, Precision.CITY,
                 coordinates=Coordinates(-6.2, 106.8), place="Jakarta, Indonesia"))
    fake = inat_mod.InatResult(available=True, taxa=[
        inat_mod.Taxon(name="Varanus salvator", common_name="Common Water Monitor",
                       group="Reptilia", observations=10)])
    monkeypatch.setattr(inat_mod, "nearby_taxa", lambda c, **k: fake)
    pipeline.stage_inaturalist(r)
    sig = next(s for s in r.signals if s.source == "inaturalist")
    assert sig.locating is False and "Water Monitor" in sig.description
    pipeline._aggregate(r)
    assert r.best_precision == Precision.CITY     # enrichment didn't change the guess


# --- config-driven pipeline ------------------------------------------------ #
def test_config_geoseer_only_skips_clip():
    from geolocator.models import AnalyzeConfig
    cfg = AnalyzeConfig(use_geoclip=False, use_streetclip=False, use_geoseer=True)
    names = [n for n, _ in pipeline._build_stages(cfg)]
    assert "landmark_model" not in names   # no local GeoCLIP
    assert "second_model" not in names     # no StreetCLIP
    assert "geoseer" in names


def test_config_all_on_matches_full_pipeline():
    from geolocator.models import AnalyzeConfig
    cfg = AnalyzeConfig(use_streetclip=True)
    names = [n for n, _ in pipeline._build_stages(cfg)]
    for s in ("exif", "ocr", "landmark_model", "geoseer", "second_model",
              "place_lookup", "street_match", "osm_crossref", "inaturalist"):
        assert s in names


def test_webui_config_from_form_geoseer_only():
    from geolocator import webui
    form = {"geoseer_only": "on", "use_geoseer": "on", "use_ocr": "on"}
    cfg = webui._config_from_form(form)
    assert cfg.use_geoclip is False and cfg.use_streetclip is False
    assert cfg.use_geoseer is True and cfg.use_ocr is True


# --- model manager --------------------------------------------------------- #
def test_modelmgr_unknown_model():
    from geolocator import modelmgr
    r = modelmgr.start_download("bogus")
    assert r["ok"] is False


def test_modelmgr_status_shape():
    from geolocator import modelmgr
    s = modelmgr.all_status()
    assert set(s.keys()) == {"geoclip", "streetclip"}
    for m in s.values():
        assert "state" in m and "percent" in m and "approx_gb" in m


# --- extractor safety ------------------------------------------------------ #
def test_extract_missing_file_returns_empty_shell():
    data = exif_mod.extract("does_not_exist_12345.jpg")
    assert data.has_gps is False
    assert data.source == "none"


if __name__ == "__main__":
    import subprocess

    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))

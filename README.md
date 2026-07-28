# geolocator — image → location OSINT tool

Guess **where a photo was taken** from the image alone. Feed it a picture; it
runs a layered pipeline of location signals and returns a best-guess location,
a **confidence score**, and the **evidence trail** that produced it — never a
bare guess.

> **Scope of this release (Phase 1):** the reliable core — EXIF GPS extraction,
> Nominatim reverse geocoding, and OCR-based language/text hinting. The heavier
> layers (ML geo-estimation, reverse image search, street matching, shadow &
> flora cross-referencing) are scaffolded and land in later phases — see
> [Roadmap](#roadmap).

---

## What it does today

| Signal | What it gives you | Reliability |
|--------|-------------------|-------------|
| **EXIF GPS** → Nominatim | Exact coordinates + full street address | ★★★★★ when GPS is present |
| **EXIF metadata** | Camera, timestamp, timezone-offset longitude hint | context only |
| **OCR** (Tesseract) | Visible text → detected language → candidate countries | ★★☆ coarse hint |
| **ML estimate** (GeoCLIP) | Predicted coordinates from visual content alone — the no-metadata workhorse | ★★★☆ on distinctive scenes, honestly low on generic/indoor |
| **OCR place lookup** | A business/place name OCR reads, geocoded and cross-checked against the visual estimate | ★★★★ when it matches — turns a sign into coordinates |
| **StreetCLIP** (opt-in) | Independent 2nd model's country vote; boosts confidence when it agrees with GeoCLIP | cross-model corroboration (GPU recommended) |
| **License plate** | Distinctive plate format in OCR text → country | ★★☆ conservative |
| **OSM cross-ref** (Overpass) | Named features near the candidate — corroborates a real place | corroboration |
| **Solar / climate** | Timezone↔longitude consistency, sun elevation, climate zone | corroboration / sanity check |
| **Street match** (Mapillary / KartaView) | Nearby street-level imagery to visually confirm the guess | pivot / corroboration |
| **Reverse-search pivots** | Ready-to-open Yandex/Lens/Bing/TinEye upload links | manual next step |

The pipeline runs stages in priority order (cheap & deterministic first):

```
EXIF → reverse-search pivots → OCR+plates → GeoCLIP → [StreetCLIP] → OCR place lookup
     → street match → OSM cross-ref → solar/climate
```

**Layering rules that keep it honest:**
- The **most precise** signal wins the location; corroboration only nudges confidence, capped so weak hints never impersonate a GPS fix.
- The **GeoCLIP** stage runs only when there's no exact GPS — no point second-guessing real coordinates — and its confidence comes from how tightly its top-5 guesses cluster (scattered → low-confidence, country-level at best).
- The **solar** stage can *contradict* a guess: if the EXIF timezone offset is inconsistent with the candidate longitude, it says so.

---

## Install

```bash
pip install -r requirements.txt
```

Or install as a command (`geolocate`):

```bash
pip install -e .
```

### Optional external tools (strongly recommended)

Two signals depend on external binaries. The tool **auto-detects** them and
degrades gracefully if they're missing — but installing them unlocks a lot.

| Tool | Enables | Windows install |
|------|---------|-----------------|
| **ExifTool** | Comprehensive EXIF (RAW/HEIC/XMP), the most robust GPS extraction | Download from [exiftool.org](https://exiftool.org), rename `exiftool(-k).exe` → `exiftool.exe`, put it on your `PATH` |
| **Tesseract** | OCR (the entire text/language signal) | Install the [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki), then add its folder to `PATH` |

Without ExifTool, EXIF still works via the pure-Python `exifread`/Pillow
fallback. Without Tesseract, OCR is skipped (the tool tells you so).

### ML geo-estimation (Phase 2 — optional, heavy)

The GeoCLIP visual estimator is the thing that actually locates images with no
metadata. It's a large dependency (torch), kept in a separate requirements file
so the core stays lightweight:

```bash
pip install -r requirements.txt -r requirements-ml.txt
```

Notes:
- **torch** is ~200 MB to download / ~2 GB installed; the pinned CPU build runs
  anywhere. For GPU, install the CUDA torch build from pytorch.org instead.
- **GeoCLIP** downloads its model weights (~hundreds of MB) on first run, then
  caches them.
- `transformers` is pinned to `<5` — GeoCLIP's encoder targets the 4.x CLIP API
  and 5.x breaks inference.
- With the extras absent, the ML stage is skipped gracefully (the tool says so).

---

## Usage

```bash
# Human-readable report
python -m geolocator path/to/photo.jpg

# Machine-readable JSON (add --verbose to include raw evidence)
python -m geolocator path/to/photo.jpg --json
python -m geolocator path/to/photo.jpg --json --verbose

# Batch: several files, or a whole directory (JSON becomes an array).
# The GeoCLIP model loads once and is reused across the batch.
python -m geolocator a.jpg b.jpg c.jpg
python -m geolocator ./my_photos/ --json

# If installed with `pip install -e .`
geolocate path/to/photo.jpg
```

### Example (image with GPS metadata)

```
  IMAGE → LOCATION  (geolocator v0.1.0)
  photo.jpg

  BEST GUESS
    Place       Avenue Gustave Eiffel, 7th Arrondissement, Paris, 75007, France
    Coordinates 48.858400, 2.294500
    Map         https://www.openstreetmap.org/?mlat=48.8584&mlon=2.2945#map=16/48.8584/2.2945
    Precision   exact coordinates
    Confidence  ██████████   95%

  EVIDENCE
    [exif] GPS coordinates embedded in image metadata → Avenue Gustave Eiffel …
          confidence 0.95 · exact
```

---

## How confidence works

Every stage emits **signals** — a finding + how much we trust it + how precise
it is (`exact` / `city` / `region` / `country` / `unknown`).

- The **most precise** signal wins the location slot. An exact GPS fix always
  beats a country-level language hint.
- **Locating vs. enrichment.** Only *independent locating* signals — ones that
  on their own say *where* the photo was taken (EXIF GPS, the ML estimate, an
  OCR place-name match, a StreetCLIP country vote, a plate country) — can raise
  confidence when they agree, and the bonus is capped so weak hints never reach
  GPS-grade certainty. *Enrichment* signals (nearby OSM features, "street
  imagery exists nearby", climate zone) are shown as evidence but **never boost
  the number** — they'd fire for any real place and would inflate even a wrong
  guess. So a GeoCLIP-only guess reports the model's own confidence, not a
  padded one.
- **Agreement must be real.** A corroborating signal only counts if it actually
  agrees with the winner: a coordinate-bearing hint must be within ~200 km, and
  a country-level hint's candidate countries must include the winner's country.
  A hint pointing elsewhere is excluded (and, for the second model, surfaced as
  a disagreement caution).
- **Enrichment never wins the location slot.** Only locating signals can become
  the reported location — an OSM/street-match signal describes a candidate but
  can't hijack the answer with a placeless coordinate.
- GPS is scored **0.95, not 1.0**: EXIF can be spoofed, stale, or reflect where
  a photo was *edited* rather than shot.
- The solar stage can **lower** trust: a timezone offset inconsistent with the
  candidate longitude is flagged as a warning.

This transparency is deliberate — for OSINT work, the reasoning matters as much
as the answer.

---

## A note on accuracy expectations

- **With GPS metadata:** 95%+ correct — this is essentially solved.
- **Without metadata:** genuinely hard. Most social-media images have EXIF
  stripped, so results depend on visible text, landmarks, and (in later phases)
  ML estimation. Expect **country/region** accuracy on good photos, not
  pinpoint coordinates on arbitrary ones. The tool always reports its
  confidence so a weak guess is never mistaken for a strong one.

---

## Roadmap

| Phase | Adds | Notes |
|-------|------|-------|
| **1 — done** | EXIF → Nominatim, OCR → language hint, confidence engine, CLI | the reliable core |
| **2 — done** | ML geo-estimation via GeoCLIP (image → coordinates, local inference) | the real no-metadata workhorse; needs a few GB RAM, GPU ideal |
| **3 — done** | Solar/timezone consistency (pysolar), climate-zone descriptor, OSM cross-ref (Overpass) | corroboration layer |
| **4 — done** | Reverse-search pivots (Yandex/Lens/Bing/TinEye), street matching (Mapillary + keyless KartaView), license-plate region | best-effort; keyed APIs gated behind env vars |

### What's intentionally left for later / needs your input

| Item | Why it's not automatic | To enable |
|------|------------------------|-----------|
| **Mapillary street matching** | better global coverage, needs a free API token | set `MAPILLARY_TOKEN` (see [dashboard](https://www.mapillary.com/dashboard/developers)); without it, street matching still works via keyless **KartaView** (sparser outside Europe) |
| **Automated reverse image search** | Yandex/Lens fight scraping; TinEye API is paid. We do **not** scrape or bypass CAPTCHAs | use the printed pivot links to search manually, or add a keyed TinEye/Bing client later |
| **Flora → region from the image** | needs a plant-ID vision model | Phase 3 currently describes climate from the *candidate* coordinate instead |
| **Shadow-angle → latitude** | needs CV shadow detection in the image | Phase 3 currently does the metadata-side solar check instead |

The pipeline (`geolocator/pipeline.py`) defines every stage in `STAGES` — adding
a new signal means writing one function and slotting it in.

---

## Configuration (environment variables)

| Variable | Effect |
|----------|--------|
| `MAPILLARY_TOKEN` | prefers Mapillary for street matching (better coverage); without it, keyless KartaView is used |
| `GEOLOCATOR_STREETCLIP` | set to `1` to enable the StreetCLIP second-model cross-check (heavy; GPU recommended — see note below) |
| `NO_COLOR` | disables ANSI colour in the terminal report |

> **StreetCLIP note:** the cross-check loads `geolocal/StreetCLIP` (~1.6 GB,
> CLIP-ViT-L/14). On a GPU it's quick; on CPU it's impractically slow (model
> load alone can exceed 10 minutes), which is why it's **off by default**. The
> integration is wired and unit-tested, but its inference was **not validated on
> CPU-only hardware** in development — enable it only with a GPU.

## Project layout

```
geolocator/
  cli.py            CLI + report formatting
  pipeline.py       stage orchestration + confidence aggregation
  models.py         Signal / GeoResult data structures
  # Phase 1 — reliable core
  exif.py           EXIF extraction (exiftool → exifread → Pillow fallback)
  geocode.py        Nominatim reverse + forward (place-name) geocoding
  ocr.py            Tesseract OCR (upscale/contrast/multi-PSM) + langdetect
  # Phase 2 — ML
  geoestimate.py    GeoCLIP image → coordinates (lazy, optional)
  secondmodel.py    StreetCLIP zero-shot country cross-check (opt-in, GPU)
  # Phase 3 — corroboration
  solar.py          timezone↔longitude + sun-elevation checks (pysolar)
  climate.py        coarse latitude/climate-zone descriptor
  osm.py            Overpass nearby-feature cross-reference
  # Phase 4 — best-effort external
  plates.py         license-plate format → country hint
  reverse_search.py reverse-image-search pivot links
  street_match.py   Mapillary (token) + KartaView (keyless) street imagery
tests/
  test_pipeline.py  44 offline tests (no network / binaries needed)
```

---

## Legal & ethical use

This is an OSINT research tool. Geolocating images can implicate people's
privacy and safety. Use it only on images you're authorized to analyze, in
line with applicable law and platform terms. The reverse-image-search and
scraping layers (Phase 4) must respect each site's terms of service and rate
limits — the tool does **not** attempt to bypass CAPTCHAs or bot detection.

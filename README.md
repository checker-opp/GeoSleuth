# geolocator — image → location OSINT tool

Guess **where a photo was taken** from the image alone. Feed it a picture; it
runs a layered pipeline of location signals and returns a best-guess location,
a **confidence score**, and the **evidence trail** that produced it — never a
bare guess.

> **Status:** all four phases are built and tested — EXIF → geocoding, OCR,
> GeoCLIP ML estimation, and corroboration + best-effort lookups. See the
> [Roadmap](#roadmap).

---

## Quickstart

```bash
# 1. Clone the repo
git clone <your-repo-url> geolocator
cd geolocator

# 2. (recommended) create a virtual environment — needs Python 3.9+
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install core + the ML engine (GeoCLIP) that locates no-metadata photos
pip install -r requirements.txt -r requirements-ml.txt

# 4. (recommended) install the OCR engine
#    Windows:
winget install UB-Mannheim.TesseractOCR
#    macOS:  brew install tesseract      Debian/Ubuntu:  sudo apt install tesseract-ocr

# 5. Run it
python -m geolocator path/to/photo.jpg
```

> ⏱ **First-time setup ≈ 15–20 min**, almost all of it the one-time `torch`
> install and a **~1.7 GB GeoCLIP model download on the first run**. After that
> it's **~30–90 s per image on CPU** (much less in batch, near-instant on GPU).
> Full breakdown in [Requirements & setup time](#requirements--expected-setup-time).

---

## Demo

Three real runs on photos that have **no GPS metadata** — so the tool guesses
from pixels alone and reports exactly how sure it is, with the evidence trail.
Reproduce any of them: `python -m geolocator docs/demo/<image>`.

### 1 · Landmark — Sydney Opera House *(public-domain photo)*

<img src="docs/demo/sydney.jpg" width="380" alt="Sydney Opera House test photo">

![geolocator run on the Sydney photo](docs/demo/shots/sydney.png)

**✅ Sydney, Australia — city-level, 60%.** GeoCLIP's top-5 predictions clustered
within ~0 km; OSM independently surfaced *Man O' War Steps*, on the Opera House
forecourt.

### 2 · No-metadata street — Varanasi, India

<img src="docs/demo/varanasi.jpg" width="300" alt="Varanasi ghat test photo">

![geolocator run on the Varanasi photo](docs/demo/shots/varanasi.png)

**✅ Varanasi, India — city-level, 65%.** Pinned the Banaras ghats; OSM
corroborated with real neighbouring ghats (Ahilyabai, Manmandir, Munshi…), and a
detected-language hint consistent with India nudged confidence up.

### 3 · Colonial street — Karachi, Pakistan

<img src="docs/demo/karachi.png" width="380" alt="Karachi street test photo">

![geolocator run on the Karachi photo](docs/demo/shots/karachi.png)

**✅ Karachi, Pakistan — country-level, 35%.** The honest case: GeoCLIP's top-5
spread ~1,000 km, so it reports *country*-level rather than faking a pinpoint —
the trustworthy part of the answer is **Pakistan**. English shop signs don't
narrow it further, which is exactly what the reverse-image-search pivots are for.

### Results at a glance

| Photo | Best guess | Truth | Correct? | Confidence |
|-------|-----------|-------|----------|------------|
| Sydney | Sydney, NSW, Australia | Sydney | ✅ city | 60% |
| Varanasi | Varanasi, UP, India | Varanasi | ✅ city | 65% |
| Karachi | Karachi, Sindh, Pakistan | Karachi | ✅ country | 35% |

Every result ships with its **confidence** and **evidence trail** — never a bare pin.

> Screenshots are captured from real runs (`docs/demo/shots/`). Confidence
> honestly tracks how sure the model is: tight prediction clusters read higher,
> a wide spread reads lower.

---

## What it does today

| Signal | What it gives you | Reliability |
|--------|-------------------|-------------|
| **EXIF GPS** → Nominatim | Exact coordinates + full street address | ★★★★★ when GPS is present |
| **EXIF metadata** | Camera, timestamp, timezone-offset longitude hint | context only |
| **OCR** (Tesseract) | Visible text → detected language → candidate countries | ★★☆ coarse hint |
| **ML estimate** (GeoCLIP) | Predicted coordinates from visual content alone — the no-metadata workhorse | ★★★☆ on distinctive scenes, honestly low on generic/indoor |
| **OCR place lookup** | A business/place name OCR reads, geocoded and cross-checked against the visual estimate | ★★★★ when it matches — turns a sign into coordinates |
| **StreetCLIP** (opt-in) | Independent 2nd model's country vote; boosts confidence when it agrees with GeoCLIP | cross-model corroboration (validated: Karachi→Pakistan) |
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

**Prerequisite:** Python **3.9+** (`python --version`).

### 1. Core (always)

```bash
pip install -r requirements.txt
```

This alone gives you the reliable core: EXIF GPS → Nominatim reverse geocoding.
Optionally install it as a `geolocate` command:

```bash
pip install -e .
```

### 2. ML engine — GeoCLIP (the no-metadata workhorse)

Most photos have their GPS stripped; GeoCLIP is what actually locates them from
pixels. It pulls in `torch`, kept in a separate requirements file so the core
stays lightweight:

```bash
pip install -r requirements.txt -r requirements-ml.txt
```

- **torch** — ~200 MB download / ~2 GB installed; the pinned CPU build runs
  anywhere. For GPU, install the CUDA torch build from [pytorch.org](https://pytorch.org) instead.
- **GeoCLIP** downloads its model weights (**~1.7 GB**, CLIP-ViT-L/14) on the
  **first run**, then caches them under `~/.cache/huggingface`.
- `transformers` is pinned `<5` — GeoCLIP's encoder targets the 4.x CLIP API and
  5.x breaks inference.
- Without these extras the ML stage is skipped gracefully (the tool says so).

### 3. External binaries (recommended)

Auto-detected; the tool degrades gracefully if they're missing.

| Tool | Enables | Install |
|------|---------|---------|
| **Tesseract** | OCR (text / language / plate signals) | Windows: `winget install UB-Mannheim.TesseractOCR` · macOS: `brew install tesseract` · Linux: `apt install tesseract-ocr` |
| **ExifTool** | Comprehensive EXIF (RAW/HEIC/XMP) | Windows: download from [exiftool.org](https://exiftool.org), rename `exiftool(-k).exe`→`exiftool.exe` onto `PATH` · macOS: `brew install exiftool` · Linux: `apt install libimage-exiftool-perl` |

> On Windows the tool **auto-detects Tesseract** at `C:\Program Files\Tesseract-OCR`,
> so the `winget` install works with no `PATH` editing. Without Tesseract, OCR is
> skipped; without ExifTool, EXIF still works via the pure-Python `exifread`/Pillow
> fallback.

### Requirements & expected setup time

| Step | Download | Approx time |
|------|----------|-------------|
| Core deps (`requirements.txt`) | ~30 MB | ~1 min |
| ML deps — torch + transformers + geoclip | ~250 MB | ~3–5 min |
| Tesseract binary (optional) | ~50 MB | ~1–2 min |
| **First GeoCLIP run** — downloads CLIP-ViT-L/14 | **~1.7 GB** | ~8–12 min |
| *(optional)* StreetCLIP first run — 2nd model | ~1.6 GB | ~8–12 min |

**Total to first result ≈ 15–20 min** (without StreetCLIP), dominated by the
torch install and the one-time ~1.7 GB model download; figures assume a
~30 Mbps connection. **Steady state: ~30–90 s per image on CPU** — mostly the
model loading from disk — and only ~5–15 s each for later images in a **batch**
(the model loads once). A GPU makes inference near-instant.

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
| `GEOLOCATOR_STREETCLIP` | set to `1` to enable the StreetCLIP second-model cross-check (~1.6 GB one-time download — see note below) |
| `NO_COLOR` | disables ANSI colour in the terminal report |

> **StreetCLIP note:** the cross-check loads `geolocal/StreetCLIP` (~1.6 GB,
> CLIP-ViT-L/14). The **first run downloads the weights (~1.6 GB, one-time)**;
> after that it's practical even on CPU (~27 s to load + ~3 s per image in
> testing). It's **off by default** because of that large download and the
> per-run load cost — enable it when you want a second opinion. Validated on a
> Karachi street scene: StreetCLIP returned *Pakistan* at p≈1.00, agreeing with
> GeoCLIP and nudging confidence up.

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

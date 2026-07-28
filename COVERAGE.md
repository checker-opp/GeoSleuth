# Coverage — spec vs. implementation

Every API, tool, and integration from the project brief and reference material,
mapped to what actually shipped and why. Legend: ✅ integrated & active ·
🟡 integrated in code (optional / not installed here) · 🔄 substituted with a
free equivalent · ❌ not integrated (reason given).

## TRACK 1 — metadata

| Spec item | Status | Notes |
|-----------|--------|-------|
| ExifTool | 🟡 | Auto-detected & preferred if on PATH; not installed in the dev env, so it falls back |
| exifread | ✅ | Pure-Python EXIF path, actively used |
| Pillow | ✅ | Image handling + last-resort EXIF fallback |
| Nominatim (reverse geocode) | ✅ | Rate-limited, UA-compliant; also used for forward (place-name) search |

## TRACK 2 — no metadata

### AI / ML locators
| Spec item | Status | Notes |
|-----------|--------|-------|
| CLIP | ✅ | GeoCLIP & StreetCLIP are CLIP-based |
| GeoEstimation (HuggingFace) | ✅ | Realized as **GeoCLIP** (image → coordinates) |
| Places365 | ❌ | A scene-type classifier, not a geolocator; GeoCLIP already outputs coordinates, so it added nothing |
| GeoSpy AI | 🔄 → ✅ | GeoSpy's free access dried up; replaced by local GeoCLIP/StreetCLIP **and** the **GeoSeer API** (a GeoSpy-style service, free ~10/day) — now the strongest locator |

### OCR / text
| Spec item | Status | Notes |
|-----------|--------|-------|
| Tesseract | ✅ | With preprocessing (upscale/contrast/multi-PSM); active |
| Tesseract.js | 🔄 | Used the Python Tesseract binary — same engine, better fit for a Python CLI |
| langdetect | ✅ | Language → country hint (gated so it only boosts on country agreement) |
| OpenALPR (plates) | 🔄 | Heavy/partly-commercial; replaced with a conservative regex plate-format → country hint |
| Google Vision API | 🔄 | Paid (breaks free-only); covered by Tesseract + GeoCLIP/GeoSeer |
| Google Places | 🔄 | Paid/Google; used OSM/Nominatim search |

### Reverse image search
| Spec item | Status | Notes |
|-----------|--------|-------|
| Yandex / Google Lens / Bing | 🟡 | Ready-to-use **pivot links** (manual upload). Not automated — scraping breaks ToS + needs CAPTCHA-bypass, which we won't do |
| TinEye (free API) | ❌ | Needs a key + HMAC signing we couldn't test; offered as a pivot link instead |
| Google search | ❌ | Skipped per the brief's own note (blocks bots) |
| requests+BeautifulSoup / selenium scraping | ❌ | ToS violation + CAPTCHA; ethical line |

### Street matching
| Spec item | Status | Notes |
|-----------|--------|-------|
| Mapillary API | ✅ | Token-gated, validated live |
| KartaView | ✅ | Keyless fallback, validated live (Europe coverage strong) |

### Sun / shadow
| Spec item | Status | Notes |
|-----------|--------|-------|
| SunCalc / SunCalc.org | 🔄 | JS tool → replaced with **pysolar** (Python) |
| pysolar | 🟡 | Sun-elevation + timezone↔longitude consistency implemented. **Shadow-angle-from-image → latitude NOT built** (needs CV shadow detection) |

### Cross-reference
| Spec item | Status | Notes |
|-----------|--------|-------|
| Overpass API (OSM) | ✅ | Nearby-feature corroboration |
| iNaturalist API | ✅ | Coordinate → nearby-species biome corroboration (keyless). The *image → species* direction needs a plant-ID vision model we don't ship |
| Google Maps (cross-ref) | 🔄 | Paid/Google; used OSM/Overpass |

## Reference layers (the 5-layer doc)

| Layer | Status | How |
|-------|--------|-----|
| 1 · OCR (signs/plates) | ✅ | Tesseract + langdetect + plate regex + OCR→Nominatim place lookup |
| 2 · Landmark/object detection | 🔄 | GeoCLIP/GeoSeer place the scene; **no explicit logo/object labeling** |
| 3 · Shadow analysis | 🟡 | Sun math + timezone check; no shadow-from-image latitude |
| 4 · GeoSpy AI | ✅ | Substituted by GeoSeer API + local GeoCLIP/StreetCLIP |
| 5 · Reverse image search | 🟡 | Pivot links (manual), not automated (ToS) |

## Confidence / correctness engine (beyond the spec)

Added on top of the brief to keep results honest:

- **Locating vs. enrichment** signals — only independent locators raise confidence; corroboration must be geographically consistent.
- **Model-disagreement resolver** — StreetCLIP vs GeoCLIP: re-rank → country-override → down-weight (corrects GeoCLIP's confident continental errors).
- **GeoSeer priority** — the strongest locator wins and is never overridden by weaker models.
- Every result ships a **confidence score** + **evidence trail**.

## Genuine gaps left (all documented, none hidden)

| Gap | Why | Reachable? |
|-----|-----|-----------|
| Shadow-angle → latitude from the image | needs CV shadow detection | yes, future |
| Logo / brand / object labeling | needs a detection model | yes, future |
| Robust plate reading (OpenALPR/CV) | heavy dependency, low payoff | yes, low priority |
| Automated reverse image search | ToS + CAPTCHA | **no — won't do** |

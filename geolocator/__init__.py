"""geolocator — image -> location OSINT tool.

Phase 1 (this release): the reliable core.
  * EXIF GPS extraction  -> exact coordinates when present
  * Nominatim reverse geocoding -> human-readable place
  * OCR (Tesseract) -> visible text + language -> country hint
  * confidence scoring + evidence trail

Later phases (CLIP / GeoEstimation, reverse image search, street match,
shadow/flora cross-reference) slot into the same pipeline.
"""

__version__ = "0.1.0"

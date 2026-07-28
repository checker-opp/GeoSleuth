"""EXIF extraction — Track 1, the reliable core.

Strategy (best available wins, graceful fallback):
  1. ``exiftool`` subprocess if the binary is on PATH — most comprehensive,
     handles RAW / HEIC / XMP / maker notes.
  2. ``exifread`` — pure-Python, no external binary, handles most JPEG/TIFF.
  3. Pillow ``_getexif`` — last-resort fallback.

The goal is a single ``ExifData`` with decimal-degree GPS coordinates (when
present) plus a few useful non-GPS clues (camera model, timestamp, offset).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

from .models import Coordinates


@dataclass
class ExifData:
    coordinates: Optional[Coordinates] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    datetime: Optional[str] = None
    timezone_offset: Optional[str] = None   # e.g. "+05:30" — a coarse longitude hint
    source: str = "none"                     # which extractor produced this
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_gps(self) -> bool:
        return self.coordinates is not None


def exiftool_available() -> bool:
    return shutil.which("exiftool") is not None


# --------------------------------------------------------------------------- #
# Coordinate helpers
# --------------------------------------------------------------------------- #
def _dms_to_decimal(dms: tuple[float, float, float], ref: str) -> float:
    """Convert (degrees, minutes, seconds) + hemisphere ref to signed decimal."""
    deg, minutes, seconds = dms
    decimal = float(deg) + float(minutes) / 60.0 + float(seconds) / 3600.0
    if ref and ref.upper() in ("S", "W"):
        decimal = -decimal
    return decimal


def _valid_coord(lat: float, lon: float) -> bool:
    if lat == 0.0 and lon == 0.0:
        return False  # null island — almost always a zeroed/absent value
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


# --------------------------------------------------------------------------- #
# Extractor 1: exiftool
# --------------------------------------------------------------------------- #
def _extract_with_exiftool(path: str) -> Optional[ExifData]:
    try:
        proc = subprocess.run(
            ["exiftool", "-json", "-n", "-c", "%.8f", path],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        tags = json.loads(proc.stdout)[0]
    except (json.JSONDecodeError, IndexError, KeyError):
        return None

    data = ExifData(source="exiftool", raw=tags)
    # With -n, exiftool emits GPSLatitude/GPSLongitude as signed decimals.
    lat = tags.get("GPSLatitude")
    lon = tags.get("GPSLongitude")
    if lat is not None and lon is not None:
        try:
            lat, lon = float(lat), float(lon)
            if _valid_coord(lat, lon):
                data.coordinates = Coordinates(lat, lon)
        except (TypeError, ValueError):
            pass

    data.camera_make = tags.get("Make")
    data.camera_model = tags.get("Model")
    data.datetime = tags.get("DateTimeOriginal") or tags.get("CreateDate")
    data.timezone_offset = tags.get("OffsetTimeOriginal") or tags.get("OffsetTime")
    return data


# --------------------------------------------------------------------------- #
# Extractor 2: exifread
# --------------------------------------------------------------------------- #
def _ratios_to_floats(values) -> Optional[tuple[float, float, float]]:
    try:
        nums = [float(v.num) / float(v.den) for v in values]
        if len(nums) == 3:
            return (nums[0], nums[1], nums[2])
    except (AttributeError, ZeroDivisionError, TypeError):
        return None
    return None


def _extract_with_exifread(path: str) -> Optional[ExifData]:
    try:
        import exifread
    except ImportError:
        return None
    try:
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, details=False)
    except (OSError, Exception):  # exifread can raise assorted parse errors
        return None
    if not tags:
        return None

    data = ExifData(source="exifread", raw={k: str(v) for k, v in tags.items()})

    lat_tag = tags.get("GPS GPSLatitude")
    lon_tag = tags.get("GPS GPSLongitude")
    lat_ref = str(tags.get("GPS GPSLatitudeRef", "")).strip()
    lon_ref = str(tags.get("GPS GPSLongitudeRef", "")).strip()
    if lat_tag is not None and lon_tag is not None:
        lat_dms = _ratios_to_floats(lat_tag.values)
        lon_dms = _ratios_to_floats(lon_tag.values)
        if lat_dms and lon_dms:
            lat = _dms_to_decimal(lat_dms, lat_ref or "N")
            lon = _dms_to_decimal(lon_dms, lon_ref or "E")
            if _valid_coord(lat, lon):
                data.coordinates = Coordinates(lat, lon)

    if "Image Make" in tags:
        data.camera_make = str(tags["Image Make"]).strip()
    if "Image Model" in tags:
        data.camera_model = str(tags["Image Model"]).strip()
    for key in ("EXIF DateTimeOriginal", "Image DateTime"):
        if key in tags:
            data.datetime = str(tags[key]).strip()
            break
    if "EXIF OffsetTimeOriginal" in tags:
        data.timezone_offset = str(tags["EXIF OffsetTimeOriginal"]).strip()
    return data


# --------------------------------------------------------------------------- #
# Extractor 3: Pillow
# --------------------------------------------------------------------------- #
def _extract_with_pillow(path: str) -> Optional[ExifData]:
    try:
        from PIL import Image, ExifTags
    except ImportError:
        return None
    try:
        img = Image.open(path)
        exif = img._getexif()  # type: ignore[attr-defined]
    except (OSError, AttributeError, Exception):
        return None
    if not exif:
        return None

    tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    data = ExifData(source="pillow", raw={str(k): str(v) for k, v in tag_map.items()})

    gps_info = tag_map.get("GPSInfo")
    if gps_info:
        gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
        lat_tuple = gps.get("GPSLatitude")
        lon_tuple = gps.get("GPSLongitude")
        lat_ref = gps.get("GPSLatitudeRef", "N")
        lon_ref = gps.get("GPSLongitudeRef", "E")
        if lat_tuple and lon_tuple:
            try:
                lat = _dms_to_decimal(tuple(float(x) for x in lat_tuple), lat_ref)
                lon = _dms_to_decimal(tuple(float(x) for x in lon_tuple), lon_ref)
                if _valid_coord(lat, lon):
                    data.coordinates = Coordinates(lat, lon)
            except (TypeError, ValueError):
                pass

    data.camera_make = tag_map.get("Make")
    data.camera_model = tag_map.get("Model")
    data.datetime = tag_map.get("DateTimeOriginal") or tag_map.get("DateTime")
    data.timezone_offset = tag_map.get("OffsetTimeOriginal")
    return data


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def extract(path: str) -> ExifData:
    """Extract EXIF using the best available method, falling back gracefully.

    Prefers whichever extractor actually finds GPS; otherwise returns the
    richest metadata we managed to read.
    """
    candidates: list[ExifData] = []

    if exiftool_available():
        result = _extract_with_exiftool(path)
        if result:
            if result.has_gps:
                return result
            candidates.append(result)

    for extractor in (_extract_with_exifread, _extract_with_pillow):
        result = extractor(path)
        if result:
            if result.has_gps:
                return result
            candidates.append(result)

    # No GPS anywhere — return the metadata-richest attempt (or an empty shell).
    if candidates:
        return max(candidates, key=lambda d: len(d.raw))
    return ExifData(source="none")

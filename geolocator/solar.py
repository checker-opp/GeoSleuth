"""Solar / sun-position corroboration — Phase 3.

Two cheap, honest cross-checks that *corroborate* an existing candidate
location rather than produce one on their own:

  1. **Timezone → longitude consistency.** If EXIF carries a UTC offset
     (e.g. ``+10:00``), that implies a longitude band (~15° per hour). We
     compare it against the candidate longitude — agreement is mild
     corroboration, a gross mismatch is a red flag.

  2. **Sun elevation plausibility.** Given the candidate coordinates and the
     EXIF capture time, ``pysolar`` tells us where the sun was. This yields a
     day/night descriptor — useful context, and a sanity check against obvious
     nonsense.

True shadow-angle → latitude estimation needs computer-vision shadow detection
in the image itself; that's a later addition. What's here works from metadata
we already have. ``pysolar`` is an optional dependency — everything degrades
gracefully if it's absent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from .models import Coordinates

# EXIF datetimes look like "2024:06:15 14:30:00".
_EXIF_DT_RE = re.compile(r"^\s*(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})")
# UTC offsets look like "+05:30", "-08:00", "+00:00".
_OFFSET_RE = re.compile(r"^\s*([+-])(\d{2}):?(\d{2})\s*$")


@dataclass
class SolarInfo:
    sun_elevation: Optional[float] = None   # degrees above horizon (neg = night)
    sun_azimuth: Optional[float] = None
    is_daytime: Optional[bool] = None
    longitude_from_offset: Optional[float] = None  # centre of the tz longitude band
    longitude_consistent: Optional[bool] = None    # offset band vs candidate lon
    longitude_delta_deg: Optional[float] = None
    reason: Optional[str] = None


def parse_exif_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    m = _EXIF_DT_RE.match(value)
    if not m:
        return None
    try:
        y, mo, d, h, mi, s = (int(x) for x in m.groups())
        return datetime(y, mo, d, h, mi, s)
    except ValueError:
        return None


def parse_utc_offset(value: Optional[str]) -> Optional[timedelta]:
    if not value:
        return None
    m = _OFFSET_RE.match(value)
    if not m:
        return None
    sign, hh, mm = m.group(1), int(m.group(2)), int(m.group(3))
    delta = timedelta(hours=hh, minutes=mm)
    return -delta if sign == "-" else delta


def longitude_from_offset(offset: timedelta) -> float:
    """Central meridian for a UTC offset: 15° of longitude per hour."""
    hours = offset.total_seconds() / 3600.0
    return hours * 15.0


def _lon_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two longitudes, degrees (0–180)."""
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def analyze(
    coord: Coordinates,
    exif_datetime: Optional[str],
    utc_offset: Optional[str],
) -> SolarInfo:
    """Compute solar corroboration for a candidate location."""
    info = SolarInfo()

    # --- Timezone -> longitude cross-check (works without pysolar) ---
    offset = parse_utc_offset(utc_offset)
    if offset is not None:
        band_lon = longitude_from_offset(offset)
        info.longitude_from_offset = band_lon
        # Daylight-saving always shifts the clock +1h (+15° to the nominal band)
        # in summer, so the true solar meridian is at the raw band OR one hour
        # back. Take the more forgiving of the two to avoid DST false alarms.
        delta = min(_lon_diff(band_lon, coord.lon), _lon_diff(band_lon - 15.0, coord.lon))
        info.longitude_delta_deg = round(delta, 1)
        # A timezone spans ~15°, plus political width — allow ~22.5° slack.
        info.longitude_consistent = delta <= 22.5

    # --- Sun elevation (needs pysolar + a timestamp) ---
    dt = parse_exif_datetime(exif_datetime)
    if dt is None:
        info.reason = "no usable EXIF timestamp for sun-position check"
        return info

    try:
        from pysolar import solar as pysolar_solar
    except ImportError:
        info.reason = "pysolar not installed (sun-elevation check skipped)"
        return info

    # Interpret the timestamp in its local offset if known, else assume UTC.
    aware = dt.replace(tzinfo=timezone(offset) if offset is not None else timezone.utc)
    try:
        elevation = pysolar_solar.get_altitude(coord.lat, coord.lon, aware)
        azimuth = pysolar_solar.get_azimuth(coord.lat, coord.lon, aware)
    except Exception as exc:  # pysolar can raise on odd inputs
        info.reason = f"sun-position computation failed: {exc}"
        return info

    info.sun_elevation = round(elevation, 1)
    info.sun_azimuth = round(azimuth, 1)
    info.is_daytime = elevation > -0.833  # standard sunrise/sunset refraction
    return info

"""Command-line interface for the geolocator.

Usage:
    python -m geolocator IMAGE [--json] [--verbose]
    geolocate IMAGE [--json] [--verbose]      # if installed as a script
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import (
    __version__,
    exif as exif_mod,
    geoestimate as geoestimate_mod,
    geoseer as geoseer_mod,
    ocr as ocr_mod,
    street_match as street_match_mod,
)
from .models import GeoResult, Precision
from .pipeline import analyze


def _prepare_stdout() -> bool:
    """Make stdout tolerant of Unicode on legacy consoles (e.g. Windows cp1252).

    Returns True if the stream can carry our fancy glyphs, False if we should
    fall back to ASCII. Never raises.
    """
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    if "utf" in enc:
        return True
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        return True
    except Exception:
        return False


_UNICODE_OK = _prepare_stdout()

# Glyph sets — Unicode when the terminal supports it, ASCII otherwise.
_G = {
    "arrow": "→" if _UNICODE_OK else "->",
    "full": "█" if _UNICODE_OK else "#",
    "empty": "░" if _UNICODE_OK else "-",
    "bullet": "•" if _UNICODE_OK else "*",
    "gear": "⚙" if _UNICODE_OK else "~",
    "ellipsis": "…" if _UNICODE_OK else "...",
}

# --- tiny ANSI helpers (auto-disabled when not a TTY or NO_COLOR is set) ----
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(t: str) -> str:
    return _c(t, "1")


def _dim(t: str) -> str:
    return _c(t, "2")


def _confidence_bar(conf: float) -> str:
    filled = round(conf * 10)
    bar = _G["full"] * filled + _G["empty"] * (10 - filled)
    pct = f"{conf * 100:4.0f}%"
    if conf >= 0.8:
        color = "32"  # green
    elif conf >= 0.4:
        color = "33"  # yellow
    else:
        color = "31"  # red
    return _c(f"{bar} {pct}", color)


_PRECISION_LABEL = {
    Precision.EXACT: "exact coordinates",
    Precision.CITY: "city-level",
    Precision.REGION: "region-level",
    Precision.COUNTRY: "country-level",
    Precision.UNKNOWN: "no geographic fix",
}


def _print_human(result: GeoResult) -> None:
    print()
    print(_bold(f"  IMAGE {_G['arrow']} LOCATION  ") + _dim(f"(geolocator v{__version__})"))
    print(_dim(f"  {result.image_path}"))
    print()

    # Best guess block
    print(_bold("  BEST GUESS"))
    if result.best_place or result.best_coordinates:
        if result.best_place:
            print(f"    Place       {result.best_place}")
        if result.best_coordinates:
            c = result.best_coordinates
            print(f"    Coordinates {c.lat:.6f}, {c.lon:.6f}")
            print(_dim(f"    Map         https://www.openstreetmap.org/?mlat={c.lat}&mlon={c.lon}#map=16/{c.lat}/{c.lon}"))
        print(f"    Precision   {_PRECISION_LABEL.get(result.best_precision, '—')}")
    else:
        print("    " + _dim("No location could be determined from available signals."))
    print(f"    Confidence  {_confidence_bar(result.overall_confidence)}")
    print()

    # Evidence trail
    if result.signals:
        print(_bold("  EVIDENCE"))
        for s in result.signals:
            head = f"    [{s.source}] {s.description}"
            print(head)
            print(_dim(f"          confidence {s.confidence:.2f} · {s.precision.value}"))
        print()

    # Notes
    if result.notes:
        print(_bold("  NOTES"))
        for n in result.notes:
            print(_dim(f"    {_G['bullet']} {n}"))
        print()

    # Manual reverse-image-search pivots (actionable next steps)
    pivots = result.meta.get("pivots") if hasattr(result, "meta") else None
    if pivots:
        print(_bold("  REVERSE-IMAGE-SEARCH PIVOTS") + _dim("  (upload the photo manually)"))
        for p in pivots:
            print(f"    {p['engine']:<18} {p['url']}")
            print(_dim(f"          {p['note']}"))
        print()

    # Tooling status footer — makes missing-binary situations obvious.
    tools = []
    tools.append(("exiftool", exif_mod.exiftool_available()))
    tools.append(("tesseract (OCR)", ocr_mod.tesseract_available()))
    tools.append(("ML geo-estimation (GeoCLIP)", geoestimate_mod.ml_available()))
    tools.append(("GeoSeer AI locator (GEOSEER_API_KEY)", geoseer_mod.api_key_configured()))
    tools.append(("Mapillary street match (MAPILLARY_TOKEN)", street_match_mod.token_configured()))
    missing = [name for name, ok in tools if not ok]
    if missing:
        print(_dim(f"  {_G['gear']} optional tools not installed: " + ", ".join(missing)))
        print(_dim("    install them to unlock more signals (see README)."))
        print()


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}


def _expand_inputs(paths: list[str]) -> tuple[list[str], list[str]]:
    """Expand paths (files and/or directories) into a de-duplicated image list.

    Returns (image_paths, errors). Directories are scanned (non-recursively) for
    known image extensions, sorted for stable output."""
    images: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()

    def add(p: str) -> None:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            images.append(p)

    for path in paths:
        if os.path.isdir(path):
            found = sorted(
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
            )
            if not found:
                errors.append(f"no images found in directory: {path}")
            for f in found:
                add(f)
        elif os.path.isfile(path):
            add(path)
        else:
            errors.append(f"file not found: {path}")
    return images, errors


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="geolocate",
        description="Guess where a photo was taken (EXIF + OCR + ML + reverse geocoding).",
    )
    p.add_argument(
        "images",
        nargs="+",
        help="one or more image files, and/or a directory of images (batch)",
    )
    p.add_argument("--json", action="store_true", help="output machine-readable JSON")
    p.add_argument(
        "--verbose", action="store_true", help="include raw signal evidence in JSON"
    )
    p.add_argument("--version", action="version", version=f"geolocator {__version__}")
    return p


def _result_json(result: GeoResult, verbose: bool) -> dict:
    payload = result.to_dict()
    if not verbose:
        for sig in payload["signals"]:
            sig.pop("evidence", None)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    images, errors = _expand_inputs(args.images)
    for err in errors:
        print(f"error: {err}", file=sys.stderr)
    if not images:
        return 2

    # Analyze each image. The GeoCLIP model is cached after the first load, so a
    # batch reuses it rather than reloading per image.
    results = [analyze(img) for img in images]

    if args.json:
        if len(results) == 1:
            print(json.dumps(_result_json(results[0], args.verbose), indent=2, ensure_ascii=False))
        else:
            payload = [_result_json(r, args.verbose) for r in results]
            print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for i, result in enumerate(results):
            if len(results) > 1:
                print(_dim(f"\n{'═' * 60}") if _UNICODE_OK else _dim("\n" + "=" * 60))
                print(_dim(f"  [{i + 1}/{len(results)}]"))
            _print_human(result)

    # Exit non-zero only if nothing produced a location at all.
    return 0 if any(r.best_coordinates or r.best_place for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

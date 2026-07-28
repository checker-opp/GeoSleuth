"""OCR text extraction + language hinting — Track 2, cheapest visual signal.

A single readable shop name, street sign, or license plate often narrows
location more than any ML model. Even when we can't identify the *text*, the
*language / script* it's written in is a coarse country hint.

Requires the Tesseract binary (``tesseract`` on PATH) plus ``pytesseract``.
If either is missing this module degrades gracefully — the pipeline simply
records that OCR was unavailable rather than crashing.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Optional

# Coarse "language -> countries where it's an official / dominant language".
# This is intentionally a *hint*, never a determination — many languages span
# many countries. Used only to nudge confidence, always surfaced as evidence.
LANG_TO_COUNTRIES: dict[str, list[str]] = {
    "en": ["US", "UK", "Canada", "Australia", "India", "Ireland", "NZ"],
    "es": ["Spain", "Mexico", "Argentina", "Colombia", "Chile", "Peru"],
    "fr": ["France", "Canada (Québec)", "Belgium", "Switzerland", "West Africa"],
    "de": ["Germany", "Austria", "Switzerland"],
    "it": ["Italy", "Switzerland", "San Marino"],
    "pt": ["Portugal", "Brazil", "Angola", "Mozambique"],
    "nl": ["Netherlands", "Belgium"],
    "ru": ["Russia", "Belarus", "Kazakhstan", "Kyrgyzstan"],
    "uk": ["Ukraine"],
    "pl": ["Poland"],
    "tr": ["Turkey", "Cyprus"],
    "ar": ["Saudi Arabia", "Egypt", "UAE", "Iraq", "Morocco", "MENA region"],
    "he": ["Israel"],
    "el": ["Greece", "Cyprus"],
    "ja": ["Japan"],
    "ko": ["South Korea", "North Korea"],
    "zh-cn": ["China", "Singapore"],
    "zh-tw": ["Taiwan", "Hong Kong"],
    "th": ["Thailand"],
    "vi": ["Vietnam"],
    "hi": ["India"],
    "bn": ["Bangladesh", "India (West Bengal)"],
    "ta": ["India (Tamil Nadu)", "Sri Lanka", "Singapore"],
    "id": ["Indonesia"],
    "ms": ["Malaysia", "Brunei"],
    "sv": ["Sweden"],
    "no": ["Norway"],
    "da": ["Denmark"],
    "fi": ["Finland"],
    "cs": ["Czechia"],
    "hu": ["Hungary"],
    "ro": ["Romania", "Moldova"],
}


@dataclass
class OcrResult:
    available: bool                       # was OCR actually run?
    text: str = ""
    language: Optional[str] = None        # langdetect code, e.g. "fr"
    language_countries: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)  # candidate name lines
    reason: Optional[str] = None          # why unavailable, if applicable

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


# Standard install locations to probe when Tesseract isn't on PATH — common on
# Windows, where the UB-Mannheim installer doesn't always update the current
# shell's PATH.
_COMMON_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
]


def _resolve_tesseract() -> Optional[str]:
    """Find the Tesseract binary via PATH or a known install location, and point
    pytesseract at it if it's off-PATH. Returns the resolved path, or None."""
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    for candidate in _COMMON_TESSERACT_PATHS:
        if candidate and os.path.isfile(candidate):
            try:
                import pytesseract

                pytesseract.pytesseract.tesseract_cmd = candidate
            except Exception:
                pass
            return candidate
    # Last resort: a pytesseract-configured path that actually resolves.
    try:
        import pytesseract

        cmd = pytesseract.pytesseract.tesseract_cmd
        if os.path.isfile(cmd) or shutil.which(cmd):
            return cmd
    except Exception:
        pass
    return None


def tesseract_available() -> bool:
    return _resolve_tesseract() is not None


# Upscale target for OCR — signage is often small in a wide street photo, and
# Tesseract does markedly better when the text is large enough.
_OCR_TARGET_LONG_EDGE = 1800


def _preprocess(image):
    """Grayscale, upscale-if-small, and autocontrast to boost OCR recall."""
    from PIL import Image, ImageOps

    g = ImageOps.grayscale(image)
    w, h = g.size
    longest = max(w, h)
    if longest < _OCR_TARGET_LONG_EDGE:
        scale = _OCR_TARGET_LONG_EDGE / float(longest)
        g = g.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return ImageOps.autocontrast(g)


def _ocr_text(image, config: str) -> str:
    import pytesseract

    try:
        return pytesseract.image_to_string(image, config=config).strip()
    except Exception:
        return ""


def _candidate_lines(text: str) -> list[str]:
    """Pull out plausible business / place-name lines from raw OCR output.

    Keeps lines with real letters and at least a few characters, drops obvious
    noise. These feed the Nominatim place-name lookup stage."""
    import re

    seen: set[str] = set()
    out: list[str] = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if len(line) < 4:
            continue
        letters = sum(c.isalpha() for c in line)
        if letters < 3 or letters < len(line) * 0.4:  # mostly symbols/digits
            continue
        # collapse repeated punctuation, strip leading/trailing junk
        line = re.sub(r"[^\w&.,'\- ]+", " ", line).strip()
        line = " ".join(line.split())
        # require at least one "real word" token to cut OCR gibberish like "bf x8"
        if not any(len(tok) >= 4 and tok.isalpha() for tok in line.split()):
            continue
        key = line.lower()
        if len(line) >= 4 and key not in seen:
            seen.add(key)
            out.append(line)
    return out


def run(path: str) -> OcrResult:
    """Extract visible text and infer its language. Never raises on the normal
    'tools not installed' path.

    Runs OCR on a preprocessed (grayscale, upscaled, contrast-boosted) image
    using two page-segmentation modes — the default block mode and sparse-text
    mode (good for scattered signage) — and merges the results.
    """
    try:
        import pytesseract  # noqa: F401
        from PIL import Image
    except ImportError as exc:
        return OcrResult(available=False, reason=f"missing python dependency: {exc.name}")

    if not _resolve_tesseract():
        return OcrResult(
            available=False,
            reason="Tesseract binary not found on PATH (install it to enable OCR)",
        )

    try:
        image = Image.open(path)
        prepped = _preprocess(image)
    except Exception as exc:
        return OcrResult(available=False, reason=f"OCR failed to open/preprocess: {exc}")

    # Two passes: default block segmentation + sparse-text (psm 11) for signage.
    block = _ocr_text(prepped, "--psm 3")
    sparse = _ocr_text(prepped, "--psm 11")

    # Merge unique lines across both passes, preserving order.
    merged_lines: list[str] = []
    seen: set[str] = set()
    for chunk in (block, sparse):
        for ln in chunk.splitlines():
            k = " ".join(ln.split()).lower()
            if k and k not in seen:
                seen.add(k)
                merged_lines.append(ln)
    text = "\n".join(merged_lines).strip()

    result = OcrResult(available=True, text=text)
    if not result.has_text:
        return result

    result.lines = _candidate_lines(text)

    # Language detection needs a bit of signal; skip on tiny/garbage output.
    cleaned = " ".join(result.text.split())
    if len(cleaned) >= 8:
        try:
            from langdetect import detect, DetectorFactory

            DetectorFactory.seed = 0  # deterministic output
            lang = detect(cleaned)
            result.language = lang
            result.language_countries = LANG_TO_COUNTRIES.get(lang, [])
        except Exception:
            pass  # detection is best-effort

    return result

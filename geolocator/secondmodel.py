"""Second-opinion geo-model: StreetCLIP zero-shot country classification.

GeoCLIP predicts coordinates; StreetCLIP (``geolocal/StreetCLIP``) is a CLIP
model tuned for *country* recognition from street imagery. Running it as an
independent second opinion gives us genuine corroboration: when two models
trained differently agree on the country, that agreement is real evidence and
legitimately raises confidence — unlike enrichment signals, which fire for any
place.

Zero-shot: we score the image against "A Street View photo in <country>" for a
list of candidate countries and take the best. Heavy and optional — lazy-loaded,
graceful skip if the ML extras or weights are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# A broad candidate set spanning regions. Not exhaustive — the caller can inject
# extra countries (e.g. the GeoCLIP guess) so the comparison is always fair.
CANDIDATE_COUNTRIES = [
    "United States", "Canada", "Mexico", "Brazil", "Argentina", "Chile", "Peru",
    "Colombia", "United Kingdom", "Ireland", "France", "Spain", "Portugal",
    "Italy", "Germany", "Netherlands", "Belgium", "Switzerland", "Austria",
    "Poland", "Czechia", "Hungary", "Romania", "Greece", "Sweden", "Norway",
    "Finland", "Denmark", "Russia", "Ukraine", "Turkey", "Israel", "Egypt",
    "Morocco", "Nigeria", "Kenya", "South Africa", "Ethiopia", "Saudi Arabia",
    "United Arab Emirates", "Iran", "Iraq", "Pakistan", "India", "Bangladesh",
    "Sri Lanka", "Nepal", "China", "Japan", "South Korea", "Taiwan", "Thailand",
    "Vietnam", "Cambodia", "Malaysia", "Singapore", "Indonesia", "Philippines",
    "Australia", "New Zealand",
]

_model = None
_processor = None
_load_error: Optional[str] = None

_MODEL_ID = "geolocal/StreetCLIP"


@dataclass
class CountryPrediction:
    country: str
    score: float                 # softmax probability of the top country
    runner_up: Optional[str] = None
    runner_up_score: float = 0.0


def ml2_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


def _load():
    global _model, _processor, _load_error
    if _model is not None or _load_error is not None:
        return _model, _processor
    try:
        from transformers import CLIPModel, CLIPProcessor

        _model = CLIPModel.from_pretrained(_MODEL_ID)
        _processor = CLIPProcessor.from_pretrained(_MODEL_ID)
    except Exception as exc:
        _load_error = f"failed to load StreetCLIP: {exc}"
        _model = _processor = None
    return _model, _processor


def predict_country(
    path: str, extra_countries: Optional[list[str]] = None
) -> Optional[CountryPrediction]:
    """Zero-shot country classification. Returns None if unavailable."""
    if not ml2_available():
        return None
    model, processor = _load()
    if model is None:
        return None

    countries = list(CANDIDATE_COUNTRIES)
    for c in extra_countries or []:
        if c and c not in countries:
            countries.append(c)

    try:
        import torch
        from PIL import Image

        prompts = [f"A Street View photo in {c}." for c in countries]
        inputs = processor(
            text=prompts, images=Image.open(path).convert("RGB"),
            return_tensors="pt", padding=True,
        )
        with torch.no_grad():
            logits = model(**inputs).logits_per_image.softmax(dim=1)[0]
        order = torch.argsort(logits, descending=True)
        top = int(order[0])
        second = int(order[1]) if len(order) > 1 else top
        return CountryPrediction(
            country=countries[top],
            score=float(logits[top]),
            runner_up=countries[second],
            runner_up_score=float(logits[second]),
        )
    except Exception:
        return None

"""Local model management for the web UI — status + download-with-progress.

Lets the browser UI show whether each local CLIP model is installed/cached,
trigger a weights download in the background, and report byte-level progress so
the UI can render a progress bar. Downloading weights needs the ``clip`` extras
(torch/transformers) installed; the pip install of those packages is a separate
prerequisite the UI surfaces as a hint (we don't pip-install into a live server).
"""

from __future__ import annotations

import importlib.util
import threading
from typing import Optional

MODELS = {
    "geoclip": {
        "label": "GeoCLIP — image → coordinates",
        "repo": "openai/clip-vit-large-patch14",   # backbone GeoCLIP loads
        "approx_gb": 1.7,
        "packages": ["torch", "geoclip", "transformers"],
    },
    "streetclip": {
        "label": "StreetCLIP — country cross-check",
        "repo": "geolocal/StreetCLIP",
        "approx_gb": 1.6,
        "packages": ["torch", "transformers"],
    },
}

_progress: dict[str, dict] = {}
_lock = threading.Lock()


def _packages_installed(model: str) -> bool:
    return all(importlib.util.find_spec(p) is not None for p in MODELS[model]["packages"])


def _weights_cached(model: str) -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(MODELS[model]["repo"], local_files_only=True)
        return True
    except Exception:
        return False


def status(model: str) -> dict:
    with _lock:
        prog = dict(_progress.get(model, {}))
    installed = _packages_installed(model)
    cached = _weights_cached(model) if installed else False

    if prog.get("state") == "downloading":
        state = "downloading"
    elif prog.get("state") == "error":
        state = "error"
    elif cached:
        state = "ready"
    elif not installed:
        state = "needs_packages"
    else:
        state = "not_downloaded"

    return {
        "model": model,
        "label": MODELS[model]["label"],
        "approx_gb": MODELS[model]["approx_gb"],
        "packages_installed": installed,
        "weights_cached": cached,
        "state": state,
        "percent": prog.get("percent", 100 if cached else 0),
        "downloaded_mb": prog.get("downloaded_mb", 0),
        "total_mb": prog.get("total_mb", 0),
        "error": prog.get("error"),
        "install_hint": None if installed else "pip install .[clip]",
    }


def all_status() -> dict:
    return {m: status(m) for m in MODELS}


def _download_worker(model: str) -> None:
    repo = MODELS[model]["repo"]
    try:
        from huggingface_hub import HfApi, snapshot_download

        total = 0
        try:
            info = HfApi().model_info(repo, files_metadata=True)
            total = sum((s.size or 0) for s in (info.siblings or []))
        except Exception:
            total = int(MODELS[model]["approx_gb"] * 1e9)

        with _lock:
            _progress[model] = {"state": "downloading", "percent": 0,
                                "downloaded_mb": 0, "total_mb": round(total / 1e6)}

        tqdm_kwargs = {}
        try:
            from tqdm.auto import tqdm as base_tqdm

            counter = {"done": 0, "active": {}}
            clock = threading.Lock()
            total_bytes = total or 1

            class _UITqdm(base_tqdm):
                def update(self, n=1):
                    super().update(n)
                    with clock:
                        counter["active"][id(self)] = self.n
                        dl = counter["done"] + sum(counter["active"].values())
                    with _lock:
                        _progress[model].update(
                            {"percent": round(min(99.9, dl / total_bytes * 100), 1),
                             "downloaded_mb": round(dl / 1e6)})

                def close(self):
                    with clock:
                        counter["done"] += counter["active"].pop(id(self), 0)
                    super().close()

            tqdm_kwargs["tqdm_class"] = _UITqdm
        except Exception:
            pass  # progress bar is best-effort; download still runs

        snapshot_download(repo, **tqdm_kwargs)
        with _lock:
            _progress[model] = {"state": "ready", "percent": 100,
                                "downloaded_mb": round(total / 1e6),
                                "total_mb": round(total / 1e6)}
    except Exception as exc:
        with _lock:
            _progress[model] = {"state": "error", "error": str(exc)[:300]}


def start_download(model: str) -> dict:
    if model not in MODELS:
        return {"ok": False, "error": "unknown model"}
    if not _packages_installed(model):
        return {"ok": False, "error": "CLIP packages not installed",
                "hint": "pip install .[clip]"}
    with _lock:
        if _progress.get(model, {}).get("state") == "downloading":
            return {"ok": True, "already": True}
        _progress[model] = {"state": "downloading", "percent": 0,
                            "downloaded_mb": 0, "total_mb": 0}
    threading.Thread(target=_download_worker, args=(model,), daemon=True).start()
    return {"ok": True}

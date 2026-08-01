"""Local web UI for the geolocator.

A single-page Flask app: upload an image, paste API keys, tick which signals to
run (including a "GeoSeer-only" mode that skips the heavy local CLIP model), and
see the best guess + confidence + evidence trail.

Run:  python -m geolocator.webui   (then open http://127.0.0.1:5000)

Keys entered in the form are used only for that request — never stored on disk.
Binds to localhost only.
"""

from __future__ import annotations

import os
import tempfile

from .models import AnalyzeConfig, Precision
from .pipeline import analyze
from . import modelmgr

_PRECISION_LABEL = {
    "exact": "exact coordinates",
    "city": "city-level",
    "region": "region-level",
    "country": "country-level",
    "unknown": "no geographic fix",
}

PAGE = """
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>geolocator — image → location</title>
<style>
  :root { color-scheme: dark; }
  body { background:#0d1117; color:#c9d1d9; font-family:-apple-system,Segoe UI,Roboto,sans-serif;
         margin:0; padding:0 16px 48px; }
  .wrap { max-width:920px; margin:0 auto; }
  h1 { font-size:22px; margin:24px 0 4px; } h1 span { color:#7ee787; }
  .sub { color:#8b949e; margin:0 0 20px; font-size:14px; }
  form { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:18px; }
  label { display:block; font-size:13px; color:#8b949e; margin:12px 0 4px; }
  input[type=text], input[type=password], input[type=file] {
    width:100%; box-sizing:border-box; background:#0d1117; border:1px solid #30363d;
    color:#e6edf3; border-radius:6px; padding:9px; font-size:14px; }
  .toggles { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:6px 16px; margin-top:8px; }
  .toggles label { display:flex; align-items:center; gap:8px; color:#c9d1d9; margin:4px 0; font-size:14px; }
  .toggles input { width:auto; }
  .row { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  button { margin-top:18px; background:#238636; color:#fff; border:0; border-radius:6px;
           padding:11px 20px; font-size:15px; font-weight:600; cursor:pointer; }
  button:hover { background:#2ea043; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:18px; margin-top:20px; }
  .place { font-size:19px; color:#e6edf3; font-weight:600; }
  .muted { color:#8b949e; font-size:13px; }
  .bar { height:14px; background:#0d1117; border-radius:7px; overflow:hidden; border:1px solid #30363d; margin:8px 0; }
  .bar > div { height:100%; }
  table { width:100%; border-collapse:collapse; font-size:13px; margin-top:6px; }
  td { padding:6px 8px; border-bottom:1px solid #21262d; vertical-align:top; }
  .tag { color:#79c0ff; font-family:ui-monospace,monospace; white-space:nowrap; }
  a { color:#79c0ff; } .err { color:#ff7b72; }
  .pill { display:inline-block; background:#30363d; border-radius:20px; padding:2px 10px; font-size:12px; color:#c9d1d9; }
  h3 { margin:18px 0 6px; font-size:15px; }
  fieldset { border:1px solid #30363d; border-radius:8px; margin:14px 0 0; padding:8px 14px 14px; }
  legend { color:#8b949e; font-size:12px; padding:0 6px; }
  .overlay { position:fixed; inset:0; background:rgba(13,17,23,.88); display:none;
             flex-direction:column; align-items:center; justify-content:center; z-index:100; }
  .overlay.show { display:flex; }
  .spinner { width:48px; height:48px; border:4px solid #30363d; border-top-color:#2ea043;
             border-radius:50%; animation:spin .8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .overlay p { color:#e6edf3; margin:16px 0 0; font-size:16px; font-weight:600; }
  .overlay .sub { color:#8b949e; font-size:13px; margin-top:6px; }
</style></head><body><div class="wrap">
  <div class="overlay" id="loadingOverlay">
    <div class="spinner"></div>
    <p>Analyzing…</p>
    <div class="sub">local models can take ~30–60 s on the first run</div>
  </div>
  <h1><span>geo</span>locator — image → location</h1>
  <p class="sub">Upload a photo, choose your signals, and get a best guess with confidence + evidence.</p>

  <form id="locateForm" method="post" action="/analyze" enctype="multipart/form-data">
    <label>Image</label>
    <input type="file" name="image" accept="image/*" required>

    <div class="row">
      <div><label>GeoSeer API key <span class="muted">(optional, ~10/day)</span></label>
        <input type="password" name="geoseer_key" value="{{ keys.geoseer_key }}" placeholder="gsk_..."></div>
      <div><label>Mapillary token <span class="muted">(optional)</span></label>
        <input type="password" name="mapillary_token" value="{{ keys.mapillary_token }}" placeholder="MLY|..."></div>
    </div>
    <label>SerpAPI key <span class="muted">(optional — automated reverse image search, ~100/mo)</span></label>
    <input type="password" name="serpapi_key" value="{{ keys.serpapi_key }}" placeholder="serpapi key">
    <div class="muted" style="margin-top:4px">Note: enabling this briefly uploads the image to a temporary public host so Google Lens can fetch it.</div>

    <fieldset><legend>Signals to run</legend>
      <label style="color:#e6edf3"><input type="checkbox" name="geoseer_only" id="geoseer_only" {% if checks.geoseer_only %}checked{% endif %}>
        <b>GeoSeer-only</b> — skip the local CLIP models (fast, API-based)</label>
      <label><input type="checkbox" name="full_workflow" {% if checks.full_workflow %}checked{% endif %}>
        Full workflow — run GeoCLIP even if GeoSeer already returns a confident fix</label>
      <div class="toggles">
        <label><input type="checkbox" name="use_geoseer" {% if checks.use_geoseer %}checked{% endif %}> GeoSeer AI</label>
        <label><input type="checkbox" name="use_geoclip" class="clip" data-model="geoclip" {% if checks.use_geoclip %}checked{% endif %}> GeoCLIP (local)</label>
        <label><input type="checkbox" name="use_streetclip" class="clip" data-model="streetclip" {% if checks.use_streetclip %}checked{% endif %}> StreetCLIP (2nd model)</label>
        <label><input type="checkbox" name="use_ocr" {% if checks.use_ocr %}checked{% endif %}> OCR + plates</label>
        <label><input type="checkbox" name="use_street_match" {% if checks.use_street_match %}checked{% endif %}> Street imagery</label>
        <label><input type="checkbox" name="use_osm" {% if checks.use_osm %}checked{% endif %}> OSM cross-ref</label>
        <label><input type="checkbox" name="use_inaturalist" {% if checks.use_inaturalist %}checked{% endif %}> iNaturalist</label>
        <label><input type="checkbox" name="use_solar" {% if checks.use_solar %}checked{% endif %}> Solar / climate</label>
      </div>
    </fieldset>

    <button type="submit">Locate</button>
  </form>

  <div class="card">
    <div class="muted">LOCAL MODELS <span style="text-transform:none">— download once, then tick to use</span></div>
    <div id="models"><div class="muted" style="margin-top:8px">checking…</div></div>
  </div>

  {% if error %}<div class="card err">{{ error }}</div>{% endif %}

  {% if result %}
  <div class="card">
    <div class="muted">BEST GUESS</div>
    {% if result.best_guess.place or result.best_guess.coordinates %}
      <div class="place">{{ result.best_guess.place or "(coordinates only)" }}</div>
      {% if result.best_guess.coordinates %}
        {% set c = result.best_guess.coordinates %}
        <div class="muted">{{ "%.6f"|format(c.lat) }}, {{ "%.6f"|format(c.lon) }} ·
          <a href="https://www.openstreetmap.org/?mlat={{c.lat}}&mlon={{c.lon}}#map=14/{{c.lat}}/{{c.lon}}" target="_blank">map ↗</a></div>
      {% endif %}
      <div style="margin-top:8px"><span class="pill">{{ prec_label }}</span></div>
    {% else %}
      <div class="place">No location could be determined.</div>
    {% endif %}
    <div class="bar"><div style="width:{{ (result.best_guess.confidence*100)|int }}%; background:{{ conf_color }}"></div></div>
    <div class="muted">confidence {{ (result.best_guess.confidence*100)|int }}%</div>

    {% if result.signals %}
    <h3>Evidence</h3>
    <table>
      {% for s in result.signals %}
      <tr><td class="tag">[{{ s.source }}]</td><td>{{ s.description }}
        <div class="muted">confidence {{ "%.2f"|format(s.confidence) }} · {{ s.precision }}</div></td></tr>
      {% endfor %}
    </table>
    {% endif %}

    {% if result.notes %}
    <h3>Notes</h3>
    {% for n in result.notes %}<div class="muted">• {{ n }}</div>{% endfor %}
    {% endif %}

    {% if result.pivots %}
    <h3>Reverse-image-search pivots <span class="muted">(upload the photo manually)</span></h3>
    {% for p in result.pivots %}<div>{{ p.engine }} — <a href="{{ p.url }}" target="_blank">{{ p.url }}</a></div>{% endfor %}
    {% endif %}
  </div>
  {% endif %}

  <p class="muted" style="margin-top:24px">Keys are used only for this request and never stored. Runs locally.</p>
</div>
<script>
  // Show the loading overlay as soon as the form is submitted (the page then
  // does a full POST and won't return for ~30-60s while models run).
  document.getElementById('locateForm').addEventListener('submit', function () {
    document.getElementById('loadingOverlay').classList.add('show');
  });

  const only = document.getElementById('geoseer_only');
  // A local-model checkbox is usable only when NOT in GeoSeer-only mode AND the
  // weights are downloaded (data-ready set by the model poller below).
  function syncClip() {
    document.querySelectorAll('.clip').forEach(cb => {
      const ready = cb.dataset.ready === '1';
      cb.disabled = only.checked || !ready;
      if (cb.disabled) cb.checked = false;
    });
  }
  only.addEventListener('change', syncClip);

  async function refreshModels() {
    let s; try { s = await (await fetch('/models/status')).json(); } catch (e) { return; }
    const box = document.getElementById('models'); box.innerHTML = '';
    for (const key in s) {
      const m = s[key];
      const badge = m.state === 'ready' ? '<span style="color:#2ea043">● ready</span>'
        : m.state === 'downloading' ? '<span style="color:#d29922">● downloading ' + m.percent + '%</span>'
        : m.state === 'error' ? '<span style="color:#f85149">● error</span>'
        : m.state === 'needs_packages' ? '<span style="color:#8b949e">● needs packages</span>'
        : '<span style="color:#8b949e">● not downloaded</span>';
      let body = '';
      if (m.state === 'downloading')
        body = '<div class="bar"><div style="width:' + m.percent + '%;background:#d29922"></div></div>'
             + '<div class="muted">' + m.downloaded_mb + ' / ' + (m.total_mb || '?') + ' MB</div>';
      else if (m.state === 'needs_packages')
        body = '<div class="muted">install first: <span class="tag">' + m.install_hint + '</span></div>';
      else if (m.state !== 'ready')
        body = '<button type="button" style="margin-top:8px;padding:7px 14px;font-size:13px" onclick="dl(\\'' + key + '\\')">Download (~' + m.approx_gb + ' GB)</button>';
      if (m.error) body += '<div class="err">' + m.error + '</div>';
      const d = document.createElement('div');
      d.style.cssText = 'border:1px solid #30363d;border-radius:8px;padding:10px;margin:8px 0';
      d.innerHTML = '<div style="display:flex;justify-content:space-between;align-items:center"><b>' + m.label + '</b>' + badge + '</div>' + body;
      box.appendChild(d);
      const cb = document.querySelector('.clip[data-model="' + key + '"]');
      if (cb) cb.dataset.ready = (m.state === 'ready') ? '1' : '0';
    }
    syncClip();
  }
  async function dl(model) {
    await fetch('/models/download', { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model }) });
    refreshModels();
  }
  setInterval(refreshModels, 1500); refreshModels();
</script>
</body></html>
"""


def _checks_from_form(form):
    if not form:  # GET: sensible defaults
        return {"use_ocr": True, "use_geoclip": True, "use_geoseer": True,
                "use_streetclip": True, "use_street_match": True, "use_osm": True,
                "use_inaturalist": True, "use_solar": True, "geoseer_only": False,
                "full_workflow": False}
    keys = ("use_ocr", "use_geoclip", "use_geoseer", "use_streetclip",
            "use_street_match", "use_osm", "use_inaturalist", "use_solar",
            "geoseer_only", "full_workflow")
    return {k: (k in form) for k in keys}


def _config_from_form(form) -> AnalyzeConfig:
    geoseer_only = "geoseer_only" in form
    return AnalyzeConfig(
        use_ocr="use_ocr" in form,
        use_geoclip=("use_geoclip" in form) and not geoseer_only,
        use_geoseer=("use_geoseer" in form) or geoseer_only,
        use_streetclip=("use_streetclip" in form) and not geoseer_only,
        use_street_match="use_street_match" in form,
        use_osm="use_osm" in form,
        use_inaturalist="use_inaturalist" in form,
        use_solar="use_solar" in form,
        short_circuit_on_geoseer=("full_workflow" not in form),
        geoseer_key=(form.get("geoseer_key") or "").strip() or None,
        mapillary_token=(form.get("mapillary_token") or "").strip() or None,
        serpapi_key=(form.get("serpapi_key") or "").strip() or None,
    )


def _conf_color(conf: float) -> str:
    return "#2ea043" if conf >= 0.6 else "#d29922" if conf >= 0.4 else "#f85149"


def create_app():
    from flask import Flask, request, render_template_string, jsonify

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB uploads

    def render(result=None, form=None, error=None):
        checks = _checks_from_form(form)
        keys = {"geoseer_key": (form.get("geoseer_key", "") if form else ""),
                "mapillary_token": (form.get("mapillary_token", "") if form else ""),
                "serpapi_key": (form.get("serpapi_key", "") if form else "")}
        prec_label = _PRECISION_LABEL.get(
            result["best_guess"]["precision"], "") if result else ""
        conf_color = _conf_color(result["best_guess"]["confidence"]) if result else "#2ea043"
        return render_template_string(PAGE, result=result, checks=checks, keys=keys,
                                      error=error, prec_label=prec_label, conf_color=conf_color)

    @app.route("/", methods=["GET"])
    def index():
        return render()

    @app.route("/models/status", methods=["GET"])
    def models_status():
        return jsonify(modelmgr.all_status())

    @app.route("/models/download", methods=["POST"])
    def models_download():
        data = request.get_json(silent=True) or {}
        return jsonify(modelmgr.start_download(data.get("model", "")))

    @app.route("/analyze", methods=["POST"])
    def do_analyze():
        # Require the local models to be set up — no analysing on a crippled config.
        ok, problems = modelmgr.preflight()
        if not ok:
            return render(form=request.form,
                          error="Required local models aren't ready — download them "
                                "in the Local models panel above: " + "; ".join(problems))
        file = request.files.get("image")
        if not file or not file.filename:
            return render(form=request.form, error="Please choose an image file.")
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        file.save(path)
        try:
            result = analyze(path, _config_from_form(request.form))
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        return render(result=result.to_dict(), form=request.form)

    return app


def main(argv=None) -> int:
    try:
        create_app  # noqa
        app = create_app()
    except ImportError:
        print("Flask is required for the web UI:  pip install Flask")
        return 1
    host = os.environ.get("GEOLOCATOR_UI_HOST", "127.0.0.1")
    port = int(os.environ.get("GEOLOCATOR_UI_PORT", "5000"))
    print(f"geolocator UI -> http://{host}:{port}  (Ctrl+C to stop)")
    # threaded=False: run analysis in the main thread. Loading a torch model in a
    # Flask worker thread leaves its weights on the 'meta' device and inference
    # fails ("Tensor on device cpu is not on the expected device meta"). Serial
    # request handling is fine for a local single-user tool.
    app.run(host=host, port=port, debug=False, threaded=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

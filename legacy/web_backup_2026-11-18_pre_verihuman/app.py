"""
Flask backend for the Audio Deepfake Detector.

Start the app with:

    python app.py

then open http://127.0.0.1:5000 in your browser.

Design notes
------------
* The trained model bundle is loaded ONCE at import time.  The notebook is never
  executed and the model is never retrained when a page is opened.
* All uploaded files are analysed in a single batch call.  Every chunk from every
  file goes through `predict_proba` together, which is much faster than looping
  file by file.
* One bad file never fails the whole request: it gets its own `error` entry and
  the rest of the batch continues.
* The response contains calibrated scores, a three-way decision
  (BONAFIDE / UNCERTAIN / SPOOF), per-chunk details and timing information.
"""

import os
import tempfile
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from audio_features import load_model, predict_audio_files

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "audio_deepfake_model.joblib"

ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".webm", ".m4a", ".oga"}
# WebM/M4A/OGG-Opus are browser-recording containers. The UI converts recordings
# to WAV in the browser, so these are only a fallback and need ffmpeg to decode.
MAX_UPLOAD_MB = 100
MAX_FILES = 20

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Load the model once, at import time.
try:
    MODEL_BUNDLE = load_model(MODEL_PATH)
    MODEL_ERROR = None
except Exception as exc:  # noqa: BLE001 - the app should still start so the page can explain
    MODEL_BUNDLE = None
    MODEL_ERROR = str(exc)
    print(f"[WARNING] {exc}")


def _warm_up():
    """Run one synthetic chunk so the first real upload is not slow.

    The first feature extraction in a fresh process pays for lazy imports and
    library initialisation.  Doing it at startup moves that cost off the user's
    first request.
    """
    if MODEL_BUNDLE is None:
        return
    try:
        import numpy as np

        import audio_features as af

        feature_set = MODEL_BUNDLE.get("feature_set", "default")
        seconds = float(MODEL_BUNDLE.get("chunk_seconds", af.CHUNK_SECONDS))
        samples = max(int(seconds * af.SAMPLE_RATE), int(af.MIN_SECONDS * af.SAMPLE_RATE))
        time_axis = np.arange(samples, dtype=np.float32) / af.SAMPLE_RATE
        rng = np.random.default_rng(0)
        waveform = (
            0.01 * np.sin(2 * np.pi * 220.0 * time_axis)
            + 0.001 * rng.standard_normal(samples).astype(np.float32)
        ).astype(np.float32)
        vector = af.extract_features_from_waveform(
            waveform, af.SAMPLE_RATE, feature_set=feature_set
        )
        MODEL_BUNDLE["model"].predict_proba(np.asarray([vector]))
        print("[startup] feature pipeline and model warmed up")
    except Exception as exc:  # noqa: BLE001 - warm-up is optional
        print(f"[startup] warm-up skipped: {exc}")


_warm_up()


def model_summary():
    """Small, JSON-safe description of the loaded model."""
    if MODEL_BUNDLE is None:
        return {"loaded": False, "error": MODEL_ERROR}
    return {
        "loaded": True,
        "model_name": MODEL_BUNDLE.get("model_name", "unknown"),
        "feature_set": MODEL_BUNDLE.get("feature_set", "unknown"),
        "n_features": MODEL_BUNDLE.get("n_features"),
        "sample_rate": MODEL_BUNDLE.get("sample_rate"),
        "chunk_seconds": MODEL_BUNDLE.get("chunk_seconds"),
        "spoof_threshold": MODEL_BUNDLE.get("spoof_threshold"),
        "bonafide_threshold": MODEL_BUNDLE.get("bonafide_threshold"),
        "calibrated": MODEL_BUNDLE.get("calibrator") is not None,
        "trained_on_files": MODEL_BUNDLE.get("trained_on_files"),
        "trained_on_speakers": MODEL_BUNDLE.get("trained_on_speakers"),
        "created_at": MODEL_BUNDLE.get("created_at"),
        "validation": MODEL_BUNDLE.get("validation"),
        "eval": {
            "metrics": (MODEL_BUNDLE.get("eval") or {}).get("metrics"),
            "calibration": (MODEL_BUNDLE.get("eval") or {}).get("calibration"),
        },
    }


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    """Diagnostic endpoint: is the model loaded, and how was it trained?"""
    return jsonify({
        "model_loaded": MODEL_BUNDLE is not None,
        "model_path": str(MODEL_PATH),
        "error": MODEL_ERROR,
        "model": model_summary(),
    })


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({
        "error": f"Upload is too large. The limit is {MAX_UPLOAD_MB} MB in total."
    }), 413


@app.post("/predict")
def predict():
    if MODEL_BUNDLE is None:
        return jsonify({
            "error": f"The model could not be loaded. {MODEL_ERROR}"
        }), 503

    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        return jsonify({"error": "Please choose at least one audio file."}), 400

    if len(files) > MAX_FILES:
        return jsonify({
            "error": f"Too many files. Please upload at most {MAX_FILES} at once."
        }), 400

    # Save uploads first. Unsupported names become immediate error entries.
    entries = []          # (display name, temp path or None, error or None)
    temp_paths = []

    try:
        for uploaded_file in files:
            display_name = Path(uploaded_file.filename).name
            suffix = Path(display_name).suffix.lower()

            if suffix not in ALLOWED_EXTENSIONS:
                entries.append((
                    display_name, None,
                    "Unsupported format. Upload WAV, FLAC, MP3 or OGG, or record "
                    "your voice in the browser.",
                ))
                continue

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as temp_file:
                    uploaded_file.save(temp_file)
                    temp_path = Path(temp_file.name)
                temp_paths.append(temp_path)
                entries.append((display_name, temp_path, None))
            except Exception as exc:  # noqa: BLE001 - keep the batch alive
                entries.append((
                    display_name, None,
                    f"Could not read the upload: {exc}",
                ))

        paths = [path for _name, path, error in entries if error is None]
        started = time.perf_counter()
        analysed = predict_audio_files(paths, MODEL_BUNDLE) if paths else []
        elapsed = time.perf_counter() - started

    finally:
        for path in temp_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    # Put results back in the original upload order.
    results = []
    analysed_index = 0
    for display_name, path, error in entries:
        if error is not None:
            results.append({"filename": display_name, "error": error})
            continue

        result = dict(analysed[analysed_index])
        analysed_index += 1
        result["filename"] = display_name
        if result.get("error"):
            # The analysis layer only knows the temp file name; show the name
            # the user actually uploaded.
            result["error"] = result["error"].replace(Path(path).name, display_name)
        results.append(result)

    summary = {"bonafide": 0, "spoof": 0, "uncertain": 0, "errors": 0}
    for result in results:
        if result.get("error"):
            summary["errors"] += 1
        else:
            key = str(result.get("prediction", "")).lower()
            if key in summary:
                summary[key] += 1

    return jsonify({
        "results": results,
        "summary": summary,
        "count": len(results),
        "elapsed_seconds": round(elapsed, 3),
        "model": model_summary(),
    })


if __name__ == "__main__":
    # threaded=True keeps the browser responsive while a batch is analysed.
    # Set FLASK_DEBUG=1 to get the reloader/verbose errors during development.
    debug = os.environ.get("FLASK_DEBUG", "").strip() in {"1", "true", "yes"}
    app.run(host="127.0.0.1", port=5000, debug=debug, threaded=True)

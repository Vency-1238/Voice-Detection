"""Website JSON API.

`/api/analyze` is the ONLY endpoint the new UI calls for detection.  It is a thin
adapter: it saves the upload, calls the existing `predict_audio_files()` interface
once per file, stores the returned values, and reports them back unchanged.
"""

from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, url_for
from flask_login import current_user, login_required

from . import db
from .services import (
    analyse_file,
    get_bundle,
    model_error,
    model_info,
    probe_metadata,
    save_analysis,
    store_audio_copy,
)

bp = Blueprint("api", __name__, url_prefix="/api")

ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".webm", ".m4a", ".oga"}


def _clean_name(raw: str) -> str:
    name = Path(raw or "audio").name.strip().replace("\x00", "")
    return name[:255] or "audio"


@bp.post("/analyze")
@login_required
def analyze():
    bundle = get_bundle()
    if bundle is None:
        return jsonify({
            "error": "The detection model is not loaded. " + (model_error() or "")
        }), 503

    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        return jsonify({"error": "Choose at least one audio file."}), 400

    max_files = int(current_app.config["VERIHUMAN_MAX_FILES"])
    if len(files) > max_files:
        return jsonify({"error": f"Upload at most {max_files} files at once."}), 400

    max_bytes = int(current_app.config["VERIHUMAN_MAX_FILE_MB"]) * 1024 * 1024
    started = time.perf_counter()
    results = []
    summary = {"bonafide": 0, "spoof": 0, "uncertain": 0, "errors": 0}

    try:
        with tempfile.TemporaryDirectory(prefix="verihuman_") as tmp:
            tmp_dir = Path(tmp)
            for uploaded in files:
                display_name = _clean_name(uploaded.filename)
                suffix = Path(display_name).suffix.lower()

                if suffix not in ALLOWED_EXTENSIONS:
                    results.append({
                        "filename": display_name,
                        "error": "Unsupported format. Use WAV, FLAC, MP3 or OGG.",
                    })
                    summary["errors"] += 1
                    continue

                temp_path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"
                try:
                    uploaded.save(temp_path)
                    size = temp_path.stat().st_size
                    if size == 0:
                        raise ValueError("The file is empty.")
                    if size > max_bytes:
                        raise ValueError(
                            f"File is larger than {current_app.config['VERIHUMAN_MAX_FILE_MB']} MB."
                        )
                except Exception as exc:  # noqa: BLE001 - per-file error
                    results.append({"filename": display_name, "error": f"Could not read the file: {exc}"})
                    summary["errors"] += 1
                    continue

                meta = probe_metadata(temp_path)
                try:
                    result = analyse_file(temp_path, display_name, bundle)
                except Exception as exc:  # noqa: BLE001 - never fail the whole batch
                    results.append({
                        "filename": display_name,
                        "error": f"Could not analyse this file: {exc}",
                    })
                    summary["errors"] += 1
                    continue

                if result.get("error"):
                    results.append({"filename": display_name, "error": result["error"]})
                    summary["errors"] += 1
                    continue

                audio_file = (
                    store_audio_copy(current_user, temp_path)
                    if getattr(current_user, "retain_audio", False)
                    else None
                )
                record = save_analysis(current_user, meta, result, audio_file=audio_file)
                result.update({
                    "analysis_id": record.id,
                    "report_id": record.report_id,
                    "report_url": url_for("website.report", analysis_id=record.id),
                    "pdf_url": url_for("website.report_pdf", analysis_id=record.id),
                    "csv_url": url_for("website.report_csv", analysis_id=record.id),
                    "audio_url": url_for("website.report_audio", analysis_id=record.id) if audio_file else None,
                    "file_format": meta.get("file_format"),
                    "size_bytes": meta.get("size_bytes"),
                    "sample_rate": meta.get("sample_rate"),
                    "channels": meta.get("channels"),
                })
                results.append(result)

                key = str(result.get("prediction", "")).lower()
                if key in summary:
                    summary[key] += 1

        db.session.commit()
    except Exception as exc:  # noqa: BLE001 - database/other unexpected failure
        db.session.rollback()
        return jsonify({"error": f"The analysis could not be saved: {exc}"}), 500

    elapsed = time.perf_counter() - started
    return jsonify({
        "results": results,
        "summary": summary,
        "count": len(results),
        "elapsed_seconds": round(elapsed, 3),
        "model": model_info(),
    })

"""Thin website adapter around the EXISTING prediction interface.

The only model code used here is the public helper that the original Flask app
already calls:

    audio_features.predict_audio_files(paths, model_bundle)

Nothing is retrained, re-implemented or re-tuned.  This module adds file metadata,
timing and database rows, which are website concerns.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from flask import current_app

from . import db
from .models import Analysis


def get_bundle():
    return current_app.config.get("VERIHUMAN_MODEL_BUNDLE")


def model_error():
    return current_app.config.get("VERIHUMAN_MODEL_ERROR")


def model_info() -> dict:
    bundle = get_bundle() or {}
    evaluation = (bundle.get("eval") or {})
    validation = bundle.get("validation") or {}
    benchmark = {}
    try:
        benchmark_path = Path(current_app.root_path) / "benchmarks" / "baseline_model.json"
        if benchmark_path.exists():
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    except Exception:
        benchmark = {}
    return {
        "loaded": bool(bundle),
        "model_name": bundle.get("model_name", "unknown"),
        "feature_set": bundle.get("feature_set", "unknown"),
        "n_features": bundle.get("n_features"),
        "sample_rate": bundle.get("sample_rate"),
        "chunk_seconds": bundle.get("chunk_seconds"),
        "spoof_threshold": bundle.get("spoof_threshold"),
        "bonafide_threshold": bundle.get("bonafide_threshold"),
        "calibrated": bundle.get("calibrator") is not None,
        "trained_on_files": bundle.get("trained_on_files"),
        "trained_on_speakers": bundle.get("trained_on_speakers"),
        "created_at": bundle.get("created_at"),
        "eval_metrics": evaluation.get("metrics"),
        "eval_calibration": evaluation.get("calibration"),
        "dev_test": validation.get("dev_test"),
        "dev_test_calibration": validation.get("dev_test_calibration"),
        "dev_test_three_way": validation.get("dev_test_three_way"),
        "eval_validation": validation.get("eval_validation"),
        "eval_validation_three_way": validation.get("eval_validation_three_way"),
        "holdout": validation.get("eval_holdout"),
        "holdout_calibration": validation.get("eval_holdout_calibration"),
        "holdout_three_way": validation.get("eval_holdout_three_way"),
        "per_attack": evaluation.get("per_attack"),
        "threshold_tuning": validation.get("threshold_tuning"),
        "target_fpr": bundle.get("target_fpr"),
        "augmentation": bundle.get("augmentation"),
        "version": bundle.get("version"),
        "feature_description": bundle.get("feature_description"),
        "benchmark": benchmark,
    }


def probe_metadata(path: Path) -> dict:
    """Read real file metadata for display/storage.

    This is not used by the model; it only fills website fields.  If the
    container cannot be inspected, the fields stay unknown rather than guessed.
    """
    meta = {
        "file_format": path.suffix.replace(".", "").upper() or "—",
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "duration_seconds": None,
        "sample_rate": None,
        "channels": None,
    }
    try:
        import soundfile as sf

        info = sf.info(str(path))
        meta["duration_seconds"] = float(info.frames) / float(info.samplerate)
        meta["sample_rate"] = int(info.samplerate)
        meta["channels"] = int(info.channels)
        return meta
    except Exception:
        pass

    try:
        from audio_features import audio_duration

        meta["duration_seconds"] = float(audio_duration(path))
    except Exception:
        pass
    return meta


def analyse_file(path: Path, display_name: str, bundle) -> dict:
    """Run the existing predictor on one file and time it.

    `predict_audio_files` is called exactly as the original app calls it.  The
    file is passed as a one-item list so the measured processing time belongs to
    this file only.

    After the model returns, the same waveform is passed to the post-hoc
    human-voice analysis.  That module is independent of the model and cannot
    change the 172-feature vector, the calibrator, or the model thresholds.
    """
    from audio_features import load_audio, predict_audio_files
    from human_voice_analysis import analyze_human_voice

    started = time.perf_counter()
    results = predict_audio_files([path], bundle)
    result = dict(results[0]) if results else {"error": "No prediction was returned."}

    if not result.get("error"):
        try:
            waveform, sample_rate = load_audio(path, max_seconds=30)
            acoustic = analyze_human_voice(
                waveform, sample_rate, model_result=result
            )
            result["acoustic_analysis"] = acoustic

            fusion = acoustic.get("fusion") or {}
            model_prediction = fusion.get("model_prediction") or result.get("prediction")
            final_prediction = fusion.get("final_prediction") or result.get("prediction")
            result["model_prediction"] = model_prediction
            result["model_reason"] = result.get("reason")
            if final_prediction in {"BONAFIDE", "SPOOF", "UNCERTAIN"}:
                result["prediction"] = final_prediction
                result["display_label"] = {
                    "SPOOF": "AI-GENERATED / SPOOF",
                    "BONAFIDE": "REAL / BONAFIDE",
                    "UNCERTAIN": "UNCERTAIN - NEEDS REVIEW",
                }[final_prediction]
            if fusion.get("reason"):
                result["reason"] = fusion["reason"]
            if fusion.get("confidence") is not None:
                result["confidence"] = fusion["confidence"]
            if fusion.get("confidence_label"):
                result["confidence_label"] = fusion["confidence_label"]
        except Exception as exc:  # noqa: BLE001 - never break the model result
            result["acoustic_analysis"] = {
                "available": False,
                "version": "1.0",
                "quality": {
                    "status": "unavailable",
                    "reliability": 0.0,
                    "reliability_label": "Low",
                    "notes": [f"Acoustic analysis could not be completed: {exc}"],
                },
                "factors": {},
                "fusion": {
                    "final_prediction": result.get("prediction", "UNCERTAIN"),
                    "model_prediction": result.get("prediction", "UNCERTAIN"),
                    "evidence_direction": "inconclusive",
                    "agreement": "inconclusive",
                    "confidence": result.get("confidence"),
                    "confidence_label": result.get("confidence_label"),
                    "reason": result.get("reason", ""),
                    "summary": "Acoustic analysis was unavailable.",
                    "changed_from_model": False,
                },
                "disclaimer": (
                    "Acoustic analysis was unavailable for this recording; the "
                    "model result is shown unchanged."
                ),
            }

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result["filename"] = display_name
    result["processing_ms"] = round(elapsed_ms, 1)
    return result


def _fraction(value):
    """The pipeline returns percentages; the website stores 0..1 for math."""
    if value is None:
        return None
    try:
        return float(value) / 100.0
    except (TypeError, ValueError):
        return None


def save_analysis(user, meta: dict, result: dict, audio_file: str | None = None) -> Analysis:
    """Persist the actual values returned by the model."""
    bundle = get_bundle() or {}
    chunk_results = result.get("chunk_results") or []
    acoustic = result.get("acoustic_analysis") or {}
    record = Analysis(
        user_id=user.id,
        filename=result.get("filename", "recording"),
        file_format=meta.get("file_format") or "—",
        size_bytes=int(meta.get("size_bytes") or 0),
        duration_seconds=meta.get("duration_seconds") or result.get("duration_seconds"),
        sample_rate=meta.get("sample_rate"),
        channels=meta.get("channels"),
        prediction=result.get("prediction", "UNCERTAIN"),
        model_prediction=result.get("model_prediction"),
        acoustic_json=json.dumps(acoustic) if acoustic else None,
        spoof_score=_fraction(result.get("spoof_score")),
        raw_spoof_score=_fraction(result.get("raw_spoof_score")),
        confidence=_fraction(result.get("confidence")),
        confidence_label=result.get("confidence_label"),
        reason=result.get("reason"),
        bonafide_threshold=result.get("bonafide_threshold"),
        spoof_threshold=result.get("threshold"),
        chunk_count=result.get("chunks_analysed"),
        chunk_seconds=result.get("chunk_seconds"),
        chunk_disagreement=result.get("chunk_disagreement"),
        chunk_json=json.dumps(chunk_results) if chunk_results else None,
        truncated=bool(result.get("truncated")),
        processing_ms=result.get("processing_ms"),
        model_name=result.get("model_name") or bundle.get("model_name"),
        model_version=bundle.get("created_at"),
        feature_set=result.get("feature_set") or bundle.get("feature_set"),
        n_features=bundle.get("n_features"),
        audio_file=audio_file,
    )
    db.session.add(record)
    db.session.flush()
    return record


def store_audio_copy(user, source: Path) -> str:
    """Optionally keep a private copy when the user enabled retention."""
    upload_dir = Path(current_app.config["VERIHUMAN_UPLOAD_DIR"])
    user_dir = upload_dir / str(user.id)
    user_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower() or ".wav"
    name = f"{uuid.uuid4().hex}{suffix}"
    target = user_dir / name
    target.write_bytes(source.read_bytes())
    return f"{user.id}/{name}"


def retained_audio_path(analysis: Analysis) -> Path | None:
    if not analysis.audio_file:
        return None
    candidate = Path(current_app.config["VERIHUMAN_UPLOAD_DIR"]) / analysis.audio_file
    return candidate if candidate.exists() else None


def delete_retained_audio(analysis: Analysis) -> None:
    path = retained_audio_path(analysis)
    if path is not None:
        try:
            path.unlink()
        except OSError:
            pass


def delete_user_audio(user) -> None:
    user_dir = Path(current_app.config["VERIHUMAN_UPLOAD_DIR"]) / str(user.id)
    if not user_dir.exists():
        return
    for path in user_dir.glob("*"):
        try:
            path.unlink()
        except OSError:
            pass
    try:
        user_dir.rmdir()
    except OSError:
        pass

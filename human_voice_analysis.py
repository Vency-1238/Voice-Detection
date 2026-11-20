"""Post-hoc human voice characteristic analysis for VeriHuman.

This module is intentionally separate from ``audio_features.py``.  It does not
change the trained model, the 172-feature vector, the calibrator, or any model
threshold.  It analyses the waveform again with transparent acoustic measurements
and returns heuristic evidence about four characteristics:

* breathing patterns
* micro-pauses
* spectral characteristics
* voice continuity

The scores are **not** calibrated probabilities and they are **not** proof of
human or AI origin.  The trained classifier remains the primary detector.  The
acoustic layer is used to explain the model's result, to flag conflicts, and to
downgrade a verdict to UNCERTAIN when the evidence is contradictory.

Design rules
------------
* All thresholds are relative to the recording's own noise floor and speech level.
* A factor that cannot be measured returns ``status="unavailable"`` or
  ``status="inconclusive"`` with ``score=None``.  No score is invented.
* Absence of breathing is not evidence of synthetic speech.
* Presence of breathing is not proof of human speech.
* The final verdict is conservative: acoustic evidence alone can only change a
  model verdict to UNCERTAIN (or resolve an already-uncertain borderline case
  when at least three independent cues agree strongly).
"""

from __future__ import annotations

import math
from typing import Any

import librosa
import numpy as np
from scipy.ndimage import binary_closing, binary_opening, median_filter

SAMPLE_RATE = 16000
HOP_LENGTH = 160          # 10 ms
FRAME_LENGTH = 400        # 25 ms
N_FFT = 512
PITCH_FRAME_LENGTH = 1024
MAX_ANALYSIS_SECONDS = 30.0
MIN_SECONDS = 0.75
N_MFCC = 20

FACTOR_ORDER = ("breathing", "micro_pauses", "spectral", "voice_continuity")

DISCLAIMER = (
    "These are measured acoustic cues and heuristic scores, not proof of human "
    "or AI origin. Breathing, pauses, spectral shape and voice continuity can be "
    "altered by microphones, codecs, noise reduction, editing and speaking style. "
    "The trained classifier remains the primary detector."
)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(max(low, min(high, value)))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: Any, digits: int = 1) -> float | None:
    number = _finite(value)
    return round(number, digits) if number is not None else None


def _fmt(value: Any, unit: str = "", digits: int = 1) -> str:
    number = _finite(value)
    if number is None:
        return "—"
    text = f"{number:.{digits}f}"
    return f"{text}{unit}" if unit else text


def _range_score(
    value: float | None,
    ideal_low: float,
    ideal_high: float,
    hard_low: float,
    hard_high: float,
) -> float | None:
    """100 inside the ideal range, falling linearly to 0 at the hard bounds."""
    if value is None:
        return None
    value = float(value)
    if ideal_low <= value <= ideal_high:
        return 100.0
    if value < ideal_low:
        span = ideal_low - hard_low
        if span <= 0:
            return 0.0
        return _clamp(100.0 * (value - hard_low) / span)
    span = hard_high - ideal_high
    if span <= 0:
        return 0.0
    return _clamp(100.0 * (hard_high - value) / span)


def _upper_score(value: float | None, ideal_high: float, hard_high: float) -> float | None:
    """100 at or below ``ideal_high``, falling linearly to 0 at ``hard_high``."""
    if value is None:
        return None
    if value <= ideal_high:
        return 100.0
    if value >= hard_high:
        return 0.0
    return _clamp(100.0 * (hard_high - value) / (hard_high - ideal_high))


def _weighted_mean(pairs: list[tuple[float | None, float]]) -> float | None:
    total = 0.0
    weight_sum = 0.0
    for value, weight in pairs:
        if value is None:
            continue
        total += float(value) * float(weight)
        weight_sum += float(weight)
    return total / weight_sum if weight_sum > 0 else None


def _mean(values: np.ndarray | None) -> float | None:
    if values is None or np.size(values) == 0:
        return None
    return float(np.mean(values))


def _median(values: np.ndarray | None) -> float | None:
    if values is None or np.size(values) == 0:
        return None
    return float(np.median(values))


def _reliability_label(value: float) -> str:
    if value >= 0.70:
        return "High"
    if value >= 0.40:
        return "Medium"
    return "Low"


def _score_status(score: float | None, natural: float = 65.0, unusual: float = 30.0) -> str:
    if score is None:
        return "inconclusive"
    if score >= natural:
        return "natural"
    if score <= unusual:
        return "unusual"
    return "some_anomalies"


def _status_label(status: str) -> str:
    return {
        "present": "Breath-like events detected",
        "absent": "No breath-like events detected",
        "inconclusive": "Inconclusive",
        "unavailable": "Not available",
        "natural": "Natural",
        "some_anomalies": "Some unusual features",
        "unusual": "Unusual",
    }.get(status, status.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Framing, VAD and segmentation
# ---------------------------------------------------------------------------
def _frame_features(y: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Compute frame-level measurements once and reuse them for every factor."""
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    rms = np.asarray(rms, dtype=np.float64)
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-8))

    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=N_FFT, hop_length=HOP_LENGTH
    )[0]
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )[0]
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )[0]
    rolloff85 = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, roll_percent=0.85
    )[0]
    rolloff99 = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, roll_percent=0.99
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]

    spectrum = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    power = spectrum * spectrum
    mel = librosa.feature.melspectrogram(
        S=power, sr=sr, n_mels=128, fmax=sr / 2.0
    )
    log_mel = librosa.power_to_db(mel, ref=1.0)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC)
    try:
        delta = librosa.feature.delta(mfcc, width=9)
    except Exception:
        delta = np.zeros_like(mfcc)

    flux_raw = np.linalg.norm(np.diff(spectrum, axis=1), axis=0) / (
        np.linalg.norm(spectrum[:, 1:], axis=0) + 1e-9
    )
    flux = np.concatenate([[0.0], flux_raw])

    contrast = librosa.feature.spectral_contrast(
        S=spectrum, sr=sr, n_bands=6, fmin=100.0
    )

    f0 = librosa.yin(
        y,
        fmin=60.0,
        fmax=400.0,
        sr=sr,
        frame_length=PITCH_FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        trough_threshold=0.1,
    )

    frames = min(
        len(rms_db), len(flatness), len(centroid), len(bandwidth),
        len(rolloff85), len(rolloff99), len(zcr), len(spectrum[0]),
        len(mfcc[0]), len(contrast[0]), len(flux), len(f0),
    )
    if frames <= 0:
        raise ValueError("Audio is too short to frame for analysis.")

    return {
        "frames": frames,
        "rms": rms[:frames],
        "rms_db": rms_db[:frames],
        "flatness": flatness[:frames],
        "centroid": centroid[:frames],
        "bandwidth": bandwidth[:frames],
        "rolloff85": rolloff85[:frames],
        "rolloff99": rolloff99[:frames],
        "zcr": zcr[:frames],
        "spectrum": spectrum[:, :frames],
        "contrast": contrast[:, :frames],
        "mfcc": mfcc[:, :frames],
        "delta": delta[:, :frames],
        "flux": flux[:frames],
        "f0": f0[:frames],
        "frequencies": librosa.fft_frequencies(sr=sr, n_fft=N_FFT),
    }


def _voice_activity(rms_db: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Adaptive energy VAD with relative thresholds and short-gap smoothing."""
    if rms_db.size == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, {"noise_floor_db": 0.0, "speech_level_db": 0.0, "dynamic_range_db": 0.0}

    p10, p50, p95 = np.percentile(rms_db, [10.0, 50.0, 95.0])
    dynamic_range = float(max(p95 - p10, 0.0))
    threshold = float(p10 + max(5.0, min(0.30 * dynamic_range, 12.0)))
    vad = rms_db > threshold

    # Fill very short gaps (stop closures) and remove isolated clicks.
    vad = median_filter(vad.astype(np.uint8), size=5).astype(bool)
    vad = binary_closing(vad, structure=np.ones(5, dtype=bool)).astype(bool)
    vad = binary_opening(vad, structure=np.ones(3, dtype=bool)).astype(bool)

    return vad, {
        "noise_floor_db": float(p10),
        "median_db": float(p50),
        "speech_level_db": float(p95),
        "dynamic_range_db": dynamic_range,
        "threshold_db": threshold,
    }


def _mask_segments(mask: np.ndarray, min_frames: int = 1) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start >= min_frames
    ]


def _internal_pauses(
    speech_segments: list[tuple[int, int]], min_frames: int
) -> list[tuple[int, int]]:
    pauses: list[tuple[int, int]] = []
    for (_first_start, first_end), (second_start, _second_end) in zip(
        speech_segments, speech_segments[1:]
    ):
        gap = second_start - first_end
        if gap >= min_frames:
            pauses.append((first_end, second_start))
    return pauses


# ---------------------------------------------------------------------------
# Factor 1: breathing patterns
# ---------------------------------------------------------------------------
def _analyze_breathing(
    features: dict[str, np.ndarray],
    pauses: list[tuple[int, int]],
    sr: int,
    speech_seconds: float,
    quality: dict[str, Any],
) -> dict[str, Any]:
    rms_db = features["rms_db"]
    flatness = features["flatness"]
    centroid = features["centroid"]
    zcr = features["zcr"]
    f0 = features["f0"]
    spectrum = features["spectrum"]
    frequencies = features["frequencies"]

    noise_floor = quality["noise_floor_db"]
    speech_level = quality["speech_level_db"]
    reliability = quality["reliability"]

    if quality["status"] in {"silent", "no_speech", "unavailable"}:
        return _factor_unavailable(
            "breathing",
            "Breathing patterns",
            "No usable speech was detected, so breathing cannot be assessed.",
            reliability,
            caveat="Breath detection needs speech with quiet gaps between phrases.",
        )

    min_frames = int(0.15 * sr / HOP_LENGTH)
    observable = [
        (start, end) for start, end in pauses
        if (end - start) >= min_frames and (end - start) <= int(2.0 * sr / HOP_LENGTH)
    ]

    if not observable:
        return _factor_inconclusive(
            "breathing",
            "Breathing patterns",
            "There were no quiet gaps long enough to look for breath sounds.",
            reliability,
            metrics={
                "Internal pauses": str(len(pauses)),
                "Observable gaps": "0",
                "Pitch-like frames": "—",
            },
            caveat=(
                "Short or continuously voiced clips often contain no observable "
                "breaths. This is not evidence of synthetic speech."
            ),
        )

    low_band = (frequencies >= 100.0) & (frequencies <= 1500.0)
    candidates: list[dict[str, float]] = []

    for start, end in observable:
        # Use the middle 60% so speech edges do not leak into the measurement.
        span = end - start
        mid_start = start + int(0.20 * span)
        mid_end = end - int(0.20 * span)
        if mid_end - mid_start < 3:
            continue

        seg_db = float(np.mean(rms_db[mid_start:mid_end]))
        rel_noise = seg_db - noise_floor
        below_speech = speech_level - seg_db
        flat = float(np.mean(flatness[mid_start:mid_end]))
        cent = float(np.mean(centroid[mid_start:mid_end]))
        zcr_mean = float(np.mean(zcr[mid_start:mid_end]))

        segment_power = spectrum[:, mid_start:mid_end] ** 2
        total_power = float(np.sum(segment_power))
        low_ratio = (
            float(np.sum(segment_power[low_band, :])) / total_power
            if total_power > 0 else 0.0
        )

        pitch_like = float(np.mean(
            (f0[mid_start:mid_end] >= 60.0)
            & (f0[mid_start:mid_end] <= 400.0)
            & (flatness[mid_start:mid_end] < 0.45)
        ))

        is_candidate = (
            rel_noise >= 1.5
            and below_speech >= 3.0
            and flat >= 0.15
            and 250.0 <= cent <= 4000.0
            and zcr_mean <= 0.45
            and pitch_like <= 0.45
        )
        if not is_candidate:
            continue

        quality_score = _weighted_mean([
            (_range_score(flat, 0.20, 0.80, 0.05, 0.95), 0.30),
            (_range_score(rel_noise, 2.0, 12.0, 0.0, 25.0), 0.20),
            (_range_score(below_speech, 3.0, 25.0, 0.0, 45.0), 0.20),
            (_range_score(low_ratio, 0.25, 0.85, 0.0, 1.0), 0.20),
            (_range_score(zcr_mean, 0.02, 0.25, 0.0, 0.50), 0.10),
        ])
        candidates.append({
            "duration": span * HOP_LENGTH / sr,
            "rel_noise": rel_noise,
            "below_speech": below_speech,
            "flatness": flat,
            "centroid": cent,
            "low_ratio": low_ratio,
            "pitch_like": pitch_like,
            "quality": float(quality_score or 0.0),
        })

    data_factor = min(1.0, len(observable) / 3.0)
    factor_reliability = _clamp(reliability * data_factor, 0.0, 1.0)
    confidence = round(100.0 * factor_reliability, 1)

    if candidates:
        event_scores = [item["quality"] for item in candidates]
        event_score = float(np.mean(event_scores))
        count_bonus = min(20.0, 8.0 * len(candidates))
        score = _clamp(0.72 * event_score + count_bonus, 0.0, 95.0)
        event_rate = (
            len(candidates) / (speech_seconds / 60.0) if speech_seconds > 0 else None
        )
        mean_flat = float(np.mean([item["flatness"] for item in candidates]))
        mean_rel = float(np.mean([item["rel_noise"] for item in candidates]))
        mean_low = float(np.mean([item["low_ratio"] for item in candidates]))
        explanation = (
            f"Detected {len(candidates)} low-energy, noise-like event"
            f"{'s' if len(candidates) != 1 else ''} in quiet gaps between speech. "
            "The events are above the noise floor, quieter than speech, broadband, "
            "and not strongly pitched, which is consistent with breath sounds. "
            "Playback noise, room noise or editing can produce similar events."
        )
        return {
            "key": "breathing",
            "title": "Breathing patterns",
            "status": "present",
            "status_label": _status_label("present"),
            "score": _round(score, 1),
            "confidence": confidence,
            "reliability": _round(factor_reliability, 3),
            "reliable": factor_reliability >= 0.40,
            "evidence": "human",
            "metrics": {
                "Breath-like events": str(len(candidates)),
                "Observable gaps": str(len(observable)),
                "Event rate": _fmt(event_rate, "/min"),
                "Mean level above noise": _fmt(mean_rel, " dB"),
                "Mean flatness": _fmt(mean_flat, "", 3),
                "Low-frequency energy": _fmt(mean_low * 100.0, "%"),
            },
            "explanation": explanation,
            "caveat": (
                "Breath detection is unreliable in noisy or heavily processed audio. "
                "Presence of breathing is not proof of human speech."
            ),
        }

    # No candidate found.  This is only "absent" when there were enough
    # observable gaps and the recording quality is usable.  Even then it is
    # explicitly neutral for the final verdict.
    if len(observable) >= 2 and factor_reliability >= 0.40:
        return {
            "key": "breathing",
            "title": "Breathing patterns",
            "status": "absent",
            "status_label": _status_label("absent"),
            "score": 0.0,
            "confidence": confidence,
            "reliability": _round(factor_reliability, 3),
            "reliable": True,
            "evidence": "none",
            "metrics": {
                "Breath-like events": "0",
                "Observable gaps": str(len(observable)),
                "Event rate": "0/min",
                "Mean level above noise": "—",
                "Mean flatness": "—",
                "Low-frequency energy": "—",
            },
            "explanation": (
                f"No breath-like event was found in {len(observable)} observable "
                "quiet gaps. Breathing is often quiet or absent from short clips, "
                "so this is not evidence of synthetic speech."
            ),
            "caveat": (
                "Absence of breathing is not evidence of AI generation. Many "
                "genuine recordings contain no audible breaths."
            ),
        }

    return _factor_inconclusive(
        "breathing",
        "Breathing patterns",
        "Too few usable quiet gaps or too much noise to judge breathing reliably.",
        reliability,
        metrics={
            "Internal pauses": str(len(pauses)),
            "Observable gaps": str(len(observable)),
            "Pitch-like frames": "—",
        },
        caveat=(
            "Breathing cannot be reliably detected in this recording. No score "
            "is reported rather than inventing one."
        ),
    )


# ---------------------------------------------------------------------------
# Factor 2: micro-pauses
# ---------------------------------------------------------------------------
def _analyze_micro_pauses(
    pauses: list[tuple[int, int]],
    sr: int,
    speech_seconds: float,
    analysis_seconds: float,
    quality: dict[str, Any],
) -> dict[str, Any]:
    reliability = quality["reliability"]
    if quality["status"] in {"silent", "no_speech", "unavailable"}:
        return _factor_unavailable(
            "micro_pauses",
            "Micro-pauses",
            "No usable speech was detected, so pause rhythm cannot be assessed.",
            reliability,
            caveat="Pause analysis needs detectable speech and silence.",
        )

    durations = np.array(
        [(end - start) * HOP_LENGTH / sr for start, end in pauses],
        dtype=np.float64,
    )
    durations = durations[durations >= 0.08]

    metrics = {
        "Internal pauses": str(int(durations.size)),
        "Pause rate": "—",
        "Median pause": "—",
        "Pause variability": "—",
        "Pause fraction": "—",
        "Longest pause": "—",
    }

    if durations.size < 3 or speech_seconds < 5.0:
        return _factor_inconclusive(
            "micro_pauses",
            "Micro-pauses",
            (
                "Fewer than three internal pauses (or less than five seconds of "
                "speech) were available, which is not enough to characterise rhythm."
            ),
            reliability,
            metrics=metrics,
            caveat=(
                "Short clips often contain only one or two pauses. No rhythm score "
                "is reported rather than inventing one."
            ),
        )

    rate = durations.size / (speech_seconds / 60.0)
    median_pause = float(np.median(durations))
    mean_pause = float(np.mean(durations))
    std_pause = float(np.std(durations))
    cv = std_pause / mean_pause if mean_pause > 0 else None
    pause_fraction = float(np.sum(durations) / max(analysis_seconds, 1e-9))
    longest = float(np.max(durations))

    short_fraction = float(np.mean(durations < 0.12))
    long_fraction = float(np.mean(durations > 2.0))

    components = [
        (_range_score(rate, 4.0, 30.0, 0.0, 60.0), 0.25),
        (_range_score(median_pause, 0.12, 0.90, 0.05, 2.50), 0.25),
        (_range_score(cv, 0.35, 1.80, 0.05, 3.00), 0.30),
        (_range_score(pause_fraction, 0.05, 0.35, 0.0, 0.60), 0.20),
    ]
    score = _weighted_mean(components)

    # Penalties for patterns that are common in edited or generated speech.
    if cv is not None and cv < 0.15:
        score = (score or 0.0) - 20.0
    if long_fraction > 0.20:
        score = (score or 0.0) - 20.0 * long_fraction
    if short_fraction > 0.60:
        score = (score or 0.0) - 10.0

    score = _clamp(score if score is not None else 0.0, 0.0, 85.0)
    status = _score_status(score)
    data_factor = min(1.0, durations.size / 5.0) * min(1.0, speech_seconds / 8.0)
    factor_reliability = _clamp(reliability * data_factor, 0.0, 1.0)
    confidence = round(100.0 * factor_reliability, 1)

    metrics = {
        "Internal pauses": str(int(durations.size)),
        "Pause rate": _fmt(rate, "/min"),
        "Median pause": _fmt(median_pause * 1000.0, " ms", 0),
        "Pause variability": _fmt(cv, "", 2),
        "Pause fraction": _fmt(pause_fraction * 100.0, "%"),
        "Longest pause": _fmt(longest * 1000.0, " ms", 0),
    }

    if status == "natural":
        explanation = (
            "Pause durations are varied and fall within a plausible conversational "
            "range. This is consistent with natural speech rhythm, but speaking "
            "style and editing also affect pauses."
        )
    elif status == "some_anomalies":
        explanation = (
            "Some pause statistics are outside the broad natural range. This is a "
            "weak cue, not proof of synthesis; scripted speech and editing can look "
            "similar."
        )
    else:
        explanation = (
            "Pause rhythm is unusual: the distribution is very uniform, unusually "
            "sparse, or dominated by very long gaps. This can happen in synthetic "
            "speech, but also in edited or heavily scripted recordings."
        )

    return {
        "key": "micro_pauses",
        "title": "Micro-pauses",
        "status": status,
        "status_label": _status_label(status),
        "score": _round(score, 1),
        "confidence": confidence,
        "reliability": _round(factor_reliability, 3),
        "reliable": factor_reliability >= 0.40,
        "evidence": (
            "human" if score >= 65.0
            else "synthetic" if score <= 30.0
            else "neutral"
        ),
        "metrics": metrics,
        "explanation": explanation,
        "caveat": (
            "Pause patterns depend on speaking style, language, script and editing. "
            "Synthetic speech can also contain natural pauses."
        ),
    }


# ---------------------------------------------------------------------------
# Factor 3: spectral characteristics
# ---------------------------------------------------------------------------
def _analyze_spectral(
    features: dict[str, np.ndarray],
    vad: np.ndarray,
    quality: dict[str, Any],
) -> dict[str, Any]:
    reliability = quality["reliability"]
    speech_frames = int(np.count_nonzero(vad))
    if quality["status"] in {"silent", "no_speech", "unavailable"} or speech_frames < 20:
        return _factor_unavailable(
            "spectral",
            "Spectral characteristics",
            "Not enough voiced speech was available for spectral analysis.",
            reliability,
            caveat="Spectral measurements need at least a few voiced frames.",
        )

    flatness = features["flatness"][vad]
    centroid = features["centroid"][vad]
    bandwidth = features["bandwidth"][vad]
    rolloff = features["rolloff85"][vad]
    flux = features["flux"][vad]
    contrast = features["contrast"][:, vad]
    mfcc = features["mfcc"][:, vad]
    delta = features["delta"][:, vad]

    flatness_mean = float(np.mean(flatness))
    centroid_mean = float(np.mean(centroid))
    bandwidth_mean = float(np.mean(bandwidth))
    rolloff_mean = float(np.mean(rolloff))
    flux_mean = float(np.mean(flux))
    contrast_mean = float(np.mean(contrast))

    mfcc_std_mean = float(np.mean(np.std(mfcc, axis=1)))
    delta_std_mean = float(np.mean(np.std(delta, axis=1)))
    delta_ratio = delta_std_mean / (mfcc_std_mean + 1e-9)

    components = [
        (_range_score(flatness_mean, 0.0003, 0.15, 0.0, 0.70), 0.18),
        (_range_score(contrast_mean, 8.0, 28.0, 2.0, 45.0), 0.18),
        (_range_score(centroid_mean, 400.0, 3500.0, 100.0, 6000.0), 0.12),
        (_range_score(bandwidth_mean, 600.0, 3000.0, 200.0, 5000.0), 0.10),
        (_range_score(rolloff_mean, 1000.0, 6000.0, 300.0, 7500.0), 0.10),
        (_range_score(flux_mean, 0.15, 0.85, 0.0, 1.20), 0.17),
        (_range_score(delta_ratio, 0.05, 1.20, 0.0, 3.00), 0.15),
    ]
    score = _clamp(_weighted_mean(components) or 0.0, 0.0, 85.0)
    status = _score_status(score)

    data_factor = min(1.0, speech_frames / 150.0)
    factor_reliability = _clamp(reliability * data_factor, 0.0, 1.0)
    confidence = round(100.0 * factor_reliability, 1)

    anomalies: list[str] = []
    if flatness_mean > 0.15:
        anomalies.append(
            f"the spectrum is unusually noise-like (flatness {flatness_mean:.3f})"
        )
    elif flatness_mean < 0.0003:
        anomalies.append(
            f"the spectrum is unusually tonal or clean (flatness {flatness_mean:.5f})"
        )
    if contrast_mean < 8.0:
        anomalies.append(
            f"harmonic peak-to-valley contrast is low ({contrast_mean:.1f} dB), "
            "which can indicate smeared harmonics"
        )
    elif contrast_mean > 28.0:
        anomalies.append(
            f"harmonic contrast is unusually high ({contrast_mean:.1f} dB)"
        )
    if centroid_mean < 400.0 or centroid_mean > 3500.0:
        anomalies.append(
            f"spectral centroid is outside the broad speech range ({centroid_mean:.0f} Hz)"
        )
    if flux_mean < 0.15:
        anomalies.append(
            f"spectrum changes very little between frames (flux {flux_mean:.2f}), "
            "which can sound static"
        )
    elif flux_mean > 0.85:
        anomalies.append(
            f"spectrum changes very abruptly (flux {flux_mean:.2f})"
        )
    if delta_ratio < 0.05:
        anomalies.append(
            "MFCC trajectories are unusually static (very low delta-to-static ratio)"
        )
    elif delta_ratio > 3.0:
        anomalies.append(
            "MFCC trajectories are unusually unstable (very high delta-to-static ratio)"
        )

    if anomalies:
        explanation = "Observed unusual spectral pattern(s): " + "; ".join(anomalies) + "."
    else:
        explanation = (
            "Spectral shape, harmonic contrast, noise-likeness and frame-to-frame "
            "change all fall within broad speech ranges. No strong spectral anomaly "
            "was detected."
        )

    return {
        "key": "spectral",
        "title": "Spectral characteristics",
        "status": status,
        "status_label": _status_label(status),
        "score": _round(score, 1),
        "confidence": confidence,
        "reliability": _round(factor_reliability, 3),
        "reliable": factor_reliability >= 0.40,
        "evidence": (
            "human" if score >= 65.0
            else "synthetic" if score <= 30.0
            else "neutral"
        ),
        "metrics": {
            "MFCC": f"{N_MFCC} coefficients",
            "Spectral centroid": _fmt(centroid_mean, " Hz", 0),
            "Spectral bandwidth": _fmt(bandwidth_mean, " Hz", 0),
            "Spectral rolloff": _fmt(rolloff_mean, " Hz", 0),
            "Spectral contrast": _fmt(contrast_mean, " dB"),
            "Spectral flatness": _fmt(flatness_mean, "", 3),
            "Spectral flux": _fmt(flux_mean, "", 2),
            "MFCC delta ratio": _fmt(delta_ratio, "", 2),
        },
        "explanation": explanation,
        "caveat": (
            "Spectral measurements depend on microphone, codec, noise reduction and "
            "room acoustics. They are not a classifier."
        ),
    }


# ---------------------------------------------------------------------------
# Factor 4: voice continuity
# ---------------------------------------------------------------------------
def _analyze_voice_continuity(
    features: dict[str, np.ndarray],
    vad: np.ndarray,
    sr: int,
    analysis_seconds: float,
    quality: dict[str, Any],
) -> dict[str, Any]:
    reliability = quality["reliability"]
    f0 = features["f0"]
    rms = features["rms"]
    rms_db = features["rms_db"]
    flatness = features["flatness"]

    voiced_mask = (
        vad
        & (f0 >= 60.0)
        & (f0 <= 400.0)
        & (flatness < 0.50)
    )
    voiced_frames = int(np.count_nonzero(voiced_mask))

    if quality["status"] in {"silent", "no_speech", "unavailable"} or voiced_frames < 20:
        return _factor_unavailable(
            "voice_continuity",
            "Voice continuity",
            "Not enough stable pitched speech was available for continuity analysis.",
            reliability,
            caveat="Pitch tracking needs sustained voiced speech.",
        )

    f0v = f0[voiced_mask]
    rmsv = rms[voiced_mask]
    dbv = rms_db[voiced_mask]
    median_f0 = float(np.median(f0v))
    semitone = 12.0 * np.log2(np.maximum(f0v, 1e-9) / max(median_f0, 1e-9))

    pitch_std = float(np.std(semitone))
    pitch_range = float(np.percentile(semitone, 95) - np.percentile(semitone, 5))

    voiced_indices = np.flatnonzero(voiced_mask)
    if voiced_indices.size >= 2:
        consecutive = np.diff(voiced_indices) == 1
        period = 1.0 / np.maximum(f0v, 1e-9)
        mean_period = float(np.mean(period))
        jitter_values = (
            np.abs(np.diff(period)[consecutive]) / max(mean_period, 1e-9)
            if np.any(consecutive) else np.array([])
        )
        mean_rms = float(np.mean(rmsv))
        shimmer_values = (
            np.abs(np.diff(rmsv)[consecutive]) / max(mean_rms, 1e-9)
            if np.any(consecutive) else np.array([])
        )
        semitone_diffs = np.abs(np.diff(semitone)[consecutive]) if np.any(consecutive) else np.array([])
    else:
        jitter_values = np.array([])
        shimmer_values = np.array([])
        semitone_diffs = np.array([])

    jitter = _mean(jitter_values)
    shimmer = _mean(shimmer_values)
    voiced_seconds = voiced_frames * HOP_LENGTH / sr
    jump_count = int(np.count_nonzero(semitone_diffs > 6.0))
    jump_rate = jump_count / (voiced_seconds / 60.0) if voiced_seconds > 0 else None

    voiced_segments = _mask_segments(voiced_mask, min_frames=3)
    segment_durations = np.array(
        [(end - start) * HOP_LENGTH / sr for start, end in voiced_segments],
        dtype=np.float64,
    )
    if segment_durations.size >= 2:
        segment_mean = float(np.mean(segment_durations))
        segment_cv = float(np.std(segment_durations) / segment_mean) if segment_mean > 0 else None
    else:
        segment_cv = None

    energy_std = float(np.std(dbv))

    components = [
        (_range_score(pitch_std, 1.5, 9.0, 0.2, 14.0), 0.20),
        (_range_score(pitch_range, 3.0, 18.0, 0.5, 30.0), 0.10),
        (_range_score((jitter or 0.0) * 100.0, 0.3, 6.0, 0.0, 15.0), 0.15),
        (_range_score((shimmer or 0.0) * 100.0, 2.0, 25.0, 0.0, 60.0), 0.15),
        (_upper_score(jump_rate, 6.0, 30.0), 0.15),
        (_range_score(segment_cv, 0.30, 2.00, 0.05, 3.00), 0.15),
        (_range_score(energy_std, 2.0, 18.0, 0.0, 30.0), 0.10),
    ]
    score = _clamp(_weighted_mean(components) or 0.0, 0.0, 85.0)
    status = _score_status(score)

    data_factor = min(1.0, voiced_frames / 150.0) * min(1.0, analysis_seconds / 3.0)
    factor_reliability = _clamp(reliability * data_factor, 0.0, 1.0)
    confidence = round(100.0 * factor_reliability, 1)

    anomalies: list[str] = []
    if pitch_std < 1.5:
        anomalies.append(
            f"pitch is unusually flat (variation {pitch_std:.2f} semitones)"
        )
    elif pitch_std > 9.0:
        anomalies.append(
            f"pitch varies unusually widely ({pitch_std:.2f} semitones)"
        )
    if jitter is not None and jitter < 0.003:
        anomalies.append(
            f"pitch period is unusually regular (jitter proxy {jitter * 100:.2f}%)"
        )
    elif jitter is not None and jitter > 0.06:
        anomalies.append(
            f"pitch period is unstable (jitter proxy {jitter * 100:.2f}%)"
        )
    if shimmer is not None and shimmer < 0.02:
        anomalies.append(
            f"energy is unusually steady (shimmer proxy {shimmer * 100:.1f}%)"
        )
    elif shimmer is not None and shimmer > 0.25:
        anomalies.append(
            f"energy fluctuates unusually (shimmer proxy {shimmer * 100:.1f}%)"
        )
    if jump_rate is not None and jump_rate > 6.0:
        anomalies.append(
            f"frequent large pitch jumps ({jump_rate:.1f}/min)"
        )
    if segment_cv is not None and segment_cv < 0.30:
        anomalies.append(
            "voiced segments have very uniform durations"
        )

    if anomalies:
        explanation = (
            "Observed continuity pattern(s): " + "; ".join(anomalies) + ". "
            "These are weak cues, not proof of synthesis."
        )
    else:
        explanation = (
            "Pitch variation, energy variation and voiced-segment rhythm look broadly "
            "natural. Natural speech can also be unusually steady or expressive, so "
            "this is supporting evidence only."
        )

    return {
        "key": "voice_continuity",
        "title": "Voice continuity",
        "status": status,
        "status_label": _status_label(status),
        "score": _round(score, 1),
        "confidence": confidence,
        "reliability": _round(factor_reliability, 3),
        "reliable": factor_reliability >= 0.40,
        "evidence": (
            "human" if score >= 65.0
            else "synthetic" if score <= 30.0
            else "neutral"
        ),
        "metrics": {
            "Median pitch": _fmt(median_f0, " Hz", 0),
            "Pitch variation": _fmt(pitch_std, " st", 2),
            "Pitch range": _fmt(pitch_range, " st", 2),
            "Jitter proxy": _fmt((jitter or 0.0) * 100.0, "%", 2),
            "Shimmer proxy": _fmt((shimmer or 0.0) * 100.0, "%", 1),
            "Pitch jumps": _fmt(jump_rate, "/min"),
            "Segment rhythm": _fmt(segment_cv, "", 2),
            "Energy variation": _fmt(energy_std, " dB"),
        },
        "explanation": explanation,
        "caveat": (
            "Pitch tracking is unreliable for whispering, singing, heavy noise or very "
            "short clips. Jitter and shimmer here are frame-level proxies, not clinical "
            "measurements."
        ),
    }


# ---------------------------------------------------------------------------
# Unavailable / inconclusive factor builders
# ---------------------------------------------------------------------------
def _factor_unavailable(
    key: str,
    title: str,
    explanation: str,
    reliability: float,
    caveat: str,
    metrics: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "title": title,
        "status": "unavailable",
        "status_label": _status_label("unavailable"),
        "score": None,
        "confidence": None,
        "reliability": _round(reliability, 3),
        "reliable": False,
        "evidence": "none",
        "metrics": metrics or {},
        "explanation": explanation,
        "caveat": caveat,
    }


def _factor_inconclusive(
    key: str,
    title: str,
    explanation: str,
    reliability: float,
    metrics: dict[str, str] | None = None,
    caveat: str = "",
) -> dict[str, Any]:
    return {
        "key": key,
        "title": title,
        "status": "inconclusive",
        "status_label": _status_label("inconclusive"),
        "score": None,
        "confidence": None,
        "reliability": _round(reliability, 3),
        "reliable": False,
        "evidence": "none",
        "metrics": metrics or {},
        "explanation": explanation,
        "caveat": caveat,
    }


# ---------------------------------------------------------------------------
# Fusion with the existing model
# ---------------------------------------------------------------------------
def _model_values(model_result: dict[str, Any] | None) -> tuple[str | None, float | None, float | None, str]:
    if not model_result:
        return None, None, None, ""
    prediction = str(model_result.get("prediction") or "").upper()
    if prediction not in {"BONAFIDE", "SPOOF", "UNCERTAIN"}:
        prediction = None
    score = _finite(model_result.get("spoof_score"))
    score = score / 100.0 if score is not None else None
    confidence = _finite(model_result.get("confidence"))
    confidence = confidence / 100.0 if confidence is not None else None
    reason = str(model_result.get("reason") or "").strip()
    return prediction, score, confidence, reason


def _evidence_direction(factors: dict[str, dict[str, Any]]) -> tuple[str, list[str], list[str]]:
    human_flags: list[str] = []
    synthetic_flags: list[str] = []

    for key in FACTOR_ORDER:
        factor = factors.get(key) or {}
        if not factor.get("reliable"):
            continue
        evidence = factor.get("evidence")
        score = _finite(factor.get("score"))
        if evidence == "human":
            if key == "breathing":
                if factor.get("status") == "present" and score is not None and score >= 55.0:
                    human_flags.append(key)
            elif score is not None and score >= 65.0:
                human_flags.append(key)
        elif evidence == "synthetic" and key != "breathing":
            if score is not None and score <= 30.0:
                synthetic_flags.append(key)

    if len(synthetic_flags) >= 2 and len(human_flags) <= 1:
        direction = "synthetic"
    elif len(human_flags) >= 2 and len(synthetic_flags) == 0:
        direction = "human"
    elif human_flags and synthetic_flags:
        direction = "mixed"
    else:
        direction = "inconclusive"
    return direction, human_flags, synthetic_flags


def _agreement(model_prediction: str | None, direction: str) -> str:
    if model_prediction is None:
        return "inconclusive"
    if model_prediction == "SPOOF":
        if direction == "human":
            return "conflicting"
        if direction == "synthetic":
            return "consistent"
        return "partially_consistent"
    if model_prediction == "BONAFIDE":
        if direction == "synthetic":
            return "conflicting"
        if direction == "human":
            return "consistent"
        return "partially_consistent"
    return "inconclusive"


def _factor_sentence(key: str, factor: dict[str, Any]) -> str:
    title = factor.get("title", key)
    status = factor.get("status", "inconclusive")
    score = _finite(factor.get("score"))
    if score is None:
        return f"{title}: {status}"
    return f"{title}: {status} (score {score:.0f}/100)"


def _fuse(
    model_result: dict[str, Any] | None,
    factors: dict[str, dict[str, Any]],
    quality: dict[str, Any],
) -> dict[str, Any]:
    model_prediction, model_score, model_confidence, model_reason = _model_values(model_result)
    direction, human_flags, synthetic_flags = _evidence_direction(factors)
    agreement = _agreement(model_prediction, direction)

    # Quality overrides: a silent or unusable recording cannot support a verdict.
    if quality["status"] in {"silent", "no_speech"}:
        final = "UNCERTAIN"
        confidence = 10.0
        reason = (
            "The recording contains little or no detectable speech, so the model "
            "score is not reliable. The final assessment is UNCERTAIN rather than "
            "a verdict."
        )
    elif quality["status"] == "too_short":
        final = "UNCERTAIN"
        confidence = 20.0
        reason = (
            "The recording is too short for a reliable acoustic assessment. The "
            "model score is reported for reference, but the final assessment is "
            "UNCERTAIN."
        )
    elif model_prediction is None:
        final = "UNCERTAIN"
        confidence = 15.0
        reason = (
            "No model score was available. Acoustic evidence alone is not sufficient "
            "for a definitive human or AI verdict."
        )
    else:
        final = model_prediction
        confidence = (model_confidence or max(model_score or 0.5, 1.0 - (model_score or 0.5))) * 100.0

        if model_prediction == "SPOOF":
            if direction == "human" and len(human_flags) >= 3 and (model_score or 0.0) < 0.90:
                final = "UNCERTAIN"
                reason = (
                    "The trained model returned SPOOF, but multiple independent "
                    "acoustic cues are human-like. The evidence conflicts, so the "
                    "final assessment is UNCERTAIN rather than a confident spoof call."
                )
            elif direction == "human":
                reason = (
                    "The trained model returned SPOOF. Some broad acoustic cues look "
                    "human-like, but they are not strong enough (or the model score is "
                    "too high) to overturn the model. The model remains primary."
                )
            else:
                reason = (
                    "The trained model returned SPOOF and the acoustic evidence does "
                    "not strongly contradict it."
                )
        elif model_prediction == "BONAFIDE":
            if direction == "synthetic" and len(synthetic_flags) >= 2 and (model_score or 0.0) > 0.20:
                final = "UNCERTAIN"
                reason = (
                    "The trained model returned BONAFIDE, but multiple independent "
                    "acoustic cues are unusual. The evidence conflicts, so the final "
                    "assessment is UNCERTAIN rather than a confident bonafide call."
                )
            elif direction == "synthetic":
                reason = (
                    "The trained model returned BONAFIDE. One acoustic cue is unusual, "
                    "but a single cue is not enough to overturn the model."
                )
            else:
                reason = (
                    "The trained model returned BONAFIDE and the acoustic evidence is "
                    "consistent with that result."
                )
        else:  # UNCERTAIN
            if direction == "synthetic" and len(synthetic_flags) >= 3 and (model_score or 0.0) >= 0.65:
                final = "SPOOF"
                reason = (
                    "The model score was in the uncertain band, but three independent "
                    "acoustic cues were strongly unusual. The final assessment leans "
                    "SPOOF; this remains an estimate, not proof."
                )
            elif direction == "human" and len(human_flags) >= 3 and (model_score or 1.0) <= 0.35:
                final = "BONAFIDE"
                reason = (
                    "The model score was in the uncertain band, but three independent "
                    "acoustic cues were strongly human-like. The final assessment leans "
                    "BONAFIDE; this remains an estimate, not proof."
                )
            else:
                reason = (
                    "The model score is inside the uncertain band and the acoustic "
                    "evidence is not strong enough to resolve it. The final assessment "
                    "remains UNCERTAIN."
                )

        if final == model_prediction:
            if agreement == "consistent":
                confidence += 4.0
            elif agreement == "conflicting":
                confidence -= 18.0
        else:
            confidence -= 25.0

        if quality["reliability"] < 0.50:
            confidence -= 10.0

        confidence = _clamp(confidence, 5.0, 99.0)
        if final == "UNCERTAIN":
            confidence = min(confidence, 55.0)

    confidence_label = (
        "Low" if final == "UNCERTAIN"
        else "High" if confidence >= 75.0
        else "Medium" if confidence >= 55.0
        else "Low"
    )

    factor_sentences = [
        _factor_sentence(key, factors[key])
        for key in FACTOR_ORDER
        if key in factors
    ]
    acoustic_summary = "; ".join(factor_sentences)
    model_text = (
        f"Model prediction: {model_prediction}"
        + (f" (spoof score {model_score * 100.0:.1f}%)" if model_score is not None else "")
    )
    if model_reason:
        reason = f"{reason} {model_reason}"
    reason = f"{reason} Acoustic evidence — {acoustic_summary}."

    summary = (
        f"{model_text}. Acoustic direction: {direction}. "
        f"Final assessment: {final}."
    )

    return {
        "final_prediction": final,
        "model_prediction": model_prediction or "UNCERTAIN",
        "model_spoof_score": _round((model_score or 0.0) * 100.0, 2) if model_score is not None else None,
        "evidence_direction": direction,
        "agreement": agreement,
        "confidence": _round(confidence, 1),
        "confidence_label": confidence_label,
        "reason": reason,
        "summary": summary,
        "human_flags": human_flags,
        "synthetic_flags": synthetic_flags,
        "changed_from_model": bool(model_prediction and final != model_prediction),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def analyze_human_voice(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    model_result: dict[str, Any] | None = None,
    max_seconds: float = MAX_ANALYSIS_SECONDS,
) -> dict[str, Any]:
    """Analyse one recording and return the four factor results plus a fusion.

    ``model_result`` is the dictionary returned by
    ``audio_features.predict_audio_files``.  It is used as the primary evidence;
    the acoustic factors are supplementary.
    """
    y = np.asarray(waveform, dtype=np.float32).ravel()
    if y.size == 0:
        return _unavailable_result("The audio is empty.", model_result)

    if int(sr) != SAMPLE_RATE:
        y = librosa.resample(
            y, orig_sr=int(sr), target_sr=SAMPLE_RATE, res_type="soxr_hq"
        )
        sr = SAMPLE_RATE

    original_duration = len(y) / float(sr)
    if original_duration < MIN_SECONDS:
        return _unavailable_result(
            f"The recording is shorter than {MIN_SECONDS:.2f} seconds.",
            model_result,
            duration=original_duration,
        )

    if len(y) > int(max_seconds * sr):
        y = y[: int(max_seconds * sr)]
    analysis_seconds = len(y) / float(sr)

    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        features = _frame_features(y, sr)
        vad, vad_info = _voice_activity(features["rms_db"])
    except Exception as exc:  # noqa: BLE001 - analysis must never break prediction
        return _unavailable_result(
            f"Acoustic analysis could not be completed: {exc}", model_result,
            duration=original_duration,
        )

    speech_frames = int(np.count_nonzero(vad))
    speech_seconds = speech_frames * HOP_LENGTH / float(sr)
    speech_fraction = speech_seconds / max(analysis_seconds, 1e-9)
    dynamic_range = float(vad_info["dynamic_range_db"])
    noise_floor = float(vad_info["noise_floor_db"])
    speech_level = float(vad_info["speech_level_db"])

    clipping_fraction = float(np.mean(np.abs(y) >= 0.999))
    effective_bandwidth = _median(features["rolloff99"][vad]) if speech_frames else None

    if speech_fraction < 0.02 or (dynamic_range < 3.0 and speech_level < -45.0):
        quality_status = "silent"
    elif speech_fraction < 0.08:
        quality_status = "no_speech"
    elif original_duration < 1.0:
        quality_status = "too_short"
    elif dynamic_range < 6.0:
        quality_status = "low_snr"
    elif clipping_fraction > 0.02:
        quality_status = "clipped"
    elif effective_bandwidth is not None and effective_bandwidth < 3500.0:
        quality_status = "limited_bandwidth"
    else:
        quality_status = "good"

    reliability = 1.0
    if original_duration < 5.0:
        reliability *= max(0.35, original_duration / 5.0)
    if speech_seconds < 3.0:
        reliability *= max(0.40, speech_seconds / 3.0)
    if speech_fraction < 0.15:
        reliability *= 0.70
    if dynamic_range < 12.0:
        reliability *= 0.75
    if clipping_fraction > 0.01:
        reliability *= 0.80
    if effective_bandwidth is not None and effective_bandwidth < 3500.0:
        reliability *= 0.80
    if quality_status == "silent":
        reliability = min(reliability, 0.05)
    elif quality_status == "no_speech":
        reliability = min(reliability, 0.15)
    elif quality_status == "too_short":
        reliability = min(reliability, 0.45)
    elif quality_status == "low_snr":
        reliability = min(reliability, 0.50)
    reliability = _clamp(reliability, 0.0, 1.0)

    quality = {
        "status": quality_status,
        "reliability": reliability,
        "reliability_label": _reliability_label(reliability),
        "duration_seconds": _round(original_duration, 2),
        "analysed_seconds": _round(analysis_seconds, 2),
        "speech_seconds": _round(speech_seconds, 2),
        "speech_fraction": _round(speech_fraction, 3),
        "dynamic_range_db": _round(dynamic_range, 1),
        "noise_floor_db": _round(noise_floor, 1),
        "speech_level_db": _round(speech_level, 1),
        "clipping_fraction": _round(clipping_fraction, 4),
        "effective_bandwidth_hz": _round(effective_bandwidth, 0),
        "notes": [],
    }
    if original_duration > max_seconds:
        quality["notes"].append(
            f"Only the first {max_seconds:.0f} seconds were analysed acoustically."
        )
    if quality_status == "limited_bandwidth":
        quality["notes"].append(
            "The recording is narrowband (effective bandwidth below 3.5 kHz); "
            "spectral cues are less reliable."
        )
    if quality_status == "low_snr":
        quality["notes"].append(
            "The energy dynamic range is low; background noise may affect the cues."
        )
    if quality_status == "clipped":
        quality["notes"].append(
            "The waveform contains clipped samples; some measurements may be distorted."
        )

    # Speech segmentation and pauses.
    speech_segments = _mask_segments(vad, min_frames=3)
    pauses = _internal_pauses(speech_segments, min_frames=int(0.08 * sr / HOP_LENGTH))

    factors = {
        "breathing": _analyze_breathing(
            features, pauses, sr, speech_seconds, quality
        ),
        "micro_pauses": _analyze_micro_pauses(
            pauses, sr, speech_seconds, analysis_seconds, quality
        ),
        "spectral": _analyze_spectral(features, vad, quality),
        "voice_continuity": _analyze_voice_continuity(
            features, vad, sr, analysis_seconds, quality
        ),
    }

    fusion = _fuse(model_result, factors, quality)

    return {
        "available": True,
        "version": "1.0",
        "analysis_seconds": _round(analysis_seconds, 2),
        "quality": quality,
        "factors": factors,
        "fusion": fusion,
        "disclaimer": DISCLAIMER,
    }


def _unavailable_result(
    reason: str,
    model_result: dict[str, Any] | None,
    duration: float | None = None,
) -> dict[str, Any]:
    quality = {
        "status": "unavailable",
        "reliability": 0.0,
        "reliability_label": "Low",
        "duration_seconds": _round(duration, 2) if duration is not None else None,
        "analysed_seconds": 0.0,
        "speech_seconds": 0.0,
        "speech_fraction": 0.0,
        "dynamic_range_db": None,
        "noise_floor_db": None,
        "speech_level_db": None,
        "clipping_fraction": None,
        "effective_bandwidth_hz": None,
        "notes": [reason],
    }
    factors = {
        key: _factor_unavailable(
            key,
            {
                "breathing": "Breathing patterns",
                "micro_pauses": "Micro-pauses",
                "spectral": "Spectral characteristics",
                "voice_continuity": "Voice continuity",
            }[key],
            reason,
            0.0,
            caveat="No acoustic score is reported for this recording.",
        )
        for key in FACTOR_ORDER
    }
    fusion = _fuse(model_result, factors, quality)
    return {
        "available": False,
        "version": "1.0",
        "analysis_seconds": 0.0,
        "quality": quality,
        "factors": factors,
        "fusion": fusion,
        "disclaimer": DISCLAIMER,
    }


def factor_list(analysis: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return factors in display order (used by templates and reports)."""
    if not analysis:
        return []
    factors = analysis.get("factors") or {}
    return [factors[key] for key in FACTOR_ORDER if key in factors]

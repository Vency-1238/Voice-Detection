"""
Shared audio feature extraction and prediction for the Audio Deepfake Detector.

This module is imported by BOTH the training script/notebook and the Flask app, so
the feature vector used to train the model is exactly the one used at prediction
time.  That rule has not changed.

What is different from the first version
----------------------------------------
* One STFT is calculated per chunk and reused for MFCC, delta-MFCC and every
  spectral feature.  The old code calculated the STFT five separate times
  (once inside MFCC, once for centroid, once for bandwidth, once for rolloff,
  and again indirectly for other features).
* RMS and zero-crossing rate share one vectorised time-domain framing pass.
* Audio is decoded with soundfile first (fast) and librosa is used only as a
  fallback, so WAV/FLAC/MP3/OGG all work.
* Several optional feature groups can be switched on for experiments without
  changing the code.  The default `base90` layout is identical to the original
  90-value vector, so the old model can still be loaded and compared.
* Long files are split into chunks, all chunk vectors are predicted in ONE
  batch, and the chunk scores are combined with duration/energy weights.
* A calibrated probability, a spoof threshold and a separate uncertain band
  are supported.  A raw model score is never presented as a true probability.

Run `python audio_features.py` to sanity-check the module.
"""

from functools import lru_cache
from pathlib import Path

import joblib
import librosa
import numpy as np

# ---------------------------------------------------------------------------
# Analysis settings - these MUST stay the same for training and prediction
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000       # ASVspoof 2019 LA recordings are 16 kHz
CHUNK_SECONDS = 5.0       # training uses the first 5 s; prediction uses 5 s chunks
N_MFCC = 20
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
DELTA_WIDTH = 9
MIN_SECONDS = 0.25        # shorter audio cannot produce stable features
MIN_CHUNK_SECONDS = 1.0   # a shorter tail is merged into the previous chunk
MAX_ANALYSIS_SECONDS = 600.0  # safety cap for very large uploads

# Feature groups.  `base90` reproduces the original vector exactly.
FEATURE_SETS = {
    "base90": ("mfcc", "delta", "centroid", "bandwidth", "rolloff", "zcr", "rms"),
    "flat_flux": ("mfcc", "delta", "centroid", "bandwidth", "rolloff", "zcr", "rms",
                  "flatness", "flux"),
    "delta2": ("mfcc", "delta", "delta2", "centroid", "bandwidth", "rolloff", "zcr", "rms"),
    "flat_flux_delta2": ("mfcc", "delta", "delta2", "centroid", "bandwidth", "rolloff",
                         "zcr", "rms", "flatness", "flux"),
    "contrast": ("mfcc", "delta", "centroid", "bandwidth", "rolloff", "zcr", "rms",
                 "contrast"),
    "flat_flux_contrast": ("mfcc", "delta", "centroid", "bandwidth", "rolloff", "zcr",
                           "rms", "flatness", "flux", "contrast"),
    "full": ("mfcc", "delta", "delta2", "centroid", "bandwidth", "rolloff", "zcr",
             "rms", "flatness", "flux", "contrast", "chroma"),
}
DEFAULT_FEATURE_SET = "base90"

# Friendly names for the scalar spectral groups.
_SCALAR_GROUPS = {
    "centroid": "spectral_centroid",
    "bandwidth": "spectral_bandwidth",
    "rolloff": "spectral_rolloff",
    "zcr": "zero_crossing_rate",
    "rms": "rms",
    "flatness": "spectral_flatness",
    "flux": "spectral_flux",
}
_CONTRAST_BANDS = 7
_CHROMA_BINS = 12


def _resolve_feature_set(feature_set):
    """Accept a set name, a tuple/list of group names, or None."""
    if feature_set is None:
        feature_set = DEFAULT_FEATURE_SET
    if isinstance(feature_set, str):
        if feature_set not in FEATURE_SETS:
            raise ValueError(
                f"Unknown feature set '{feature_set}'. Choose from {sorted(FEATURE_SETS)}."
            )
        return FEATURE_SETS[feature_set]
    return tuple(feature_set)


@lru_cache(maxsize=32)
def _feature_names_cached(groups):
    names = []
    for group in groups:
        if group in {"mfcc", "delta", "delta2"}:
            prefix = {"mfcc": "mfcc", "delta": "delta_mfcc", "delta2": "delta2_mfcc"}[group]
            for statistic in ("mean", "std"):
                for coefficient in range(N_MFCC):
                    names.append(f"{prefix}_{statistic}_{coefficient}")
        elif group in _SCALAR_GROUPS:
            name = _SCALAR_GROUPS[group]
            names.append(f"{name}_mean")
            names.append(f"{name}_std")
        elif group == "contrast":
            for band in range(_CONTRAST_BANDS):
                names.append(f"spectral_contrast_mean_{band}")
            for band in range(_CONTRAST_BANDS):
                names.append(f"spectral_contrast_std_{band}")
        elif group == "chroma":
            for band in range(_CHROMA_BINS):
                names.append(f"chroma_mean_{band}")
            for band in range(_CHROMA_BINS):
                names.append(f"chroma_std_{band}")
        else:
            raise ValueError(f"Unknown feature group: {group}")
    return tuple(names)


def feature_names(feature_set=None):
    """Return feature names in the exact order used by the vector."""
    return list(_feature_names_cached(_resolve_feature_set(feature_set)))


def infer_feature_set(n_features):
    """Map a stored feature count back to a feature-set name (for old models)."""
    for name in FEATURE_SETS:
        if len(feature_names(name)) == n_features:
            return name
    return DEFAULT_FEATURE_SET


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def _friendly_decode_error(path, exc):
    hint = ""
    if Path(path).suffix.lower() in {".webm", ".m4a", ".oga"}:
        hint = (
            " Browser-recording containers (WebM/M4A) need ffmpeg: install it and "
            "put it on PATH, or convert the clip to WAV."
        )
    return ValueError(
        f"Could not decode '{Path(path).name}'. It may be corrupted, empty, "
        f"or not a real audio file.{hint}"
    )


def load_audio(audio_path, sr=SAMPLE_RATE, max_seconds=None):
    """Load any supported file as mono float32 at the requested sample rate.

    Tries soundfile first because it is much faster for WAV/FLAC/OGG and modern
    libsndfile builds also decode MP3.  Falls back to librosa (audioread) for
    formats the local libsndfile build cannot handle.
    """
    path = Path(audio_path)
    if not path.exists():
        raise ValueError(f"Audio file not found: {path.name}")

    data = None
    orig_sr = None

    try:
        import soundfile as sf

        info = sf.info(str(path))
        frames = -1
        if max_seconds is not None:
            frames = max(1, int(float(max_seconds) * info.samplerate))
            frames = min(frames, info.frames) if info.frames > 0 else frames
        data, orig_sr = sf.read(
            str(path), frames=frames, dtype="float32", always_2d=True
        )
    except Exception:
        # MP3/OGG on old libsndfile builds, or an unusual container.
        try:
            data, orig_sr = librosa.load(
                str(path), sr=None, mono=False, duration=max_seconds
            )
            if isinstance(data, np.ndarray) and data.ndim == 1:
                data = data[:, None]
            elif isinstance(data, np.ndarray):
                data = data.T
        except Exception as exc:  # noqa: BLE001 - user-facing message
            raise _friendly_decode_error(path, exc) from exc

    if data is None or len(data) == 0:
        raise ValueError("The audio file is empty.")

    if data.ndim == 2 and data.shape[1] > 1:
        y = data.mean(axis=1)  # stereo -> mono
    else:
        y = data.reshape(-1)

    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError("The audio file is empty.")

    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if int(orig_sr) != int(sr):
        y = librosa.resample(
            y, orig_sr=int(orig_sr), target_sr=int(sr), res_type="soxr_hq"
        )

    y = np.ascontiguousarray(y, dtype=np.float32)

    if max_seconds is not None:
        limit = int(float(max_seconds) * sr)
        if y.size > limit:
            y = y[:limit]

    return y, int(sr)


def audio_duration(audio_path):
    """Return duration in seconds without decoding the whole file when possible."""
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        y, sr = load_audio(audio_path)
        return float(len(y)) / float(sr)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _delta(matrix, order=1):
    """delta/delta-delta that also works on very short clips."""
    frames = matrix.shape[1]
    if frames >= DELTA_WIDTH:
        return librosa.feature.delta(matrix, width=DELTA_WIDTH, order=order)

    width = frames if frames % 2 == 1 else frames - 1
    if width < 3:
        return np.zeros_like(matrix)
    return librosa.feature.delta(matrix, width=width, order=order)


def _time_domain_features(y):
    """RMS and zero-crossing rate from one shared framing pass.

    This reproduces librosa's centered framing with constant padding, so the
    values are numerically equivalent to librosa.feature.rms(y=y) and
    librosa.feature.zero_crossing_rate(y) while avoiding two separate passes.
    """
    pad = N_FFT // 2
    padded = np.pad(y, (pad, pad), mode="constant")
    if padded.size < N_FFT:
        padded = np.pad(padded, (0, N_FFT - padded.size), mode="constant")

    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]
    if frames.size == 0:
        return np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)

    energy = np.mean(frames * frames, axis=1, keepdims=True)
    rms = np.sqrt(np.maximum(energy, 0.0)).astype(np.float32)

    signs = np.signbit(frames)
    crossings = np.abs(np.diff(signs.astype(np.int8), axis=1))
    zcr = np.mean(crossings, axis=1, keepdims=True).astype(np.float32)

    # librosa returns (1, n_frames); keep the same orientation so mean(axis=1)
    # gives one value per feature block.
    return rms.T, zcr.T


@lru_cache(maxsize=4)
def _fft_frequencies():
    return librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT).astype(np.float64)


@lru_cache(maxsize=4)
def _mel_filterbank():
    # float32 keeps the matmul in the same precision as the audio and is ~2x
    # faster here; the resulting MFCCs differ from float64 by ~1e-7 relative.
    return librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS
    ).astype(np.float32)


def _spectral_features(magnitude, groups):
    """Calculate only the spectral groups the caller asked for.

    Centroid, bandwidth, rolloff and flatness are implemented with vectorised
    NumPy from the single STFT magnitude.  This is mathematically the same as
    their librosa counterparts but avoids repeatedly entering Python helpers.
    """
    values = {}
    if not groups.intersection(
        {"centroid", "bandwidth", "rolloff", "flatness", "flux", "contrast", "chroma"}
    ):
        return values

    frequencies = _fft_frequencies()
    magnitude64 = magnitude.astype(np.float64, copy=False)
    total_energy = np.sum(magnitude64, axis=0)
    safe_energy = np.maximum(total_energy, np.finfo(np.float64).eps)

    # einsum avoids OpenBLAS's large-thread-count overhead on these small
    # matrix-vector products.  On Windows OpenBLAS may spin up 20+ threads and
    # make a sub-millisecond GEMV take tens of milliseconds.
    if "centroid" in groups or "bandwidth" in groups:
        mean_frequency = (
            np.einsum("f,ft->t", frequencies, magnitude64, optimize=True) / safe_energy
        )
    if "centroid" in groups:
        values["centroid"] = mean_frequency[None, :]
    if "bandwidth" in groups:
        mean_square_frequency = (
            np.einsum(
                "f,ft->t", np.square(frequencies), magnitude64, optimize=True
            ) / safe_energy
        )
        variance = np.maximum(mean_square_frequency - np.square(mean_frequency), 0.0)
        values["bandwidth"] = np.sqrt(variance)[None, :]
    if "rolloff" in groups:
        cumulative = np.cumsum(magnitude64, axis=0)
        threshold = 0.85 * cumulative[-1]
        index = np.argmax(cumulative >= threshold, axis=0)
        values["rolloff"] = frequencies[index][None, :]
    if "flatness" in groups:
        # librosa's default is power=2.0, so flatness is measured on the power
        # spectrum rather than the magnitude spectrum.
        amin = 1e-10
        clipped = np.maximum(np.square(magnitude64), amin)
        geometric = np.exp(np.mean(np.log(clipped), axis=0))
        arithmetic = np.mean(clipped, axis=0)
        values["flatness"] = (geometric / arithmetic)[None, :]
    if "flux" in groups:
        if magnitude64.shape[1] > 1:
            flux = np.mean(np.abs(np.diff(magnitude64, axis=1)), axis=0, keepdims=True)
            flux = np.concatenate(
                [np.zeros((1, 1), dtype=magnitude64.dtype), flux], axis=1
            )
        else:
            flux = np.zeros((1, magnitude64.shape[1]), dtype=magnitude64.dtype)
        values["flux"] = flux
    if "contrast" in groups:
        values["contrast"] = librosa.feature.spectral_contrast(
            S=magnitude, sr=SAMPLE_RATE, n_bands=6, fmin=200.0
        )
    if "chroma" in groups:
        values["chroma"] = librosa.feature.chroma_stft(
            S=magnitude * magnitude, sr=SAMPLE_RATE, tuning=0.0
        )
    return values


def extract_features_from_waveform(y, sr=SAMPLE_RATE, feature_set=None):
    """Turn a mono waveform into a fixed-length feature vector.

    `feature_set` defaults to `base90`, the original 90-value layout.  The
    training script can select a richer set; the bundle records which one was
    used, and prediction automatically uses the same one.
    """
    groups = _resolve_feature_set(feature_set)
    y = np.asarray(y, dtype=np.float32).ravel()

    if y.size == 0:
        raise ValueError("The audio file is empty.")

    if int(sr) != SAMPLE_RATE:
        y = librosa.resample(
            y, orig_sr=int(sr), target_sr=SAMPLE_RATE, res_type="soxr_hq"
        )
        sr = SAMPLE_RATE

    if y.size < int(MIN_SECONDS * SAMPLE_RATE):
        raise ValueError(
            f"Audio is too short: at least {MIN_SECONDS} seconds are required."
        )

    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # ONE STFT, reused by MFCC and every spectral feature.
    magnitude = np.abs(
        librosa.stft(
            y,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=N_FFT,
            window="hann",
            center=True,
            pad_mode="constant",
        )
    )
    power = magnitude * magnitude

    blocks = []

    if any(group in groups for group in ("mfcc", "delta", "delta2")):
        # Cached filterbank + one einsum.  Do NOT use `@` here: with the default
        # 20+ OpenBLAS threads, the small (128 x 1025) @ (1025 x frames) product
        # can take hundreds of milliseconds because thread overhead dominates.
        # einsum does the same arithmetic in ~2 ms and agrees with librosa's
        # melspectrogram to ~1e-7 relative.
        mel = np.einsum(
            "mf,ft->mt", _mel_filterbank(), power, optimize=True
        )
        log_mel = librosa.power_to_db(mel, ref=1.0)
        mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC)
    else:
        mfcc = None

    for group in groups:
        if group == "mfcc":
            blocks.append(np.mean(mfcc, axis=1))
            blocks.append(np.std(mfcc, axis=1))
        elif group == "delta":
            delta = _delta(mfcc, order=1)
            blocks.append(np.mean(delta, axis=1))
            blocks.append(np.std(delta, axis=1))
        elif group == "delta2":
            delta2 = _delta(mfcc, order=2)
            blocks.append(np.mean(delta2, axis=1))
            blocks.append(np.std(delta2, axis=1))

    if any(group in groups for group in ("zcr", "rms")):
        rms, zcr = _time_domain_features(y)
    else:
        rms = zcr = None

    spectral = _spectral_features(magnitude, set(groups))

    for group in groups:
        if group in _SCALAR_GROUPS:
            if group == "rms":
                matrix = rms
            elif group == "zcr":
                matrix = zcr
            else:
                matrix = spectral[group]
            blocks.append(np.mean(matrix, axis=1))
            blocks.append(np.std(matrix, axis=1))
        elif group == "contrast":
            matrix = spectral["contrast"]
            blocks.append(np.mean(matrix, axis=1))
            blocks.append(np.std(matrix, axis=1))
        elif group == "chroma":
            matrix = spectral["chroma"]
            blocks.append(np.mean(matrix, axis=1))
            blocks.append(np.std(matrix, axis=1))

    vector = np.concatenate([np.asarray(block, dtype=np.float32).ravel() for block in blocks])

    expected = len(feature_names(groups))
    if vector.size != expected:
        raise RuntimeError(
            f"Feature extraction returned {vector.size} values, expected {expected}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("Could not extract valid audio features from this file.")

    return vector


def extract_features_from_file(audio_path, max_seconds=CHUNK_SECONDS, feature_set=None):
    """Load a file and extract one feature vector (first `max_seconds`)."""
    y, sr = load_audio(audio_path, max_seconds=max_seconds)
    return extract_features_from_waveform(y, sr, feature_set=feature_set)


# ---------------------------------------------------------------------------
# Chunking and prediction
# ---------------------------------------------------------------------------
def split_into_chunks(y, sr=SAMPLE_RATE, chunk_seconds=CHUNK_SECONDS,
                      min_chunk_seconds=MIN_CHUNK_SECONDS):
    """Split audio into chunks and merge a tiny tail into the previous chunk.

    Returns a list of (chunk_waveform, start_sample, duration_seconds).
    """
    chunk_size = max(1, int(float(chunk_seconds) * sr))
    min_size = max(1, int(float(min_chunk_seconds) * sr))

    chunks = []
    for start in range(0, len(y), chunk_size):
        chunk = y[start:start + chunk_size]
        if len(chunk) < min_size and chunks:
            previous, previous_start = chunks[-1]
            # Merge the tail, but never let the merged chunk grow without bound.
            if len(previous) + len(chunk) <= chunk_size + min_size:
                chunks[-1] = (np.concatenate([previous, chunk]), previous_start)
            continue
        chunks.append((chunk, start))

    return [
        (chunk, start, len(chunk) / float(sr))
        for chunk, start in chunks
        if len(chunk) > 0
    ]


def _apply_calibrator(probabilities, calibrator):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if calibrator is None:
        return probabilities
    try:
        if isinstance(calibrator, dict) and calibrator.get("kind") == "platt":
            # Platt scaling stored as plain numbers, so it loads without any
            # extra class or library version dependency.
            clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
            logit = np.log(clipped / (1.0 - clipped))
            z = float(calibrator["coef"]) * logit + float(calibrator["intercept"])
            return 1.0 / (1.0 + np.exp(-z))
        return np.asarray(calibrator.predict(probabilities), dtype=np.float64)
    except Exception:
        # A broken calibrator must never take the app down.
        return probabilities


def _aggregate_chunk_scores(probabilities, durations, rms_values, bundle):
    """Combine chunk scores without letting a tiny/noisy chunk dominate."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    rms_values = np.asarray(rms_values, dtype=np.float64)

    max_rms = float(np.max(rms_values)) if rms_values.size else 0.0
    if max_rms > 0:
        relative_energy = np.clip(rms_values / max_rms, 0.0, 1.0)
    else:
        relative_energy = np.ones_like(rms_values)

    # Duration matters, energy matters, but neither can explode the weight.
    chunk_seconds = float(bundle.get("chunk_seconds", CHUNK_SECONDS))
    weight = np.minimum(durations, chunk_seconds) * (0.5 + 0.5 * relative_energy)
    weight = np.maximum(weight, 1e-6)

    score = float(np.average(probabilities, weights=weight))

    if probabilities.size >= 2:
        variance = float(np.average((probabilities - score) ** 2, weights=weight))
        spread = float(np.sqrt(max(variance, 0.0)))
        score_range = float(np.max(probabilities) - np.min(probabilities))
    else:
        spread = 0.0
        score_range = 0.0

    return score, spread, score_range, weight


def _decision(score, spread, score_range, bundle):
    spoof_threshold = float(bundle.get("spoof_threshold", bundle.get("threshold", 0.5)))
    bonafide_threshold = float(bundle.get("bonafide_threshold", 0.5))

    if score >= spoof_threshold:
        prediction = "SPOOF"
        reason = "Spoof score is above the spoof threshold."
    elif score <= bonafide_threshold:
        prediction = "BONAFIDE"
        reason = "Spoof score is below the bonafide threshold."
    else:
        prediction = "UNCERTAIN"
        reason = "Spoof score falls inside the uncertain band."

    # Chunks that disagree strongly are a reliability warning, not a fake chunk.
    if prediction == "SPOOF" and score < 0.99 and (spread > 0.30 or score_range > 0.60):
        prediction = "UNCERTAIN"
        reason = "Chunks disagree strongly; the recording may contain mixed content."

    return prediction, reason, spoof_threshold, bonafide_threshold


def _confidence_label(prediction, score):
    if prediction == "UNCERTAIN":
        return "Low"
    distance = abs(score - 0.5) * 2.0
    if distance >= 0.6:
        return "High"
    if distance >= 0.3:
        return "Medium"
    return "Low"


def _validate_bundle(bundle, feature_set):
    names = feature_names(feature_set)
    if bundle.get("feature_names"):
        stored = list(bundle["feature_names"])
        if stored != names:
            raise RuntimeError(
                "This model was trained with a different feature layout. "
                "Retrain it with the current audio_features.py."
            )
    model = bundle["model"]
    if hasattr(model, "n_features_in_") and int(model.n_features_in_) != len(names):
        raise RuntimeError(
            f"Model expects {model.n_features_in_} features but the extractor "
            f"produces {len(names)}. Retrain the model."
        )


def predict_audio_files(audio_paths, model_bundle, progress_callback=None):
    """Predict several files efficiently.

    All chunks from all files are collected and passed to `predict_proba` in a
    single batch.  One bad file only produces an error entry; the rest of the
    batch continues.

    Returns a list of result dictionaries in the same order as `audio_paths`.
    """
    model = model_bundle["model"]
    feature_set = model_bundle.get("feature_set")
    if feature_set is None:
        feature_set = infer_feature_set(int(model_bundle.get("n_features", 90)))
    feature_set = _resolve_feature_set(feature_set)

    _validate_bundle(model_bundle, feature_set)

    chunk_seconds = float(model_bundle.get("chunk_seconds", CHUNK_SECONDS))
    calibrator = model_bundle.get("calibrator")

    paths = [Path(path) for path in audio_paths]
    total = len(paths)

    all_rows = []
    file_records = []
    results = [None] * total

    for index, path in enumerate(paths):
        record = {
            "path": path,
            "error": None,
            "chunks": [],
            "duration_seconds": 0.0,
            "truncated": False,
        }
        file_records.append(record)

        try:
            y, _sr = load_audio(path, max_seconds=MAX_ANALYSIS_SECONDS)
            if y.size < int(MIN_SECONDS * SAMPLE_RATE):
                raise ValueError(
                    f"Audio is too short: at least {MIN_SECONDS} seconds are required."
                )

            full_duration = len(y) / float(SAMPLE_RATE)
            record["duration_seconds"] = full_duration
            record["truncated"] = full_duration >= MAX_ANALYSIS_SECONDS - 0.01

            chunks = split_into_chunks(y, SAMPLE_RATE, chunk_seconds)
            if not chunks:
                raise ValueError("No usable audio chunks were found.")

            for start_index, (chunk, start_sample, duration) in enumerate(chunks):
                try:
                    vector = extract_features_from_waveform(
                        chunk, SAMPLE_RATE, feature_set=feature_set
                    )
                except Exception:
                    continue  # one bad chunk must not kill the file
                row_index = len(all_rows)
                all_rows.append(vector)
                rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
                record["chunks"].append({
                    "row": row_index,
                    "index": start_index,
                    "start_seconds": round(start_sample / SAMPLE_RATE, 2),
                    "duration_seconds": round(duration, 2),
                    "rms": rms,
                })

            if not record["chunks"]:
                raise ValueError("Could not extract features from any part of this file.")

        except Exception as exc:  # noqa: BLE001 - per-file error reporting
            record["error"] = str(exc).strip() or exc.__class__.__name__

        if progress_callback is not None:
            progress_callback(index + 1, total, path.name, record["error"])

    if all_rows:
        raw_probabilities = model.predict_proba(np.vstack(all_rows))[:, 1]
        calibrated_probabilities = _apply_calibrator(raw_probabilities, calibrator)
    else:
        raw_probabilities = np.asarray([], dtype=np.float64)
        calibrated_probabilities = np.asarray([], dtype=np.float64)

    for index, record in enumerate(file_records):
        if record["error"] is not None:
            results[index] = {
                "filename": record["path"].name,
                "error": f"Could not analyse this file: {record['error']}",
            }
            continue

        rows = [chunk["row"] for chunk in record["chunks"]]
        durations = [chunk["duration_seconds"] for chunk in record["chunks"]]
        rms_values = [chunk["rms"] for chunk in record["chunks"]]
        probabilities = calibrated_probabilities[rows]
        raw_scores = raw_probabilities[rows]

        score, spread, score_range, weights = _aggregate_chunk_scores(
            probabilities, durations, rms_values, model_bundle
        )
        prediction, reason, spoof_threshold, bonafide_threshold = _decision(
            score, spread, score_range, model_bundle
        )

        chunk_results = []
        for chunk, probability, weight in zip(record["chunks"], probabilities, weights):
            chunk_results.append({
                "index": chunk["index"],
                "start_seconds": chunk["start_seconds"],
                "duration_seconds": chunk["duration_seconds"],
                "spoof_score": round(float(probability) * 100, 2),
                "weight": round(float(weight), 3),
            })

        display_label = {
            "SPOOF": "AI-GENERATED / SPOOF",
            "BONAFIDE": "REAL / BONAFIDE",
            "UNCERTAIN": "UNCERTAIN - NEEDS REVIEW",
        }[prediction]

        result = {
            "filename": record["path"].name,
            "prediction": prediction,
            "display_label": display_label,
            "spoof_score": round(float(score) * 100, 2),
            "raw_spoof_score": round(float(np.average(raw_scores, weights=weights)) * 100, 2),
            "confidence": round(max(float(score), 1.0 - float(score)) * 100, 2),
            "confidence_label": _confidence_label(prediction, score),
            "calibrated": calibrator is not None,
            "chunks_analysed": len(record["chunks"]),
            "chunk_seconds": round(float(chunk_seconds), 2),
            "chunk_results": chunk_results,
            "chunk_disagreement": round(float(spread), 3),
            "duration_seconds": round(float(record["duration_seconds"]), 2),
            "truncated": bool(record["truncated"]),
            "threshold": spoof_threshold,
            "bonafide_threshold": bonafide_threshold,
            "reason": reason,
            "model_name": model_bundle.get("model_name", type(model).__name__),
            "feature_set": str(model_bundle.get("feature_set", DEFAULT_FEATURE_SET)),
        }
        results[index] = result

    return results


def predict_audio_file(audio_path, model_bundle):
    """Predict one file.  Raises a clear error if the file cannot be analysed."""
    result = predict_audio_files([audio_path], model_bundle)[0]
    if "error" in result:
        raise ValueError(result["error"])
    return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path):
    """Load the bundle saved by the training script/notebook."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Run `python train_model.py` or the training notebook to create it."
        )

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"{path} is not a valid model bundle. Retrain and save it again.")

    # Fill in defaults so old bundles still work.
    bundle = dict(bundle)
    if "feature_set" not in bundle:
        bundle["feature_set"] = infer_feature_set(int(bundle.get("n_features", 90)))
    if "spoof_threshold" not in bundle:
        bundle["spoof_threshold"] = float(bundle.get("threshold", 0.5))
    if "bonafide_threshold" not in bundle:
        bundle["bonafide_threshold"] = 0.5
    if "chunk_seconds" not in bundle:
        bundle["chunk_seconds"] = CHUNK_SECONDS
    if "calibrator" not in bundle:
        bundle["calibrator"] = None
    if "model_name" not in bundle:
        bundle["model_name"] = type(bundle["model"]).__name__
    return bundle


# ---------------------------------------------------------------------------
# Manual check: python audio_features.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import config

    names = feature_names()
    print("Audio feature extractor")
    print(f"  sample rate : {SAMPLE_RATE} Hz")
    print(f"  chunk       : {CHUNK_SECONDS} s")
    print(f"  feature set : {DEFAULT_FEATURE_SET}")
    print(f"  features    : {len(names)}")
    print(f"  first 3     : {names[:3]}")
    print(f"  last 3      : {names[-3:]}")

    sample = next(iter(sorted(config.SPLITS["dev"]["audio_dir"].glob("*.flac"))), None)
    if sample is None:
        raise SystemExit("No .flac files found - check DATASET_ROOT in config.py")

    vector = extract_features_from_file(sample)
    print(f"\n  test file   : {sample.name}")
    print(f"  vector shape: {vector.shape}")
    print(f"  value range : {vector.min():.2f} .. {vector.max():.2f}")
    print("OK")

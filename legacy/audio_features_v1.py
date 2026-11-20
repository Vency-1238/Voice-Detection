"""
Shared audio feature extraction for the Audio Deepfake Detector.

Both the training notebook and the Flask app import THIS module, so the feature
vector used to train the Random Forest is exactly the same one used at
prediction time. That is the single most important rule in the project.

Feature vector layout (90 values, always in this order):

    MFCC mean          (20 values)   coefficients 0..19
    MFCC std           (20 values)
    delta-MFCC mean    (20 values)
    delta-MFCC std     (20 values)
    spectral centroid  mean, std
    spectral bandwidth mean, std
    spectral rolloff   mean, std
    zero-crossing rate mean, std
    RMS                mean, std

Run `python audio_features.py` to sanity-check the module and see the layout.
"""

from pathlib import Path

import joblib
import librosa
import numpy as np

# ---------------------------------------------------------------------------
# Analysis settings - these MUST stay the same for training and prediction
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000     # ASVspoof 2019 LA recordings are 16 kHz
CHUNK_SECONDS = 5       # training reads the first 5 s; prediction uses 5 s chunks
N_MFCC = 20
DELTA_WIDTH = 9         # standard width for librosa.feature.delta
MIN_SECONDS = 0.25      # shorter audio cannot produce stable features

# (name, function) pairs keep the feature names and their order in sync.
_SPECTRAL_FEATURES = (
    ("spectral_centroid", lambda y, sr: librosa.feature.spectral_centroid(y=y, sr=sr)),
    ("spectral_bandwidth", lambda y, sr: librosa.feature.spectral_bandwidth(y=y, sr=sr)),
    ("spectral_rolloff", lambda y, sr: librosa.feature.spectral_rolloff(y=y, sr=sr)),
    ("zero_crossing_rate", lambda y, sr: librosa.feature.zero_crossing_rate(y=y)),
    ("rms", lambda y, sr: librosa.feature.rms(y=y)),
)


def feature_names():
    """Return the 90 feature names in the exact order they appear in the vector."""
    names = []
    for block in ("mfcc", "delta_mfcc"):
        for statistic in ("mean", "std"):
            for coefficient in range(N_MFCC):
                names.append(f"{block}_{statistic}_{coefficient}")
    for name, _ in _SPECTRAL_FEATURES:
        names.append(f"{name}_mean")
        names.append(f"{name}_std")
    return names


def _delta_mfcc(mfcc):
    """Delta-MFCC that also works on very short clips.

    librosa.feature.delta() needs more frames than its window width (9).
    A short chunk is handled with a narrower window instead of crashing.
    """
    frames = mfcc.shape[1]
    if frames >= DELTA_WIDTH:
        return librosa.feature.delta(mfcc, width=DELTA_WIDTH)

    width = frames if frames % 2 == 1 else frames - 1
    if width < 3:
        return np.zeros_like(mfcc)
    return librosa.feature.delta(mfcc, width=width)


def extract_features_from_waveform(y, sr=SAMPLE_RATE):
    """Turn a mono waveform into the fixed-length 90-value feature vector."""
    y = np.asarray(y, dtype=np.float32).ravel()

    if y.size == 0:
        raise ValueError("The audio file is empty.")

    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    if y.size < int(MIN_SECONDS * SAMPLE_RATE):
        raise ValueError(f"Audio is too short: at least {MIN_SECONDS} seconds are required.")

    if not np.all(np.isfinite(y)):
        raise ValueError("The audio contains invalid (NaN or infinite) samples.")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = _delta_mfcc(mfcc)

    features = []
    for matrix in (mfcc, delta):
        features.extend(np.mean(matrix, axis=1).tolist())
        features.extend(np.std(matrix, axis=1).tolist())

    for _, function in _SPECTRAL_FEATURES:
        values = function(y, sr)
        features.append(float(np.mean(values)))
        features.append(float(np.std(values)))

    vector = np.asarray(features, dtype=np.float32)

    expected = len(feature_names())
    if vector.size != expected:
        raise RuntimeError(f"Feature extraction returned {vector.size} values, expected {expected}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Could not extract valid audio features from this file.")

    return vector


def _load_audio(audio_path, max_seconds=None):
    """Load a file as mono 16 kHz audio, with a readable error for bad files."""
    try:
        return librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True, duration=max_seconds)
    except Exception as exc:
        raise ValueError(
            "This file could not be decoded. It may be corrupted, "
            "empty, or not a real audio file."
        ) from exc


def extract_features_from_file(audio_path, max_seconds=CHUNK_SECONDS):
    """Training-time extraction: read the first `max_seconds` of a file."""
    y, sr = _load_audio(audio_path, max_seconds=max_seconds)
    return extract_features_from_waveform(y, sr)


def predict_audio_file(audio_path, model_bundle):
    """Predict one file.

    Long recordings are split into chunks, each chunk gets a spoof probability,
    and the chunk probabilities are averaged. This keeps the input statistics
    close to what the model saw during training.
    """
    model = model_bundle["model"]
    threshold = float(model_bundle.get("threshold", 0.5))
    chunk_seconds = float(model_bundle.get("chunk_seconds", CHUNK_SECONDS))

    if hasattr(model, "n_features_in_") and model.n_features_in_ != len(feature_names()):
        raise RuntimeError(
            "This model was trained with a different feature set. "
            "Retrain it with the current audio_features.py."
        )

    y, _sr = _load_audio(audio_path)
    if y.size == 0:
        raise ValueError("The uploaded audio file is empty.")

    chunk_size = int(chunk_seconds * SAMPLE_RATE)
    minimum_chunk = int(0.5 * SAMPLE_RATE)

    rows = []
    for start in range(0, len(y), chunk_size):
        chunk = y[start:start + chunk_size]
        if len(chunk) < minimum_chunk:
            continue          # ignore a tiny leftover tail
        rows.append(extract_features_from_waveform(chunk, SAMPLE_RATE))

    if not rows:
        # Shorter than one chunk: analyse the whole recording in one piece.
        rows.append(extract_features_from_waveform(y, SAMPLE_RATE))

    probabilities = model.predict_proba(np.vstack(rows))[:, 1]
    spoof_score = float(np.mean(probabilities))
    prediction = "SPOOF" if spoof_score >= threshold else "BONAFIDE"

    return {
        "prediction": prediction,
        "display_label": "AI-GENERATED / SPOOF" if prediction == "SPOOF" else "REAL / BONAFIDE",
        "spoof_score": round(spoof_score * 100, 2),
        "confidence": round(max(spoof_score, 1 - spoof_score) * 100, 2),
        "chunks_analysed": len(rows),
        "duration_seconds": round(len(y) / SAMPLE_RATE, 2),
        "threshold": threshold,
    }


def load_model(model_path):
    """Load the model bundle saved by the training notebook."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Run the training notebook (Section 6) to create it."
        )

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"{path} is not a valid model bundle. Retrain and save it again.")
    return bundle


# ---------------------------------------------------------------------------
# Manual check:  python audio_features.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    import config

    names = feature_names()
    print(f"Audio feature extractor")
    print(f"  sample rate : {SAMPLE_RATE} Hz")
    print(f"  chunk       : {CHUNK_SECONDS} s")
    print(f"  features    : {len(names)}")
    print(f"  first 3     : {names[:3]}")
    print(f"  last 3      : {names[-3:]}")

    sample = next(iter(sorted(config.SPLITS["dev"]["audio_dir"].glob("*.flac"))), None)
    if sample is None:
        sys.exit("No .flac files found - check DATASET_ROOT in config.py")

    vector = extract_features_from_file(sample)
    print(f"\n  test file   : {sample.name}")
    print(f"  vector shape: {vector.shape}")
    print(f"  value range : {vector.min():.2f} .. {vector.max():.2f}")
    print("OK")

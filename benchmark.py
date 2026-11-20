"""
Diagnostics for the Audio Deepfake Detector.

This file is a MEASUREMENT tool, not part of the app. It answers three questions:

    python benchmark.py build     # build a fixed, reusable evaluation set
    python benchmark.py features  # where does feature extraction spend its time?
    python benchmark.py model     # how accurate and how fast is the saved model?

Every number is written to benchmarks/*.json so later optimisation stages can be
compared against this exact baseline. Nothing here changes the app.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil
from sklearn.metrics import confusion_matrix, roc_auc_score
from tqdm import tqdm

import config
import audio_features as af
from audio_features import (
    CHUNK_SECONDS,
    N_MFCC,
    SAMPLE_RATE,
    extract_features_from_file,
    extract_features_from_waveform,
    load_model,
    predict_audio_file,
)

BENCH_DIR = Path(__file__).resolve().parent / "benchmarks"
CACHE_PATH = BENCH_DIR / "dev_bench_features.npz"
EVAL_CACHE_PATH = BENCH_DIR / "eval_bench_features.npz"
SEED = 42
TUNE_FRACTION = 0.5      # half of the benchmark set tunes, half reports
FILES_PER_CLASS = 2000   # 2,000 bonafide + 2,000 spoof = 4,000 files


def cached_dev_path(feature_set):
    """Prefer the large full-feature cache built by train_model.py."""
    candidate = BENCH_DIR / "dev_all_full.npz"
    if feature_set == "full" and candidate.exists():
        return candidate
    return BENCH_DIR / f"dev_bench_{feature_set}.npz"


def cached_eval_path(feature_set):
    """Prefer the held-out unseen-attack cache built by train_model.py."""
    candidate = BENCH_DIR / "eval_holdout_2000_full.npz"
    if feature_set == "full" and candidate.exists():
        return candidate
    return BENCH_DIR / f"eval_bench_{feature_set}.npz"


def calibrated_scores(bundle, X):
    """Apply the bundle's calibrator so thresholds mean the same thing everywhere."""
    raw = bundle["model"].predict_proba(X)[:, 1]
    return raw, af._apply_calibrator(raw, bundle.get("calibrator"))


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------
def read_split(split):
    """Return [(audio_path, label, speaker_id), ...] for train/dev/eval."""
    spec = config.SPLITS[split]
    rows = []
    with open(spec["protocol"], encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            label_name = parts[-1].lower()
            if label_name not in {"bonafide", "spoof"}:
                continue
            audio_path = spec["audio_dir"] / f"{parts[1]}.flac"
            if audio_path.exists():
                rows.append((str(audio_path), 0 if label_name == "bonafide" else 1, parts[0]))
    return rows


def metrics(y_true, probabilities, threshold):
    """All classification metrics used in the final comparison table."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": threshold,
        "accuracy": round((tp + tn) / (tp + tn + fp + fn), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "fpr": round(fp / (fp + tn) if (fp + tn) else 0.0, 4),
        "fnr": round(fn / (fn + tp) if (fn + tp) else 0.0, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# build: cache a fixed evaluation set
# ---------------------------------------------------------------------------
def build_cache(feature_set="full"):
    """Cache features for a balanced, fixed dev subset (tune + test halves)."""
    BENCH_DIR.mkdir(exist_ok=True)
    existing = cached_dev_path(feature_set)
    if existing.exists():
        print(f"Cache already exists: {existing}")
        print("Delete it to rebuild.")
        return
    output = BENCH_DIR / f"dev_bench_{feature_set}.npz"

    rows = read_split("dev")
    bonafide = [row for row in rows if row[1] == 0]
    spoof = [row for row in rows if row[1] == 1]
    print(f"dev split: {len(bonafide):,} bonafide / {len(spoof):,} spoof")

    rng = np.random.default_rng(SEED)
    chosen = []
    for pool, count in ((bonafide, FILES_PER_CLASS), (spoof, FILES_PER_CLASS)):
        taken = rng.choice(len(pool), size=min(count, len(pool)), replace=False)
        chosen.extend(pool[index] for index in taken)

    # Alternate between class-balanced tune/test halves to keep both halves equal.
    chosen.sort(key=lambda row: (row[1], row[0]))
    halves = []
    for label in (0, 1):
        group = [row for row in chosen if row[1] == label]
        rng.shuffle(group)
        cut = int(len(group) * TUNE_FRACTION)
        for index, row in enumerate(group):
            halves.append((*row, "tune" if index < cut else "test"))

    print(f"extracting {feature_set} features for {len(halves):,} files ...")
    started = time.perf_counter()

    X, y, speakers, splits, paths = [], [], [], [], []
    for audio_path, label, speaker, part in tqdm(halves, desc="benchmark set"):
        try:
            X.append(extract_features_from_file(audio_path, feature_set=feature_set))
        except Exception:
            continue
        y.append(label)
        speakers.append(speaker)
        splits.append(part)
        paths.append(audio_path)

    elapsed = time.perf_counter() - started
    X = np.vstack(X)
    np.savez_compressed(
        output,
        X=X,
        y=np.asarray(y, dtype=np.int8),
        speaker=np.asarray(speakers),
        split=np.asarray(splits),
        path=np.asarray(paths),
        feature_names=np.asarray(af.feature_names(feature_set)),
    )

    per_file = elapsed / len(y) * 1000
    print(f"\nsaved {output}  X={X.shape}")
    print(f"extraction: {elapsed:.1f} s total, {per_file:.1f} ms/file, "
          f"{len(y) / elapsed:.1f} files/s")


# ---------------------------------------------------------------------------
# features: per-stage timing
# ---------------------------------------------------------------------------
def _time_call(function, repeats=5):
    """Mean milliseconds per call."""
    function()  # warm-up (numba compilation happens here)
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    return (time.perf_counter() - started) / repeats * 1000


def benchmark_features(n_files=150):
    import librosa
    import soundfile as sf

    rows = read_split("train")
    rng = np.random.default_rng(SEED)
    picks = [rows[i] for i in rng.choice(len(rows), size=n_files, replace=False)]

    stages = {
        "librosa.load (decode 16k)": [],
        "soundfile.read (decode)": [],
        "mfcc(n_mfcc=20)": [],
        "delta + delta-width": [],
        "spectral_centroid": [],
        "spectral_bandwidth": [],
        "spectral_rolloff": [],
        "zero_crossing_rate": [],
        "rms": [],
        "stft once + centroid/bw/rolloff": [],
        "extract_features_from_waveform": [],
    }

    # --- decode only, on real files -------------------------------------
    for audio_path, _label, _speaker in tqdm(picks[:60], desc="decode timing"):
        stages["librosa.load (decode 16k)"].append(
            _time_call(lambda p=audio_path: librosa.load(p, sr=SAMPLE_RATE, mono=True,
                                                         duration=CHUNK_SECONDS), repeats=3)
        )
        stages["soundfile.read (decode)"].append(
            _time_call(lambda p=audio_path: sf.read(p, dtype="float32"), repeats=3)
        )

    # --- compute only, on an already-decoded waveform --------------------
    y, sr = librosa.load(picks[0][0], sr=SAMPLE_RATE, mono=True, duration=CHUNK_SECONDS)
    frames_5s = len(y)

    def run_stage(name, function):
        for audio_path, _label, _speaker in tqdm(picks, desc=name, leave=False):
            waveform, rate = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True,
                                          duration=CHUNK_SECONDS)
            stages[name].append(_time_call(lambda: function(waveform, rate), repeats=3))

    run_stage("mfcc(n_mfcc=20)", lambda w, r: librosa.feature.mfcc(y=w, sr=r, n_mfcc=N_MFCC))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    stages["delta + delta-width"] = [_time_call(
        lambda: [librosa.feature.delta(mfcc, width=9), librosa.feature.delta(mfcc, width=9)],
        repeats=20)]

    run_stage("spectral_centroid", lambda w, r: librosa.feature.spectral_centroid(y=w, sr=r))
    run_stage("spectral_bandwidth", lambda w, r: librosa.feature.spectral_bandwidth(y=w, sr=r))
    run_stage("spectral_rolloff", lambda w, r: librosa.feature.spectral_rolloff(y=w, sr=r))
    run_stage("zero_crossing_rate", lambda w, r: librosa.feature.zero_crossing_rate(y=w))
    run_stage("rms", lambda w, r: librosa.feature.rms(y=w))

    def one_stft(w, r):
        magnitude = np.abs(librosa.stft(w))
        librosa.feature.spectral_centroid(S=magnitude, sr=r)
        librosa.feature.spectral_bandwidth(S=magnitude, sr=r)
        librosa.feature.spectral_rolloff(S=magnitude, sr=r)
        return magnitude

    run_stage("stft once + centroid/bw/rolloff", one_stft)
    run_stage("extract_features_from_waveform", extract_features_from_waveform)

    # --- resampling cost, measured at 44.1 kHz input --------------------
    y44 = librosa.resample(y, orig_sr=SAMPLE_RATE, target_sr=44100)
    resample_ms = _time_call(lambda: librosa.resample(y44, orig_sr=44100, target_sr=SAMPLE_RATE),
                             repeats=10)

    summary = {name: round(float(np.mean(values)), 3) for name, values in stages.items()}
    total = summary["extract_features_from_waveform"]
    decode = summary["librosa.load (decode 16k)"]

    print(f"\n{'stage':<40}{'ms/file':>10}")
    print("-" * 50)
    for name, value in summary.items():
        print(f"{name:<40}{value:>10.2f}")
    print("-" * 50)
    print(f"{'resample 44.1k -> 16k':<40}{resample_ms:>10.2f}")
    print(f"{'audio seconds per file (mean)':<40}"
          f"{np.mean([len(librosa.load(p, sr=SAMPLE_RATE, mono=True, duration=CHUNK_SECONDS)[0]) / SAMPLE_RATE for p, _, _ in picks[:30]]):>10.2f}")

    print(f"\nend-to-end per file : {decode + total:.1f} ms  "
          f"({1000 / (decode + total):.1f} files/s)")
    print(f"full train split    : {(decode + total) * 25380 / 1000 / 60:.1f} min")
    print(f"full dev split      : {(decode + total) * 24844 / 1000 / 60:.1f} min")

    payload = {
        "files": n_files,
        "seconds_per_file": round(frames_5s / SAMPLE_RATE, 2),
        "stage_ms": summary,
        "resample_44k_to_16k_ms": round(resample_ms, 3),
        "end_to_end_ms": round(decode + total, 2),
        "files_per_second": round(1000 / (decode + total), 2),
        "projected_train_split_minutes": round((decode + total) * 25380 / 1000 / 60, 1),
    }
    BENCH_DIR.mkdir(exist_ok=True)
    (BENCH_DIR / "baseline_features.json").write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {BENCH_DIR / 'baseline_features.json'}")


# ---------------------------------------------------------------------------
# model: accuracy + speed of the currently saved model
# ---------------------------------------------------------------------------
def benchmark_model():
    started = time.perf_counter()
    bundle = load_model(config.MODEL_PATH)
    load_ms = (time.perf_counter() - started) * 1000
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    feature_set = bundle.get("feature_set", "base90")

    cache_path = cached_dev_path(feature_set)
    if not cache_path.exists():
        raise SystemExit(
            f"Run `python benchmark.py build` first (expects {cache_path.name})."
        )

    with np.load(cache_path, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"].astype(int)
        speakers = data["speaker"]
        paths = data["path"] if "path" in data.files else data["paths"]
        if "split" in data.files:
            part = data["split"]
        else:
            # Large train_model cache: reproduce the same speaker-disjoint halves.
            import train_model as tm

            tune, _test = tm.split_dev_speakers(
                [{"speaker": s, "label": int(label)}
                 for s, label in zip(speakers, y)]
            )
            part = np.where(np.isin(speakers, list(tune)), "tune", "test")

    print(f"model      : {config.MODEL_PATH.name} "
          f"({config.MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"loaded in  : {load_ms:.0f} ms")
    print(f"trained on : {bundle.get('trained_on_files', '?')} files")
    print(f"feature set: {feature_set} ({bundle.get('n_features')})")
    print(f"cache      : {cache_path.name}  X={X.shape}")
    print(f"threshold  : {threshold}\n")

    _raw, probabilities = calibrated_scores(bundle, X)

    results = {}
    for name in ("tune", "test"):
        mask = part == name
        results[name] = metrics(y[mask], probabilities[mask], threshold)
        print(f"{name.upper()}  n={results[name]['n']:,}")
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "fpr", "fnr"):
            print(f"  {key:<10}{results[name][key]}")
        print(f"  tp={results[name]['tp']} tn={results[name]['tn']} "
              f"fp={results[name]['fp']} fn={results[name]['fn']}\n")

    # --- per-speaker reliability (speaker-independent check) -----------
    mask = np.ones(len(y), dtype=bool)
    per_speaker = {}
    for speaker in sorted(set(speakers[mask])):
        rows = mask & (speakers == speaker)
        if rows.sum() < 20:
            continue
        predictions = (probabilities[rows] >= threshold).astype(int)
        per_speaker[speaker] = {
            "n": int(rows.sum()),
            "fpr": round(float(np.mean(predictions[y[rows] == 0])), 4) if (y[rows] == 0).any() else None,
            "fnr": round(float(np.mean(predictions[y[rows] == 1] == 0)), 4) if (y[rows] == 1).any() else None,
        }

    fprs = [v["fpr"] for v in per_speaker.values() if v["fpr"] is not None]
    fnrs = [v["fnr"] for v in per_speaker.values() if v["fnr"] is not None]
    print(f"speakers in benchmark: {len(per_speaker)}")
    print(f"  bonafide FPR range : {min(fprs):.3f} .. {max(fprs):.3f} "
          f"(mean {np.mean(fprs):.3f})")
    print(f"  spoof FNR range    : {min(fnrs):.3f} .. {max(fnrs):.3f} "
          f"(mean {np.mean(fnrs):.3f})\n")

    # --- prediction speed ----------------------------------------------
    def batch_rate(batch_size, repeats=20):
        block = X[:batch_size]
        model.predict_proba(block)
        started_at = time.perf_counter()
        for _ in range(repeats):
            model.predict_proba(block)
        elapsed = (time.perf_counter() - started_at) / repeats
        return elapsed * 1000

    batch_ms = {str(size): round(batch_rate(size), 3) for size in (1, 8, 32, 128)}
    print("predict_proba latency:")
    for size, value in batch_ms.items():
        print(f"  {size:>3} rows : {value:>7.2f} ms  "
              f"({value / int(size) * 1000:>6.0f} us/row)")

    # End-to-end on real files (decode + features + predict).  The first call in
    # a fresh process pays for lazy imports, so time it separately.
    sample = list(paths[:40])
    cold_started = time.perf_counter()
    predict_audio_file(sample[0], bundle)
    cold_ms = (time.perf_counter() - cold_started) * 1000

    end_to_end = []
    for audio_path in sample[1:]:
        started_at = time.perf_counter()
        predict_audio_file(audio_path, bundle)
        end_to_end.append((time.perf_counter() - started_at) * 1000)

    print(f"\nend-to-end predict_audio_file: mean {np.mean(end_to_end):.0f} ms, "
          f"median {np.median(end_to_end):.0f} ms "
          f"({1000 / np.mean(end_to_end):.1f} files/s)")
    print(f"cold first call: {cold_ms:.0f} ms")

    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024

    payload = {
        "model": config.MODEL_PATH.name,
        "model_size_mb": round(config.MODEL_PATH.stat().st_size / 1024 / 1024, 2),
        "load_ms": round(load_ms, 1),
        "trained_on_files": bundle.get("trained_on_files"),
        "threshold": threshold,
        "metrics": {name: results[name] for name in results},
        "speakers": len(per_speaker),
        "speaker_fpr_mean": round(float(np.mean(fprs)), 4),
        "speaker_fpr_max": round(float(np.max(fprs)), 4),
        "speaker_fnr_mean": round(float(np.mean(fnrs)), 4),
        "speaker_fnr_max": round(float(np.max(fnrs)), 4),
        "predict_proba_ms": batch_ms,
        "end_to_end_ms_mean": round(float(np.mean(end_to_end)), 1),
        "end_to_end_ms_median": round(float(np.median(end_to_end)), 1),
        "cold_start_ms": round(cold_ms, 1),
        "rss_mb": round(memory_mb, 1),
    }
    (BENCH_DIR / "baseline_model.json").write_text(json.dumps(payload, indent=2))
    (BENCH_DIR / "baseline_predictions.npz").write_bytes(b"")
    np.savez_compressed(BENCH_DIR / "baseline_predictions.npz",
                        proba=probabilities, y=y, split=part)
    print(f"saved {BENCH_DIR / 'baseline_model.json'}")


# ---------------------------------------------------------------------------
# eval: unseen attacks (A07-A19) - the honest generalisation check
# ---------------------------------------------------------------------------
def build_eval_cache(per_class=1000, feature_set="full"):
    BENCH_DIR.mkdir(exist_ok=True)
    existing = cached_eval_path(feature_set)
    if existing.exists():
        print(f"Cache already exists: {existing}")
        return
    output = BENCH_DIR / f"eval_bench_{feature_set}.npz"

    rows = read_split("eval")
    bonafide = [row for row in rows if row[1] == 0]
    spoof = [row for row in rows if row[1] == 1]
    print(f"eval split: {len(bonafide):,} bonafide / {len(spoof):,} spoof")

    rng = np.random.default_rng(SEED)
    chosen = []
    for pool in (bonafide, spoof):
        picked = rng.choice(len(pool), size=min(per_class, len(pool)), replace=False)
        chosen.extend(pool[index] for index in picked)

    X, y, speakers, paths = [], [], [], []
    for audio_path, label, speaker in tqdm(chosen, desc="eval set"):
        try:
            X.append(extract_features_from_file(audio_path, feature_set=feature_set))
        except Exception:
            continue
        y.append(label)
        speakers.append(speaker)
        paths.append(audio_path)

    X = np.vstack(X)
    np.savez_compressed(output, X=X, y=np.asarray(y, dtype=np.int8),
                        speaker=np.asarray(speakers), path=np.asarray(paths),
                        feature_names=np.asarray(af.feature_names(feature_set)))
    print(f"saved {output}  X={X.shape}")


def benchmark_eval(tuned_threshold=0.873):
    bundle = load_model(config.MODEL_PATH)
    feature_set = bundle.get("feature_set", "base90")
    cache_path = cached_eval_path(feature_set)
    if not cache_path.exists():
        raise SystemExit(
            f"Run `python benchmark.py eval` first (expects {cache_path.name})."
        )

    with np.load(cache_path, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"].astype(int)

    threshold = float(bundle["spoof_threshold"])
    _raw, probabilities = calibrated_scores(bundle, X)

    print(f"eval split (unseen attacks A07-A19), n={len(y):,} "
          f"({int((y == 0).sum())} bonafide / {int((y == 1).sum())} spoof)")
    print(f"cache: {cache_path.name}  feature set: {feature_set}\n")

    output = {}
    for name, value in (("shipped threshold", threshold),
                        ("threshold tuned on dev", tuned_threshold)):
        result = metrics(y, probabilities, value)
        output[name] = result
        print(f"{name} ({value:.3f})")
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "fpr", "fnr"):
            print(f"  {key:<10}{result[key]}")
        print()

    payload = {
        "n": int(len(y)),
        "auc": round(float(roc_auc_score(y, probabilities)), 4),
        "at_shipped_threshold": output["shipped threshold"],
        "at_dev_tuned_threshold": output["threshold tuned on dev"],
    }
    (BENCH_DIR / "baseline_eval.json").write_text(json.dumps(payload, indent=2))
    print(f"saved {BENCH_DIR / 'baseline_eval.json'}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build", "features", "model", "eval", "all"])
    parser.add_argument("--files", type=int, default=150)
    args = parser.parse_args()

    if args.mode in ("build", "all"):
        build_cache()
    if args.mode in ("features", "all"):
        benchmark_features(n_files=args.files)
    if args.mode in ("model", "all"):
        benchmark_model()
    if args.mode == "eval":
        build_eval_cache()
        benchmark_eval()

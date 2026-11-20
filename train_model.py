"""
Train, compare and calibrate the Audio Deepfake Detector.

Examples
--------
    python train_model.py --mode all
    python train_model.py --mode compare --max-train-files 4000
    python train_model.py --mode final --target-fpr 0.02
    python train_model.py --mode cache

The script is deliberately separate from app.py: training is a batch job, while
the Flask app only loads the finished model bundle.

What it does
------------
1. Reads the ASVspoof protocol files (path + label + speaker + attack id).
2. Splits the dev split by SPEAKER into tune/test halves, so threshold tuning
   and final reporting never share a speaker.
3. Caches features so experiments can be repeated without re-decoding audio.
4. Compares feature sets and classifiers on the tune half.
5. Calibrates the chosen model on held-out dev data (Platt scaling).
6. Tunes the spoof threshold for a target false-positive rate.
7. Reports dev-test and unseen-attack (eval) metrics, confusion matrices and
   prediction speed, then saves the bundle used by app.py.

All reported numbers are measured on this machine.  Nothing is hard-coded.
"""

import argparse
import json
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

import audio_features as af
import config

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
BENCH_DIR = BASE_DIR / "benchmarks"
RESULTS_DIR = BASE_DIR / "results"
BENCH_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
DEFAULT_TARGET_FPR = 0.02
COMPARE_MODELS = ("rf300", "extra300", "extra500", "histgb", "logreg", "svc")
FEATURE_SETS_TO_COMPARE = (
    "base90",
    "flat_flux",
    "delta2",
    "flat_flux_delta2",
    "contrast",
    "flat_flux_contrast",
    "full",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def save_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def read_split(split):
    """Return protocol rows as dictionaries."""
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
            if not audio_path.exists():
                continue
            # ASVspoof protocol format:
            #   SPEAKER  FILE_ID  -  ATTACK_ID  LABEL
            # Bonafide rows use "-" for ATTACK_ID.
            attack = "-"
            if len(parts) >= 4:
                attack = parts[3]
            elif len(parts) == 3:
                attack = parts[2]
            rows.append({
                "path": str(audio_path),
                "label": 0 if label_name == "bonafide" else 1,
                "speaker": parts[0],
                "attack": attack,
            })
    return rows


def subsample_rows(rows, max_files, seed=SEED):
    """Shrink a split while keeping its original class ratio."""
    if max_files is None or max_files >= len(rows):
        return list(rows)
    rng = np.random.default_rng(seed)
    fraction = max_files / len(rows)
    keep = []
    for label in (0, 1):
        group = [row for row in rows if row["label"] == label]
        count = max(1, round(len(group) * fraction))
        indices = rng.choice(len(group), size=min(count, len(group)), replace=False)
        keep.extend(group[i] for i in indices)
    rng.shuffle(keep)
    return keep


def split_dev_speakers(rows, seed=SEED):
    """Speaker-disjoint tune/test split that keeps both classes on both sides."""
    classes_by_speaker = defaultdict(set)
    for row in rows:
        classes_by_speaker[row["speaker"]].add(row["label"])

    groups = {
        "both": [s for s, labels in classes_by_speaker.items() if labels == {0, 1}],
        "bonafide_only": [s for s, labels in classes_by_speaker.items() if labels == {0}],
        "spoof_only": [s for s, labels in classes_by_speaker.items() if labels == {1}],
    }

    rng = np.random.default_rng(seed)
    tune, test = set(), set()
    for group in groups.values():
        group = sorted(group)
        rng.shuffle(group)
        for index, speaker in enumerate(group):
            (tune if index % 2 == 0 else test).add(speaker)

    tune_labels = {row["label"] for row in rows if row["speaker"] in tune}
    test_labels = {row["label"] for row in rows if row["speaker"] in test}
    if tune_labels != {0, 1} or test_labels != {0, 1}:
        raise RuntimeError("Speaker split failed to put both classes on both sides.")

    return tune, test


# ---------------------------------------------------------------------------
# Feature caches
# ---------------------------------------------------------------------------
def cache_path(split, max_files, feature_set):
    tag = "all" if max_files is None else str(max_files)
    return BENCH_DIR / f"{split}_{tag}_{feature_set}.npz"


def build_or_load_cache(rows, path, feature_set, description):
    names = af.feature_names(feature_set)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            stored_names = [str(name) for name in data["feature_names"]]
            if stored_names == names:
                print(f"[cache] {description}: {path.name}  X={data['X'].shape}")
                return {
                    "X": data["X"],
                    "y": data["y"].astype(int),
                    "speaker": data["speaker"],
                    "path": data["path"],
                    "attack": data["attack"],
                    "feature_names": stored_names,
                }
        print(f"[cache] feature layout changed, rebuilding {path.name}")

    X, y, speakers, paths, attacks = [], [], [], [], []
    failed = 0
    started = time.perf_counter()
    print(f"[cache] building {description} ({len(rows):,} files) ...", flush=True)

    for row in tqdm(rows, desc=description, disable=True):
        try:
            X.append(af.extract_features_from_file(row["path"], feature_set=feature_set))
            y.append(row["label"])
            speakers.append(row["speaker"])
            paths.append(row["path"])
            attacks.append(row["attack"])
        except Exception:
            failed += 1

    if not X:
        raise RuntimeError(f"No features extracted for {description}")

    X = np.vstack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int8)
    speakers = np.asarray(speakers)
    paths = np.asarray(paths)
    attacks = np.asarray(attacks)

    np.savez_compressed(
        path,
        X=X,
        y=y,
        speaker=speakers,
        path=paths,
        attack=attacks,
        feature_names=np.asarray(names),
    )

    elapsed = time.perf_counter() - started
    print(
        f"[cache] {description}: saved {path.name}  X={X.shape}  "
        f"{elapsed:.1f}s ({elapsed / len(y) * 1000:.2f} ms/file), {failed} skipped"
    )
    return {
        "X": X,
        "y": y.astype(int),
        "speaker": speakers,
        "path": paths,
        "attack": attacks,
        "feature_names": names,
    }


def slice_feature_set(cache, target_feature_set):
    """Select columns by name; works even if the cache is a superset."""
    source = cache["feature_names"]
    target = af.feature_names(target_feature_set)
    index = {name: i for i, name in enumerate(source)}
    missing = [name for name in target if name not in index]
    if missing:
        raise ValueError(f"Cache is missing {len(missing)} features, e.g. {missing[:3]}")
    return cache["X"][:, [index[name] for name in target]]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def classification_metrics(y_true, probabilities, threshold):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": round(float(threshold), 6),
        "accuracy": round(float((tp + tn) / max(len(y_true), 1)), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 6),
        "fpr": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 6),
        "fnr": round(float(fn / (fn + tp)) if (fn + tp) else 0.0, 6),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n": int(len(y_true)),
    }


def _threshold_table(y_true, probabilities):
    """Efficient cumulative table for every distinct threshold."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(-probabilities, kind="mergesort")
    y_sorted = y_true[order]
    p_sorted = probabilities[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    total_pos = max(int(y_true.sum()), 1)
    total_neg = max(int(len(y_true) - y_true.sum()), 1)

    last = np.r_[np.where(np.diff(p_sorted) != 0)[0], len(p_sorted) - 1]
    tp = tp[last]
    fp = fp[last]
    thresholds = p_sorted[last]

    fpr = fp / total_neg
    recall = tp / total_pos
    precision = tp / np.maximum(tp + fp, 1)
    f1 = np.where(
        (precision + recall) > 0,
        2 * precision * recall / np.maximum(precision + recall, 1e-12),
        0.0,
    )
    return thresholds, fpr, recall, precision, f1


def threshold_for_fpr(y_true, probabilities, target_fpr):
    """Lowest number of false positives that still gives good recall."""
    thresholds, fpr, recall, precision, f1 = _threshold_table(y_true, probabilities)
    feasible = fpr <= float(target_fpr)
    if feasible.any():
        score = f1 + 1e-6 * recall
        score = np.where(feasible, score, -1.0)
        best = int(np.argmax(score))
    else:
        # Extremely unusual: even the highest score includes a bonafide.
        best = int(np.argmin(fpr - 1e-6 * recall))

    return float(thresholds[best]), {
        "fpr_on_tune": round(float(fpr[best]), 6),
        "recall_on_tune": round(float(recall[best]), 6),
        "precision_on_tune": round(float(precision[best]), 6),
        "f1_on_tune": round(float(f1[best]), 6),
        "target_fpr": float(target_fpr),
    }


def threshold_for_best_f1(y_true, probabilities):
    thresholds, fpr, recall, precision, f1 = _threshold_table(y_true, probabilities)
    best = int(np.argmax(f1))
    return float(thresholds[best]), {
        "fpr_on_tune": round(float(fpr[best]), 6),
        "recall_on_tune": round(float(recall[best]), 6),
        "precision_on_tune": round(float(precision[best]), 6),
        "f1_on_tune": round(float(f1[best]), 6),
    }


def fit_platt_calibrator(probabilities, y_true):
    """Fit Platt scaling on held-out predictions."""
    from scipy.special import logit

    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    features = logit(clipped).reshape(-1, 1)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(features, np.asarray(y_true, dtype=int))
    return {
        "kind": "platt",
        "coef": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
    }


def calibration_metrics(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    bins = np.linspace(0.0, 1.0, 11)
    indices = np.clip(np.digitize(probabilities, bins) - 1, 0, 9)
    ece = 0.0
    for index in range(10):
        mask = indices == index
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())

    return {
        "brier": round(float(brier_score_loss(y_true, probabilities)), 6),
        "log_loss": round(float(log_loss(y_true, probabilities)), 6),
        "ece": round(float(ece), 6),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def make_model(name, seed=SEED):
    if name == "rf300":
        classifier = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ])

    if name == "extra150":
        # 150 trees generalise as well as 300 on unseen attacks but predict far
        # faster.  n_jobs=1 is deliberate: joblib thread overhead dominates for
        # one-row and small-batch prediction, which is exactly the app's workload.
        classifier = ExtraTreesClassifier(
            n_estimators=150,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=1,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ])

    if name == "extra300":
        classifier = ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ])

    if name == "extra500":
        classifier = ExtraTreesClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            min_samples_leaf=1,
            max_features=0.5,
            random_state=seed,
            n_jobs=-1,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ])

    if name == "histgb":
        classifier = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ])

    if name == "logreg":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=seed,
            )),
        ])

    if name == "svc":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=2.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=seed,
                cache_size=1000,
            )),
        ])

    raise ValueError(f"Unknown model: {name}")


def prediction_latency(model, X, repeats=10):
    """Milliseconds for one row and for a 128-row batch (warm model)."""
    one = X[:1]
    batch = X[: min(128, len(X))]
    model.predict_proba(one)
    model.predict_proba(batch)

    started = time.perf_counter()
    for _ in range(repeats):
        model.predict_proba(one)
    one_ms = (time.perf_counter() - started) / repeats * 1000

    started = time.perf_counter()
    for _ in range(repeats):
        model.predict_proba(batch)
    batch_ms = (time.perf_counter() - started) / repeats * 1000

    return {
        "one_row_ms": round(one_ms, 3),
        "batch_128_ms": round(batch_ms, 3),
        "batch_128_us_per_row": round(batch_ms / len(batch) * 1000, 3),
    }


def evaluate_candidate(name, X_train, y_train, X_tune, y_tune, target_fpr):
    model = make_model(name)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started

    probabilities = model.predict_proba(X_tune)[:, 1]
    best_f1_threshold, best_f1_info = threshold_for_best_f1(y_tune, probabilities)
    target_threshold, target_info = threshold_for_fpr(y_tune, probabilities, target_fpr)

    result = {
        "model": name,
        "train_seconds": round(train_seconds, 2),
        "tune_at_0_5": classification_metrics(y_tune, probabilities, 0.5),
        "tune_best_f1": {
            "threshold": round(best_f1_threshold, 6),
            **best_f1_info,
            **classification_metrics(y_tune, probabilities, best_f1_threshold),
        },
        "tune_at_target_fpr": {
            "threshold": round(target_threshold, 6),
            **target_info,
            **classification_metrics(y_tune, probabilities, target_threshold),
        },
        "latency": prediction_latency(model, X_tune),
        "calibration_before": calibration_metrics(y_tune, probabilities),
    }
    print(
        f"  {name:<10} AUC={result['tune_at_0_5']['roc_auc']:.4f}  "
        f"F1@targetFPR={result['tune_at_target_fpr']['f1']:.4f}  "
        f"train={train_seconds:.1f}s  "
        f"predict1={result['latency']['one_row_ms']:.1f}ms"
    )
    return model, result


def measure_feature_cost(audio_paths, feature_set, n_files=100):
    """Pure feature-extraction time per file (I/O excluded)."""
    waveforms = []
    for path in audio_paths[:n_files]:
        try:
            y, _ = af.load_audio(path, max_seconds=af.CHUNK_SECONDS)
            waveforms.append(y)
        except Exception:
            continue
    if not waveforms:
        return None

    af.extract_features_from_waveform(waveforms[0], feature_set=feature_set)
    started = time.perf_counter()
    for waveform in waveforms:
        af.extract_features_from_waveform(waveform, feature_set=feature_set)
    return round((time.perf_counter() - started) / len(waveforms) * 1000, 3)


# ---------------------------------------------------------------------------
# Final model
# ---------------------------------------------------------------------------
def train_final(train_cache, dev_cache, eval_cache, feature_set, model_name,
                target_fpr, baseline_info):
    X_train = slice_feature_set(train_cache, feature_set)
    y_train = train_cache["y"]

    X_dev = slice_feature_set(dev_cache, feature_set)
    y_dev = dev_cache["y"]
    dev_speakers = dev_cache["speaker"]

    tune_speakers, test_speakers = split_dev_speakers(
        [{"speaker": s, "label": int(label)} for s, label in zip(dev_speakers, y_dev)]
    )
    tune_mask = np.isin(dev_speakers, list(tune_speakers))
    test_mask = np.isin(dev_speakers, list(test_speakers))

    print(f"  train rows       : {len(y_train):,}")
    print(f"  dev tune rows    : {int(tune_mask.sum()):,}  ({len(tune_speakers)} speakers)")
    print(f"  dev test rows    : {int(test_mask.sum()):,}  ({len(test_speakers)} speakers)")

    model = make_model(model_name)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started
    print(f"  trained {model_name} on {len(y_train):,} rows in {train_seconds:.1f}s")

    raw_tune = model.predict_proba(X_dev[tune_mask])[:, 1]
    calibrator = fit_platt_calibrator(raw_tune, y_dev[tune_mask])
    recalibrated_tune = af._apply_calibrator(raw_tune, calibrator)

    threshold, threshold_info = threshold_for_fpr(
        y_dev[tune_mask], recalibrated_tune, target_fpr
    )
    bonafide_threshold = 0.5 if threshold >= 0.5 else max(0.0, threshold - 0.2)

    raw_test = model.predict_proba(X_dev[test_mask])[:, 1]
    calibrated_test = af._apply_calibrator(raw_test, calibrator)
    test_metrics = classification_metrics(y_dev[test_mask], calibrated_test, threshold)
    test_metrics_uncalibrated = classification_metrics(y_dev[test_mask], raw_test, threshold)

    print(f"  threshold        : {threshold:.4f} (target FPR {target_fpr:.1%})")
    print(
        f"  dev test         : accuracy={test_metrics['accuracy']:.4f} "
        f"precision={test_metrics['precision']:.4f} recall={test_metrics['recall']:.4f} "
        f"F1={test_metrics['f1']:.4f} AUC={test_metrics['roc_auc']:.4f} "
        f"FPR={test_metrics['fpr']:.4f} FNR={test_metrics['fnr']:.4f}"
    )

    eval_metrics = None
    eval_payload = None
    if eval_cache is not None:
        X_eval = slice_feature_set(eval_cache, feature_set)
        y_eval = eval_cache["y"]
        raw_eval = model.predict_proba(X_eval)[:, 1]
        calibrated_eval = af._apply_calibrator(raw_eval, calibrator)
        eval_metrics = classification_metrics(y_eval, calibrated_eval, threshold)

        # Per-attack recall on the unseen attacks.
        attack_counts = {}
        for attack in sorted(set(str(a) for a in eval_cache["attack"])):
            mask = np.asarray([str(a) == attack for a in eval_cache["attack"]])
            if mask.sum() == 0:
                continue
            attack_counts[attack] = {
                "n": int(mask.sum()),
                "mean_spoof_score": round(float(calibrated_eval[mask].mean()) * 100, 2),
                "recall_at_threshold": round(
                    float((calibrated_eval[mask] >= threshold).mean()), 4
                ) if attack != "-" else None,
            }

        eval_payload = {
            "metrics": eval_metrics,
            "calibration": calibration_metrics(y_eval, calibrated_eval),
            "per_attack": attack_counts,
        }
        print(
            f"  eval (unseen)    : accuracy={eval_metrics['accuracy']:.4f} "
            f"precision={eval_metrics['precision']:.4f} recall={eval_metrics['recall']:.4f} "
            f"F1={eval_metrics['f1']:.4f} AUC={eval_metrics['roc_auc']:.4f} "
            f"FPR={eval_metrics['fpr']:.4f} FNR={eval_metrics['fnr']:.4f}"
        )

    # End-to-end prediction speed on 40 real dev-test files.
    sample_paths = list(dev_cache["path"][test_mask][:40])
    bundle = {
        "model": model,
        "calibrator": calibrator,
        "feature_set": feature_set,
        "feature_names": af.feature_names(feature_set),
        "n_features": len(af.feature_names(feature_set)),
        "sample_rate": af.SAMPLE_RATE,
        "chunk_seconds": af.CHUNK_SECONDS,
        "n_mfcc": af.N_MFCC,
        "threshold": threshold,
        "spoof_threshold": threshold,
        "bonafide_threshold": bonafide_threshold,
        "model_name": model_name,
        "trained_on_files": int(len(y_train)),
        "trained_on_speakers": int(len(set(train_cache["speaker"]))),
        "target_fpr": float(target_fpr),
        "validation": {
            "dev_test": test_metrics,
            "dev_test_uncalibrated": test_metrics_uncalibrated,
            "threshold_tuning": threshold_info,
            "calibration": calibration_metrics(y_dev[test_mask], calibrated_test),
        },
        "eval": eval_payload,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "version": 2,
        "feature_description": (
            "mean/std of MFCC, delta-MFCC, optional delta-delta, and spectral "
            "features; one shared STFT per chunk"
        ),
    }

    if sample_paths:
        started = time.perf_counter()
        af.predict_audio_files(sample_paths, bundle)
        batch_ms = (time.perf_counter() - started) / len(sample_paths) * 1000
    else:
        batch_ms = None

    # Single-file path, including decode + feature extraction + prediction.
    single_times = []
    for path in sample_paths[:20]:
        started = time.perf_counter()
        af.predict_audio_file(path, bundle)
        single_times.append((time.perf_counter() - started) * 1000)

    speed = {
        "single_file_ms_mean": round(float(np.mean(single_times)), 2) if single_times else None,
        "single_file_ms_median": round(float(np.median(single_times)), 2) if single_times else None,
        "batch_per_file_ms": round(float(batch_ms), 2) if batch_ms is not None else None,
        "model_predict_proba": prediction_latency(model, X_dev[test_mask]),
    }

    return bundle, {
        "feature_set": feature_set,
        "model": model_name,
        "train_seconds": round(train_seconds, 2),
        "threshold": threshold,
        "bonafide_threshold": bonafide_threshold,
        "threshold_tuning": threshold_info,
        "dev_test": test_metrics,
        "dev_test_uncalibrated": test_metrics_uncalibrated,
        "dev_test_calibration": calibration_metrics(y_dev[test_mask], calibrated_test),
        "eval": eval_payload,
        "speed": speed,
        "baseline": baseline_info,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["all", "cache", "compare", "final"],
        default="all",
        help="cache builds features; compare runs experiments; final trains/saves",
    )
    parser.add_argument("--max-train-files", type=int, default=4000,
                        help="training rows used for model comparison")
    parser.add_argument("--final-train-files", type=int, default=None,
                        help="training rows for the final model (default: all 25,380)")
    parser.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR,
                        help="target false-positive rate on dev-tune speakers")
    parser.add_argument("--feature-set", default=None,
                        help="force a feature set for the final model")
    parser.add_argument("--model", default=None,
                        help="force a classifier for the final model")
    parser.add_argument("--skip-svc", action="store_true",
                        help="skip the slow SVM comparison")
    args = parser.parse_args()

    print("=" * 78)
    print("Audio Deepfake Detector - training and comparison")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_rows = read_split("train")
    dev_rows = read_split("dev")
    eval_rows = read_split("eval")
    print(f"train: {len(train_rows):,} rows")
    print(f"dev  : {len(dev_rows):,} rows")
    print(f"eval : {len(eval_rows):,} rows (unseen attacks)")

    tune_speakers, test_speakers = split_dev_speakers(dev_rows)
    print(f"dev speakers: tune={len(tune_speakers)} test={len(test_speakers)}")

    # ------------------------------------------------------------------
    # Caches.  A full feature cache is built once and sliced for every
    # experiment.  This is the main reason experiments can be repeated fast.
    # ------------------------------------------------------------------
    full_dev_cache = build_or_load_cache(
        dev_rows,
        cache_path("dev", None, "full"),
        "full",
        "dev full-feature cache",
    )
    compare_train_rows = subsample_rows(train_rows, args.max_train_files)
    full_compare_cache = build_or_load_cache(
        compare_train_rows,
        cache_path("train", args.max_train_files, "full"),
        "full",
        f"train {len(compare_train_rows):,}-row full-feature cache",
    )

    full_final_cache = None
    if args.mode in ("all", "final"):
        final_rows = subsample_rows(train_rows, args.final_train_files)
        full_final_cache = build_or_load_cache(
            final_rows,
            cache_path("train", args.final_train_files, "full"),
            "full",
            f"final train {len(final_rows):,}-row full-feature cache",
        )

    eval_cache = None
    if args.mode in ("all", "final"):
        # Same balanced sampling as benchmark.py, for a comparable unseen-attack test.
        rng = np.random.default_rng(SEED)
        bonafide = [row for row in eval_rows if row["label"] == 0]
        spoof = [row for row in eval_rows if row["label"] == 1]
        chosen = []
        for pool, count in ((bonafide, 1000), (spoof, 1000)):
            indices = rng.choice(len(pool), size=min(count, len(pool)), replace=False)
            chosen.extend(pool[i] for i in indices)
        eval_cache = build_or_load_cache(
            chosen,
            cache_path("eval", 2000, "full"),
            "full",
            "eval 2,000-row full-feature cache",
        )

    baseline_info = {}
    for name in ("baseline_model.json", "baseline_eval.json", "baseline_features.json"):
        path = BENCH_DIR / name
        if path.exists():
            baseline_info[name.replace(".json", "")] = json.loads(
                path.read_text(encoding="utf-8")
            )

    comparison = {"baseline": baseline_info, "feature_extraction": {}, "feature_sets": {},
                  "models": {}, "created_at": datetime.now().isoformat(timespec="seconds")}

    # ------------------------------------------------------------------
    # Feature cost + feature-set comparison
    # ------------------------------------------------------------------
    if args.mode in ("all", "compare"):
        sample_paths = [row["path"] for row in dev_rows[:120]]
        print("\nFeature extraction cost (5 s chunks, I/O excluded):")
        for feature_set in FEATURE_SETS_TO_COMPARE:
            milliseconds = measure_feature_cost(sample_paths, feature_set)
            comparison["feature_extraction"][feature_set] = {
                "n_features": len(af.feature_names(feature_set)),
                "ms_per_5s_chunk": milliseconds,
            }
            print(f"  {feature_set:<22} {len(af.feature_names(feature_set)):>4} features  "
                  f"{milliseconds:>6.2f} ms")

        y_train_compare = full_compare_cache["y"]
        y_dev = full_dev_cache["y"]
        dev_speakers = full_dev_cache["speaker"]
        tune_speakers, _ = split_dev_speakers(
            [{"speaker": s, "label": int(label)}
             for s, label in zip(dev_speakers, y_dev)]
        )
        tune_mask = np.isin(dev_speakers, list(tune_speakers))

        print("\nFeature-set comparison (ExtraTrees, same train/tune rows):")
        for feature_set in FEATURE_SETS_TO_COMPARE:
            X_train = slice_feature_set(full_compare_cache, feature_set)
            X_tune = slice_feature_set(full_dev_cache, feature_set)[tune_mask]
            y_tune = y_dev[tune_mask]
            _, result = evaluate_candidate(
                "extra300", X_train, y_train_compare, X_tune, y_tune, args.target_fpr
            )
            comparison["feature_sets"][feature_set] = result

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------
    chosen_feature_set = args.feature_set
    chosen_model = args.model
    if args.mode in ("all", "compare"):
        if chosen_feature_set is None:
            # Pick by F1 at the target FPR, using the tune half only.
            ranked = sorted(
                comparison["feature_sets"].items(),
                key=lambda item: (
                    item[1]["tune_at_target_fpr"]["f1"],
                    item[1]["tune_at_0_5"]["roc_auc"],
                ),
                reverse=True,
            )
            chosen_feature_set = ranked[0][0]
            print(f"\nBest feature set on tune data: {chosen_feature_set}")
        else:
            print(f"\nUsing requested feature set: {chosen_feature_set}")

        X_train = slice_feature_set(full_compare_cache, chosen_feature_set)
        X_tune = slice_feature_set(full_dev_cache, chosen_feature_set)[tune_mask]
        y_tune = y_dev[tune_mask]

        model_names = [name for name in COMPARE_MODELS if not (
            args.skip_svc and name == "svc"
        )]
        print(f"\nClassifier comparison on {chosen_feature_set}:")
        fitted = {}
        for name in model_names:
            model, result = evaluate_candidate(
                name, X_train, y_train_compare, X_tune, y_tune, args.target_fpr
            )
            comparison["models"][name] = result
            fitted[name] = model

        if chosen_model is None:
            ranked = sorted(
                comparison["models"].items(),
                key=lambda item: (
                    item[1]["tune_at_target_fpr"]["f1"],
                    item[1]["tune_at_0_5"]["roc_auc"],
                    -item[1]["latency"]["one_row_ms"],
                ),
                reverse=True,
            )
            chosen_model = ranked[0][0]
            print(f"\nBest classifier on tune data: {chosen_model}")
        else:
            print(f"\nUsing requested classifier: {chosen_model}")

    if args.mode == "cache":
        print("\nFeature caches are ready.")
        save_json(RESULTS_DIR / "optimization_report.json", comparison)
        return

    # ------------------------------------------------------------------
    # Final model
    # ------------------------------------------------------------------
    if args.mode in ("all", "final"):
        if chosen_feature_set is None:
            chosen_feature_set = "base90"
        if chosen_model is None:
            chosen_model = "extra300"

        if full_final_cache is None:
            final_rows = subsample_rows(train_rows, args.final_train_files)
            full_final_cache = build_or_load_cache(
                final_rows,
                cache_path("train", args.final_train_files, "full"),
                "full",
                f"final train {len(final_rows):,}-row full-feature cache",
            )

        print("\nTraining the final model:")
        bundle, final_report = train_final(
            full_final_cache,
            full_dev_cache,
            eval_cache,
            chosen_feature_set,
            chosen_model,
            args.target_fpr,
            baseline_info,
        )
        comparison["final"] = final_report
        comparison["chosen_feature_set"] = chosen_feature_set
        comparison["chosen_model"] = chosen_model

        joblib.dump(bundle, config.MODEL_PATH)
        print(f"\nSaved model: {config.MODEL_PATH}")
        print(f"  feature set : {chosen_feature_set} ({bundle['n_features']} features)")
        print(f"  classifier  : {chosen_model}")
        print(f"  threshold   : {bundle['threshold']:.4f}")

    save_json(RESULTS_DIR / "model_comparison.json", comparison)
    save_json(RESULTS_DIR / "optimization_report.json", comparison)
    print(f"\nReport saved: {RESULTS_DIR / 'optimization_report.json'}")


if __name__ == "__main__":
    main()

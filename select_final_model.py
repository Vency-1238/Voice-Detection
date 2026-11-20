"""
Choose the production model using unseen-attack validation data.

Data roles
----------
train full     : 25,380 ASVspoof train files (attacks A01-A06)
dev tune       : 10 speakers, used for calibration and threshold tuning
dev test       : 10 different speakers, speaker-independent report
eval validation: 2,000 files, attacks A07-A19 (used for model selection)
eval holdout   : 2,000 different files, attacks A07-A19 (final report only)

This script does not touch the holdout until the final table.

Run:
    python select_final_model.py
"""

import json
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np

import audio_features as af
import train_model as tm

BASE_DIR = Path(__file__).resolve().parent
BENCH_DIR = BASE_DIR / "benchmarks"
RESULTS_DIR = BASE_DIR / "results"

FULL_TRAIN = BENCH_DIR / "train_all_full.npz"
DEV = BENCH_DIR / "dev_all_full.npz"
EVAL_VAL = BENCH_DIR / "eval_2000_full.npz"
EVAL_HOLDOUT = BENCH_DIR / "eval_holdout_2000_full.npz"

FEATURE_SET = "full"
TARGET_FPR = 0.02


def load_cache(path):
    with np.load(path, allow_pickle=False) as data:
        return {
            "X": data["X"],
            "y": data["y"].astype(int),
            "speaker": data["speaker"],
            "path": data["path"],
            "feature_names": [str(name) for name in data["feature_names"]],
        }


def three_way(y_true, probabilities, spoof_threshold, bonafide_threshold):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    decisions = np.where(
        probabilities >= spoof_threshold,
        "SPOOF",
        np.where(probabilities <= bonafide_threshold, "BONAFIDE", "UNCERTAIN"),
    )
    output = {}
    for label, name in ((0, "bonafide"), (1, "spoof")):
        mask = y_true == label
        total = max(int(mask.sum()), 1)
        values, counts = np.unique(decisions[mask], return_counts=True)
        counts_by_decision = {str(v): int(c) for v, c in zip(values, counts)}
        output[name] = {
            "n": int(mask.sum()),
            "bonafide_pct": round(counts_by_decision.get("BONAFIDE", 0) / total * 100, 2),
            "uncertain_pct": round(counts_by_decision.get("UNCERTAIN", 0) / total * 100, 2),
            "spoof_pct": round(counts_by_decision.get("SPOOF", 0) / total * 100, 2),
        }
    return output


def evaluate(model, calibrator, X, y, spoof_threshold, bonafide_threshold):
    raw = model.predict_proba(X)[:, 1]
    calibrated = af._apply_calibrator(raw, calibrator)
    return {
        "metrics": tm.classification_metrics(y, calibrated, spoof_threshold),
        "three_way": three_way(y, calibrated, spoof_threshold, bonafide_threshold),
        "calibration": tm.calibration_metrics(y, calibrated),
        "probabilities": calibrated,
    }


def latency(model, X):
    one = X[:1]
    batch = X[: min(128, len(X))]
    model.predict_proba(one)
    model.predict_proba(batch)
    started = time.perf_counter()
    for _ in range(20):
        model.predict_proba(one)
    one_ms = (time.perf_counter() - started) / 20 * 1000
    started = time.perf_counter()
    for _ in range(20):
        model.predict_proba(batch)
    batch_ms = (time.perf_counter() - started) / 20 * 1000
    return {
        "one_row_ms": round(one_ms, 3),
        "batch_128_ms": round(batch_ms, 3),
        "batch_us_per_row": round(batch_ms / len(batch) * 1000, 3),
    }


def model_size_mb(model):
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as handle:
        path = Path(handle.name)
    try:
        joblib.dump(model, path)
        return round(path.stat().st_size / 1024 / 1024, 2)
    finally:
        path.unlink(missing_ok=True)


def candidate(name):
    if name == "rf300":
        return tm.make_model("rf300")
    if name == "extra300":
        return tm.make_model("extra300")
    if name == "extra150":
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", ExtraTreesClassifier(
                n_estimators=150,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=tm.SEED,
                n_jobs=-1,
            )),
        ])
    if name == "svc":
        return tm.make_model("svc")
    if name == "histgb":
        return tm.make_model("histgb")
    raise ValueError(name)


def main():
    train = load_cache(FULL_TRAIN)
    dev = load_cache(DEV)
    eval_val = load_cache(EVAL_VAL)
    eval_holdout = load_cache(EVAL_HOLDOUT)

    X_train = tm.slice_feature_set(train, FEATURE_SET)
    y_train = train["y"]
    X_dev = tm.slice_feature_set(dev, FEATURE_SET)
    y_dev = dev["y"]
    X_val = tm.slice_feature_set(eval_val, FEATURE_SET)
    y_val = eval_val["y"]
    X_hold = tm.slice_feature_set(eval_holdout, FEATURE_SET)
    y_hold = eval_holdout["y"]

    tune_speakers, test_speakers = tm.split_dev_speakers(
        [{"speaker": s, "label": int(label)}
         for s, label in zip(dev["speaker"], y_dev)]
    )
    tune_mask = np.isin(dev["speaker"], list(tune_speakers))
    test_mask = np.isin(dev["speaker"], list(test_speakers))

    print(f"train {X_train.shape}, dev tune {int(tune_mask.sum())}, "
          f"dev test {int(test_mask.sum())}, eval val {len(y_val)}, "
          f"holdout {len(y_hold)}")

    results = {}
    for name in ("rf300", "extra150", "extra300", "histgb", "svc"):
        print(f"\n=== {name} ===")
        model = candidate(name)
        started = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - started

        raw_tune = model.predict_proba(X_dev[tune_mask])[:, 1]
        calibrator = tm.fit_platt_calibrator(raw_tune, y_dev[tune_mask])
        p_tune = af._apply_calibrator(raw_tune, calibrator)
        dev_threshold, dev_info = tm.threshold_for_fpr(
            y_dev[tune_mask], p_tune, TARGET_FPR
        )
        bonafide_threshold = 0.5 if dev_threshold >= 0.5 else max(0.0, dev_threshold - 0.2)

        dev_test = evaluate(
            model, calibrator, X_dev[test_mask], y_dev[test_mask],
            dev_threshold, bonafide_threshold,
        )
        val = evaluate(
            model, calibrator, X_val, y_val, dev_threshold, bonafide_threshold
        )
        hold_at_dev = evaluate(
            model, calibrator, X_hold, y_hold, dev_threshold, bonafide_threshold
        )

        # Threshold re-tuned on eval validation, reported on the holdout.
        val_threshold, val_info = tm.threshold_for_fpr(y_val, val["probabilities"], TARGET_FPR)
        val_bonafide = 0.5 if val_threshold >= 0.5 else max(0.0, val_threshold - 0.2)
        hold_at_val = evaluate(
            model, calibrator, X_hold, y_hold, val_threshold, val_bonafide
        )
        val_at_val = evaluate(
            model, calibrator, X_val, y_val, val_threshold, val_bonafide
        )

        # Tree latency with n_jobs=1 is often much lower for one-row batches.
        latency_info = latency(model, X_val)
        if hasattr(model, "named_steps") and hasattr(model.named_steps["classifier"], "n_jobs"):
            model.named_steps["classifier"].set_params(n_jobs=1)
            latency_single_thread = latency(model, X_val)
        else:
            latency_single_thread = latency_info

        record = {
            "train_seconds": round(train_seconds, 2),
            "model_size_mb": model_size_mb(model),
            "dev_threshold": round(float(dev_threshold), 6),
            "dev_threshold_info": dev_info,
            "bonafide_threshold": bonafide_threshold,
            "dev_test": {k: v for k, v in dev_test.items() if k != "probabilities"},
            "eval_val_at_dev_threshold": {k: v for k, v in val.items() if k != "probabilities"},
            "eval_holdout_at_dev_threshold": {k: v for k, v in hold_at_dev.items() if k != "probabilities"},
            "eval_val_threshold": round(float(val_threshold), 6),
            "eval_val_threshold_info": val_info,
            "eval_val_at_val_threshold": {k: v for k, v in val_at_val.items() if k != "probabilities"},
            "eval_holdout_at_val_threshold": {k: v for k, v in hold_at_val.items() if k != "probabilities"},
            "latency": {
                "default": latency_info,
                "single_thread": latency_single_thread,
            },
        }
        results[name] = record
        print(f"  train {train_seconds:.1f}s  size {record['model_size_mb']:.2f} MB")
        print(f"  dev threshold {dev_threshold:.4f} -> dev test F1 "
              f"{dev_test['metrics']['f1']:.4f} FPR {dev_test['metrics']['fpr']:.4f}")
        print(f"  eval val  F1 {val['metrics']['f1']:.4f} "
              f"FPR {val['metrics']['fpr']:.4f} FNR {val['metrics']['fnr']:.4f} "
              f"AUC {val['metrics']['roc_auc']:.4f}")
        print(f"  holdout   F1 {hold_at_dev['metrics']['f1']:.4f} "
              f"FPR {hold_at_dev['metrics']['fpr']:.4f} "
              f"FNR {hold_at_dev['metrics']['fnr']:.4f} "
              f"AUC {hold_at_dev['metrics']['roc_auc']:.4f}")
        print(f"  val-tuned threshold {val_threshold:.4f} -> holdout F1 "
              f"{hold_at_val['metrics']['f1']:.4f} FPR {hold_at_val['metrics']['fpr']:.4f} "
              f"FNR {hold_at_val['metrics']['fnr']:.4f}")
        print(f"  latency 1 row {latency_info['one_row_ms']:.2f} ms, "
              f"128 rows {latency_info['batch_128_ms']:.2f} ms; "
              f"n_jobs=1: {latency_single_thread['one_row_ms']:.2f} / "
              f"{latency_single_thread['batch_128_ms']:.2f} ms")

    # Rank on validation F1, then latency.  The holdout is not used for ranking.
    ranked = sorted(
        results.items(),
        key=lambda item: (
            item[1]["eval_val_at_dev_threshold"]["metrics"]["f1"],
            -item[1]["latency"]["single_thread"]["one_row_ms"],
        ),
        reverse=True,
    )
    print("\nRanking by eval-validation F1 (holdout untouched):")
    for name, record in ranked:
        print(f"  {name:<10} val F1 {record['eval_val_at_dev_threshold']['metrics']['f1']:.4f} "
              f"holdout F1 {record['eval_holdout_at_dev_threshold']['metrics']['f1']:.4f} "
              f"one-row {record['latency']['single_thread']['one_row_ms']:.2f} ms")

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "feature_set": FEATURE_SET,
        "target_fpr": TARGET_FPR,
        "results": results,
        "ranking": [name for name, _ in ranked],
    }
    (RESULTS_DIR / "final_model_selection.json").write_text(
        json.dumps(payload, indent=2, default=tm._json_default), encoding="utf-8"
    )
    print(f"\nSaved {RESULTS_DIR / 'final_model_selection.json'}")


if __name__ == "__main__":
    main()

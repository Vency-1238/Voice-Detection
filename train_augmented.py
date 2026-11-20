"""
Train the production model with codec/resampling augmentation.

This is the recommended final training step.  It combines

    train_all_full.npz              (native 16 kHz FLAC features)
    train_aug_balanced_*_full.npz   (44.1 kHz + MP3 round-trip features)

so that a genuine recording that was saved as MP3 or resampled is not mistaken
for a spoof.  The augmentation cache is built by `augment_training.py` from the
training split only.  Dev and eval files are never augmented.

Run:
    python augment_training.py --count 2580 --balanced --formats resample44,mp3
    python train_augmented.py

The script writes `models/audio_deepfake_model.joblib` and
`results/augmented_model_report.json`.
"""

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

import audio_features as af
import config
import train_model as tm

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
BENCH_DIR = BASE_DIR / "benchmarks"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_cache(path):
    with np.load(path, allow_pickle=False) as data:
        return {
            "X": data["X"],
            "y": data["y"].astype(int),
            "speaker": data["speaker"],
            "path": data["path"],
            "feature_names": [str(name) for name in data["feature_names"]],
        }


def find_augmentation_cache(requested=None):
    if requested:
        path = Path(requested)
        if not path.exists():
            raise SystemExit(f"Augmentation cache not found: {path}")
        return path

    candidates = sorted(
        BENCH_DIR.glob("train_aug_balanced_*_full.npz"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            "No balanced augmentation cache found. Run:\n"
            "  python augment_training.py --count 2580 --balanced --formats resample44,mp3"
        )
    return candidates[0]


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


def per_attack(paths, y_true, probabilities, threshold):
    mapping = {}
    for row in tm.read_split("eval"):
        mapping[Path(row["path"]).stem] = row["attack"]
    attacks = np.asarray([mapping.get(Path(p).stem, "-") for p in paths])
    report = {}
    for attack in sorted(set(attacks)):
        mask = attacks == attack
        if not mask.any():
            continue
        if attack == "-":
            report["bonafide"] = {
                "n": int(mask.sum()),
                "false_positive_rate": round(
                    float((probabilities[mask] >= threshold).mean()), 4
                ),
            }
        else:
            report[attack] = {
                "n": int(mask.sum()),
                "recall": round(float((probabilities[mask] >= threshold).mean()), 4),
                "mean_spoof_score": round(float(probabilities[mask].mean()) * 100, 2),
            }
    return report


def scored(model, calibrator, X, y, threshold, bonafide_threshold):
    raw = model.predict_proba(X)[:, 1]
    calibrated = af._apply_calibrator(raw, calibrator)
    return {
        "metrics": tm.classification_metrics(y, calibrated, threshold),
        "three_way": three_way(y, calibrated, threshold, bonafide_threshold),
        "calibration": tm.calibration_metrics(y, calibrated),
        "probabilities": calibrated,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aug-cache", default=None)
    parser.add_argument("--target-fpr", type=float, default=0.02)
    parser.add_argument("--model", default="extra150")
    parser.add_argument("--feature-set", default="full")
    args = parser.parse_args()

    train = load_cache(BENCH_DIR / "train_all_full.npz")
    dev = load_cache(BENCH_DIR / "dev_all_full.npz")
    eval_val = load_cache(BENCH_DIR / "eval_2000_full.npz")
    eval_hold = load_cache(BENCH_DIR / "eval_holdout_2000_full.npz")

    aug_path = find_augmentation_cache(args.aug_cache)
    aug = load_cache(aug_path)
    print(f"augmentation cache: {aug_path.name}  X={aug['X'].shape}")

    X_native = tm.slice_feature_set(train, args.feature_set)
    X_aug = tm.slice_feature_set(aug, args.feature_set)
    X_train = np.vstack([X_native, X_aug])
    y_train = np.concatenate([train["y"], aug["y"]])
    speakers = np.concatenate([train["speaker"], aug["speaker"]])

    X_dev = tm.slice_feature_set(dev, args.feature_set)
    y_dev = dev["y"]
    tune, test = tm.split_dev_speakers(
        [{"speaker": s, "label": int(label)}
         for s, label in zip(dev["speaker"], y_dev)]
    )
    tune_mask = np.isin(dev["speaker"], list(tune))
    test_mask = np.isin(dev["speaker"], list(test))

    print(f"train rows: {len(y_train):,} "
          f"({len(train['y']):,} native + {len(aug['y']):,} augmented)")
    print(f"dev tune   : {int(tune_mask.sum()):,} rows, dev test {int(test_mask.sum()):,} rows")

    model = tm.make_model(args.model)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started

    raw_tune = model.predict_proba(X_dev[tune_mask])[:, 1]
    calibrator = tm.fit_platt_calibrator(raw_tune, y_dev[tune_mask])
    p_tune = af._apply_calibrator(raw_tune, calibrator)
    threshold, threshold_info = tm.threshold_for_fpr(
        y_dev[tune_mask], p_tune, args.target_fpr
    )
    bonafide_threshold = 0.5 if threshold >= 0.5 else max(0.0, threshold - 0.2)

    dev_test = scored(
        model, calibrator, X_dev[test_mask], y_dev[test_mask],
        threshold, bonafide_threshold,
    )
    val = scored(
        model, calibrator, tm.slice_feature_set(eval_val, args.feature_set),
        eval_val["y"], threshold, bonafide_threshold,
    )
    hold = scored(
        model, calibrator, tm.slice_feature_set(eval_hold, args.feature_set),
        eval_hold["y"], threshold, bonafide_threshold,
    )

    print(f"trained {args.model} in {train_seconds:.1f}s")
    print(f"threshold {threshold:.4f} (bonafide {bonafide_threshold:.4f}, "
          f"target FPR {args.target_fpr:.1%})")
    for name, block in (("dev test", dev_test), ("eval validation", val),
                        ("eval holdout", hold)):
        m = block["metrics"]
        print(f"  {name:<16} F1={m['f1']:.4f} FPR={m['fpr']:.4f} "
              f"FNR={m['fnr']:.4f} AUC={m['roc_auc']:.4f}")

    bundle = {
        "model": model,
        "calibrator": calibrator,
        "feature_set": args.feature_set,
        "feature_names": af.feature_names(args.feature_set),
        "n_features": len(af.feature_names(args.feature_set)),
        "sample_rate": af.SAMPLE_RATE,
        "chunk_seconds": af.CHUNK_SECONDS,
        "n_mfcc": af.N_MFCC,
        "threshold": threshold,
        "spoof_threshold": threshold,
        "bonafide_threshold": bonafide_threshold,
        "model_name": args.model,
        "trained_on_files": int(len(y_train)),
        "trained_on_speakers": int(len(set(speakers))),
        "target_fpr": float(args.target_fpr),
        "augmentation": {
            "cache": aug_path.name,
            "rows": int(len(aug["y"])),
            "native_rows": int(len(train["y"])),
            "kinds": ["44.1 kHz resample round-trip", "MP3 44.1 kHz round-trip"],
        },
        "validation": {
            "dev_test": dev_test["metrics"],
            "dev_test_calibration": dev_test["calibration"],
            "dev_test_three_way": dev_test["three_way"],
            "eval_validation": val["metrics"],
            "eval_validation_three_way": val["three_way"],
            "eval_holdout": hold["metrics"],
            "eval_holdout_three_way": hold["three_way"],
            "eval_holdout_calibration": hold["calibration"],
            "threshold_tuning": threshold_info,
        },
        "eval": {
            "dataset": "held-out eval files, unseen attacks A07-A19",
            "metrics": hold["metrics"],
            "calibration": hold["calibration"],
            "per_attack": per_attack(
                eval_hold["path"], eval_hold["y"], hold["probabilities"], threshold
            ),
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "version": 3,
        "feature_description": (
            "mean/std of MFCC, delta, delta-delta and spectral features; "
            "one shared STFT per chunk; trained with resample/MP3 augmentation"
        ),
    }

    joblib.dump(bundle, config.MODEL_PATH)
    print(f"\nSaved model: {config.MODEL_PATH}")

    # End-to-end speed on 40 real holdout files.
    paths = [str(p) for p in eval_hold["path"][:40]]
    af.predict_audio_files(paths, bundle)
    started = time.perf_counter()
    af.predict_audio_files(paths, bundle)
    batch_ms = (time.perf_counter() - started) / len(paths) * 1000
    single = []
    for path in paths[:20]:
        started = time.perf_counter()
        af.predict_audio_file(path, bundle)
        single.append((time.perf_counter() - started) * 1000)
    print(f"speed: single-file mean {np.mean(single):.1f} ms, "
          f"batch {batch_ms:.1f} ms/file")

    report = {
        "augmentation_cache": aug_path.name,
        "native_rows": int(len(train["y"])),
        "augmented_rows": int(len(aug["y"])),
        "total_rows": int(len(y_train)),
        "model": args.model,
        "feature_set": args.feature_set,
        "train_seconds": round(train_seconds, 2),
        "threshold": round(float(threshold), 6),
        "bonafide_threshold": round(float(bonafide_threshold), 6),
        "threshold_tuning": threshold_info,
        "dev_test": dev_test["metrics"],
        "dev_test_three_way": dev_test["three_way"],
        "eval_validation": val["metrics"],
        "eval_validation_three_way": val["three_way"],
        "eval_holdout": hold["metrics"],
        "eval_holdout_three_way": hold["three_way"],
        "eval_holdout_calibration": hold["calibration"],
        "eval_holdout_per_attack": bundle["eval"]["per_attack"],
        "speed": {
            "single_file_ms_mean": round(float(np.mean(single)), 2),
            "batch_per_file_ms": round(float(batch_ms), 2),
        },
    }
    (RESULTS_DIR / "augmented_model_report.json").write_text(
        json.dumps(report, indent=2, default=tm._json_default), encoding="utf-8"
    )
    print(f"Saved report: {RESULTS_DIR / 'augmented_model_report.json'}")


if __name__ == "__main__":
    main()

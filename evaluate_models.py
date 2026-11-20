"""
Compare feature sets and classifiers on three different views of the data:

    dev tune  : speakers never used for training (threshold tuning)
    dev test  : different speakers again (speaker-independent report)
    eval val  : unseen attacks A07-A12 + held-out bonafide speakers
    eval test : unseen attacks A13-A19 + the other bonafide speakers

Why this file exists
--------------------
The first comparison showed almost perfect dev scores.  That is expected:
ASVspoof train and dev contain the SAME attack types (A01-A06), so a model can
learn the vocoder signature.  The honest test is the eval split, which contains
attacks A07-A19 that never appear in training.

Run:
    python evaluate_models.py
"""

import json
from pathlib import Path

import numpy as np

import audio_features as af
import train_model as tm

BASE_DIR = Path(__file__).resolve().parent
BENCH_DIR = BASE_DIR / "benchmarks"
RESULTS_DIR = BASE_DIR / "results"

TRAIN_CACHE = BENCH_DIR / "train_4000_full.npz"
DEV_CACHE = BENCH_DIR / "dev_all_full.npz"
EVAL_CACHE = BENCH_DIR / "eval_2000_full.npz"

EVAL_VAL_ATTACKS = {"A07", "A08", "A09", "A10", "A11", "A12"}
EVAL_TEST_ATTACKS = {"A13", "A14", "A15", "A16", "A17", "A18", "A19"}
TARGET_FPR = 0.02
SEED = 42

FEATURE_SETS = (
    "base90",
    "flat_flux",
    "delta2",
    "flat_flux_delta2",
    "contrast",
    "flat_flux_contrast",
    "full",
)
MODELS = ("rf300", "extra150", "extra300", "extra500", "histgb", "logreg", "svc")


def load_cache(path):
    with np.load(path, allow_pickle=False) as data:
        return {
            "X": data["X"],
            "y": data["y"].astype(int),
            "speaker": data["speaker"],
            "path": data["path"],
            "feature_names": [str(name) for name in data["feature_names"]],
        }


def eval_masks(cache):
    """Attack-disjoint and bonafide-speaker-disjoint eval split."""
    protocol = tm.read_split("eval")
    attack_by_stem = {Path(row["path"]).stem: row["attack"] for row in protocol}
    attacks = np.asarray(
        [attack_by_stem.get(Path(path).stem, "-") for path in cache["path"]]
    )
    y = cache["y"]
    speaker = cache["speaker"]

    bonafide_speakers = sorted(set(speaker[y == 0]))
    rng = np.random.default_rng(SEED)
    rng.shuffle(bonafide_speakers)
    val_speakers = set(bonafide_speakers[: len(bonafide_speakers) // 2])

    bonafide_val = (y == 0) & np.isin(speaker, list(val_speakers))
    bonafide_test = (y == 0) & ~np.isin(speaker, list(val_speakers))
    spoof_val = (y == 1) & np.isin(attacks, list(EVAL_VAL_ATTACKS))
    spoof_test = (y == 1) & np.isin(attacks, list(EVAL_TEST_ATTACKS))

    val = bonafide_val | spoof_val
    test = bonafide_test | spoof_test
    return val, test, attacks


def calibrated_eval(model, X_fit, y_fit, X_score, y_score, threshold):
    raw = model.predict_proba(X_score)[:, 1]
    calibrator = tm.fit_platt_calibrator(
        model.predict_proba(X_fit)[:, 1], y_fit
    )
    calibrated = af._apply_calibrator(raw, calibrator)
    return calibrated, tm.classification_metrics(y_score, calibrated, threshold)


def run_candidate(feature_set, model_name, train, dev, eval_cache, masks):
    X_train = tm.slice_feature_set(train, feature_set)
    y_train = train["y"]

    X_dev = tm.slice_feature_set(dev, feature_set)
    y_dev = dev["y"]
    dev_speakers = dev["speaker"]

    tune_speakers, test_speakers = tm.split_dev_speakers(
        [{"speaker": s, "label": int(label)}
         for s, label in zip(dev_speakers, y_dev)]
    )
    tune_mask = np.isin(dev_speakers, list(tune_speakers))
    test_mask = np.isin(dev_speakers, list(test_speakers))

    model = tm.make_model(model_name)
    model.fit(X_train, y_train)

    raw_tune = model.predict_proba(X_dev[tune_mask])[:, 1]
    calibrator = tm.fit_platt_calibrator(raw_tune, y_dev[tune_mask])
    p_tune = af._apply_calibrator(raw_tune, calibrator)
    threshold, threshold_info = tm.threshold_for_fpr(
        y_dev[tune_mask], p_tune, TARGET_FPR
    )

    result = {
        "feature_set": feature_set,
        "model": model_name,
        "n_features": len(af.feature_names(feature_set)),
        "threshold": round(float(threshold), 6),
        "threshold_info": threshold_info,
        "dev_tune": tm.classification_metrics(y_dev[tune_mask], p_tune, threshold),
    }

    # Dev test (speaker-independent, seen attack types)
    p_test = af._apply_calibrator(
        model.predict_proba(X_dev[test_mask])[:, 1], calibrator
    )
    result["dev_test"] = tm.classification_metrics(y_dev[test_mask], p_test, threshold)

    # Eval validation (unseen attacks A07-A12)
    X_eval = tm.slice_feature_set(eval_cache, feature_set)
    y_eval = eval_cache["y"]
    val_mask, test_mask_eval, attacks = masks
    p_val = af._apply_calibrator(model.predict_proba(X_eval[val_mask])[:, 1], calibrator)
    result["eval_val"] = tm.classification_metrics(y_eval[val_mask], p_val, threshold)
    result["eval_val"]["attacks"] = sorted(
        set(attacks[val_mask]) - {"-"}
    )

    # Threshold needed on eval val to reach specific FPRs (oracle, for diagnosis)
    result["eval_val_fpr_targets"] = {}
    for target in (0.005, 0.01, 0.02, 0.05):
        th, info = tm.threshold_for_fpr(y_eval[val_mask], p_val, target)
        result["eval_val_fpr_targets"][str(target)] = {
            "threshold": round(float(th), 6),
            **tm.classification_metrics(y_eval[val_mask], p_val, th),
        }

    # Eval test is not used for selection; it is filled in for the chosen model
    # later so the final report stays honest.
    result["eval_test"] = None

    return model, calibrator, result


def main():
    train = load_cache(TRAIN_CACHE)
    dev = load_cache(DEV_CACHE)
    eval_cache = load_cache(EVAL_CACHE)
    val_mask, test_mask, attacks = eval_masks(eval_cache)

    y_eval = eval_cache["y"]
    print("eval validation :", int(val_mask.sum()), "rows",
          int((y_eval[val_mask] == 0).sum()), "bonafide /",
          int((y_eval[val_mask] == 1).sum()), "spoof")
    print("eval test       :", int(test_mask.sum()), "rows",
          int((y_eval[test_mask] == 0).sum()), "bonafide /",
          int((y_eval[test_mask] == 1).sum()), "spoof")

    report = {
        "target_fpr": TARGET_FPR,
        "eval_val_attacks": sorted(EVAL_VAL_ATTACKS),
        "eval_test_attacks": sorted(EVAL_TEST_ATTACKS),
        "feature_sets": {},
        "models": {},
    }

    print("\nFeature sets (ExtraTrees, selected on dev tune, checked on eval val):")
    print(f"{'feature set':<22}{'#f':>4}{'devAUC':>9}{'valAUC':>9}"
          f"{'valF1':>9}{'valFPR':>9}{'valFNR':>9}")
    for feature_set in FEATURE_SETS:
        _, _, result = run_candidate(
            feature_set, "extra300", train, dev, eval_cache, (val_mask, test_mask, attacks)
        )
        report["feature_sets"][feature_set] = result
        print(
            f"{feature_set:<22}{result['n_features']:>4}"
            f"{result['dev_test']['roc_auc']:>9.4f}"
            f"{result['eval_val']['roc_auc']:>9.4f}"
            f"{result['eval_val']['f1']:>9.4f}"
            f"{result['eval_val']['fpr']:>9.4f}"
            f"{result['eval_val']['fnr']:>9.4f}"
        )

    # Rank by unseen-attack AUC, then F1 at the target FPR.
    ranked_features = sorted(
        report["feature_sets"].items(),
        key=lambda item: (
            item[1]["eval_val"]["roc_auc"],
            item[1]["eval_val"]["f1"],
        ),
        reverse=True,
    )
    best_feature_set = ranked_features[0][0]
    print(f"\nBest feature set on unseen attacks: {best_feature_set}")

    print(f"\nClassifiers on {best_feature_set} (checked on eval val):")
    print(f"{'model':<12}{'devAUC':>9}{'valAUC':>9}{'valF1':>9}"
          f"{'valFPR':>9}{'valFNR':>9}{'train_s':>9}")
    for model_name in MODELS:
        _, _, result = run_candidate(
            best_feature_set, model_name, train, dev, eval_cache,
            (val_mask, test_mask, attacks),
        )
        report["models"][model_name] = result
        print(
            f"{model_name:<12}{result['dev_test']['roc_auc']:>9.4f}"
            f"{result['eval_val']['roc_auc']:>9.4f}"
            f"{result['eval_val']['f1']:>9.4f}"
            f"{result['eval_val']['fpr']:>9.4f}"
            f"{result['eval_val']['fnr']:>9.4f}"
            f"{result['threshold_info'].get('f1_on_tune', 0):>9.4f}"
        )

    ranked_models = sorted(
        report["models"].items(),
        key=lambda item: (
            item[1]["eval_val"]["roc_auc"],
            item[1]["eval_val"]["f1"],
        ),
        reverse=True,
    )
    best_model = ranked_models[0][0]
    print(f"\nBest classifier on unseen attacks: {best_model}")

    report["best_feature_set"] = best_feature_set
    report["best_model"] = best_model

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "feature_model_selection.json").write_text(
        json.dumps(report, indent=2, default=tm._json_default), encoding="utf-8"
    )
    print(f"\nSaved {RESULTS_DIR / 'feature_model_selection.json'}")


if __name__ == "__main__":
    main()

"""
Honest final evaluation and measured comparisons.

This script produces every number used in the README and the report.  Nothing is
hard-coded: it trains (or loads) the models and measures them on this machine.

Data roles
----------
train (A01-A06)     : model fitting only
dev tune speakers   : calibration + threshold tuning only
dev test speakers   : speaker-independent report (same attack types as train)
eval validation     : unseen attacks A07-A19, used during model selection
eval holdout        : different files with unseen attacks A07-A19, report only

Also measures
-------------
* old pipeline (RF300 + 90 features + 4,000 files) vs new pipeline
  (ExtraTrees150 + 172 features + 25,380 files), end-to-end
* 5-second vs 10-second chunking on long synthetic recordings
* WAV / FLAC / MP3 / OGG round-trips
* empty, corrupted and too-short files

Run:
    python final_evaluation.py
"""

import json
import shutil
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

import audio_features as af
import config
import evaluate_models as em
import train_model as tm

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
BENCH_DIR = BASE_DIR / "benchmarks"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TARGET_FPR = 0.02
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_cache(path):
    with np.load(path, allow_pickle=False) as data:
        return {
            "X": data["X"],
            "y": data["y"].astype(int),
            "speaker": data["speaker"],
            "path": data["path"],
            "feature_names": [str(name) for name in data["feature_names"]],
        }


def dev_masks(dev):
    tune, test = tm.split_dev_speakers(
        [{"speaker": s, "label": int(label)}
         for s, label in zip(dev["speaker"], dev["y"])]
    )
    return (
        np.isin(dev["speaker"], list(tune)),
        np.isin(dev["speaker"], list(test)),
    )


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
    metrics = tm.classification_metrics(y, calibrated, spoof_threshold)
    return {
        "metrics": metrics,
        "three_way": three_way(y, calibrated, spoof_threshold, bonafide_threshold),
        "calibration": tm.calibration_metrics(y, calibrated),
        "probabilities": calibrated,
    }


def public(block):
    return {key: value for key, value in block.items() if key != "probabilities"}


def attack_by_stem():
    mapping = {}
    for row in tm.read_split("eval"):
        mapping[Path(row["path"]).stem] = row["attack"]
    return mapping


def per_attack_report(paths, y_true, probabilities, threshold):
    mapping = attack_by_stem()
    attacks = np.asarray([mapping.get(Path(p).stem, "-") for p in paths])
    report = {}
    for attack in sorted(set(attacks)):
        mask = attacks == attack
        if not mask.any():
            continue
        if attack == "-":
            report["bonafide"] = {
                "n": int(mask.sum()),
                "mean_spoof_score": round(float(probabilities[mask].mean()) * 100, 2),
                "false_positive_rate": round(
                    float((probabilities[mask] >= threshold).mean()), 4
                ),
            }
        else:
            report[attack] = {
                "n": int(mask.sum()),
                "mean_spoof_score": round(float(probabilities[mask].mean()) * 100, 2),
                "recall": round(float((probabilities[mask] >= threshold).mean()), 4),
            }
    return report


def benchmark(fn, repeats=3):
    fn()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        times.append((time.perf_counter() - started) * 1000)
    return {
        "mean_ms": round(float(np.mean(times)), 2),
        "median_ms": round(float(np.median(times)), 2),
        "min_ms": round(float(np.min(times)), 2),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_old_pipeline(train_4k, dev, tune_mask):
    """Reproduce the original shipped recipe: RF300 + 90 features + 4k files."""
    X_train = tm.slice_feature_set(train_4k, "base90")
    y_train = train_4k["y"]
    X_dev = tm.slice_feature_set(dev, "base90")
    y_dev = dev["y"]

    model = tm.make_model("rf300")
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started

    raw_tune = model.predict_proba(X_dev[tune_mask])[:, 1]
    calibrator = tm.fit_platt_calibrator(raw_tune, y_dev[tune_mask])
    p_tune = af._apply_calibrator(raw_tune, calibrator)
    threshold, info = tm.threshold_for_fpr(y_dev[tune_mask], p_tune, TARGET_FPR)
    bonafide_threshold = 0.5 if threshold >= 0.5 else max(0.0, threshold - 0.2)

    bundle = {
        "model": model,
        "calibrator": calibrator,
        "feature_set": "base90",
        "feature_names": af.feature_names("base90"),
        "n_features": 90,
        "sample_rate": af.SAMPLE_RATE,
        "chunk_seconds": af.CHUNK_SECONDS,
        "threshold": threshold,
        "spoof_threshold": threshold,
        "bonafide_threshold": bonafide_threshold,
        "model_name": "rf300",
    }
    return bundle, {
        "train_seconds": round(train_seconds, 2),
        "threshold": round(float(threshold), 6),
        "bonafide_threshold": bonafide_threshold,
        "threshold_info": info,
    }


def build_new_pipeline(train_all, dev, tune_mask):
    """The bundle that actually ships, retrained from the same caches.

    If a balanced codec/resampling augmentation cache exists, it is included so
    the measured model is the same one that ships.
    """
    X_train = tm.slice_feature_set(train_all, "full")
    y_train = train_all["y"]
    speakers = train_all["speaker"]
    augmentation = None

    aug_candidates = sorted(
        BENCH_DIR.glob("train_aug_balanced_*_full.npz"),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    if aug_candidates:
        aug = load_cache(aug_candidates[0])
        X_train = np.vstack([X_train, tm.slice_feature_set(aug, "full")])
        y_train = np.concatenate([y_train, aug["y"]])
        speakers = np.concatenate([speakers, aug["speaker"]])
        augmentation = {
            "cache": aug_candidates[0].name,
            "native_rows": int(len(train_all["y"])),
            "augmented_rows": int(len(aug["y"])),
        }

    X_dev = tm.slice_feature_set(dev, "full")
    y_dev = dev["y"]

    model = tm.make_model("extra150")
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started

    raw_tune = model.predict_proba(X_dev[tune_mask])[:, 1]
    calibrator = tm.fit_platt_calibrator(raw_tune, y_dev[tune_mask])
    p_tune = af._apply_calibrator(raw_tune, calibrator)
    threshold, info = tm.threshold_for_fpr(y_dev[tune_mask], p_tune, TARGET_FPR)
    bonafide_threshold = 0.5 if threshold >= 0.5 else max(0.0, threshold - 0.2)

    bundle = {
        "model": model,
        "calibrator": calibrator,
        "feature_set": "full",
        "feature_names": af.feature_names("full"),
        "n_features": 172,
        "sample_rate": af.SAMPLE_RATE,
        "chunk_seconds": af.CHUNK_SECONDS,
        "threshold": threshold,
        "spoof_threshold": threshold,
        "bonafide_threshold": bonafide_threshold,
        "model_name": "extra150",
    }
    return bundle, {
        "train_seconds": round(train_seconds, 2),
        "threshold": round(float(threshold), 6),
        "bonafide_threshold": bonafide_threshold,
        "threshold_info": info,
        "total_rows": int(len(y_train)),
        "speakers": int(len(set(speakers))),
        "augmentation": augmentation,
    }


# ---------------------------------------------------------------------------
# Extra checks
# ---------------------------------------------------------------------------
def find_ffmpeg():
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidate = Path(r"C:\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe")
    return str(candidate) if candidate.exists() else None


def format_robustness(bundle, paths, labels):
    ffmpeg = find_ffmpeg()
    # Real uploads are usually 44.1 kHz, not 16 kHz.  Keep a native-rate WAV as
    # the control and test the realistic resampled/lossy variants.
    formats = {
        "wav_16k": (".wav", ["-ar", "16000", "-c:a", "pcm_s16le"]),
        "wav_44k": (".wav", ["-ar", "44100", "-c:a", "pcm_s16le"]),
        "flac_44k": (".flac", ["-ar", "44100", "-c:a", "flac"]),
        "ogg_44k": (".ogg", ["-ar", "44100", "-c:a", "libvorbis", "-q:a", "6"]),
        "mp3_44k": (".mp3", ["-ar", "44100", "-c:a", "libmp3lame", "-b:a", "192k"]),
    }
    output = {"ffmpeg": ffmpeg, "formats": {}}
    if ffmpeg is None:
        output["error"] = "ffmpeg not found"
        return output

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        baseline = af.predict_audio_files(paths, bundle)
        output["original"] = [
            {"file": r["filename"], "prediction": r["prediction"],
             "spoof_score": r["spoof_score"]}
            for r in baseline
        ]

        for name, (extension, codec) in formats.items():
            converted = []
            for index, path in enumerate(paths):
                target = tmp / f"clip_{index}{extension}"
                command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
                           "-ac", "1", *codec, str(target)]
                try:
                    subprocess.run(command, check=True, capture_output=True)
                    converted.append(target)
                except Exception:
                    converted.append(None)

            usable = [(i, p) for i, p in enumerate(converted) if p is not None]
            results = af.predict_audio_files([p for _, p in usable], bundle)

            compared = 0
            label_matches = 0
            score_deltas = []
            decode_errors = 0
            details = []
            for (index, _path), result in zip(usable, results):
                original = baseline[index]
                if result.get("error"):
                    decode_errors += 1
                    details.append({
                        "file": original["filename"],
                        "original": original["prediction"],
                        "converted": None,
                        "error": result["error"],
                    })
                    continue
                compared += 1
                if result["prediction"] == original["prediction"]:
                    label_matches += 1
                score_deltas.append(abs(result["spoof_score"] - original["spoof_score"]))
                details.append({
                    "file": original["filename"],
                    "original": original["prediction"],
                    "converted": result["prediction"],
                    "original_score": original["spoof_score"],
                    "converted_score": result["spoof_score"],
                })

            output["formats"][name] = {
                "encoded": len(usable),
                "decoded": compared,
                "decode_errors": decode_errors,
                "label_agreement_pct": round(label_matches / compared * 100, 2) if compared else None,
                "mean_abs_score_change_pct": round(float(np.mean(score_deltas)), 3)
                if score_deltas else None,
                "files": details,
            }
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("Final evaluation - all numbers measured on this machine")
    print("=" * 78)

    train_4k = load_cache(BENCH_DIR / "train_4000_full.npz")
    train_all = load_cache(BENCH_DIR / "train_all_full.npz")
    dev = load_cache(BENCH_DIR / "dev_all_full.npz")
    eval_val = load_cache(BENCH_DIR / "eval_2000_full.npz")
    eval_hold = load_cache(BENCH_DIR / "eval_holdout_2000_full.npz")

    tune_mask, test_mask = dev_masks(dev)
    print(f"dev tune {int(tune_mask.sum()):,} rows | dev test {int(test_mask.sum()):,} rows")
    print(f"eval validation {len(eval_val['y']):,} rows | eval holdout {len(eval_hold['y']):,} rows")

    old_bundle, old_info = build_old_pipeline(train_4k, dev, tune_mask)
    new_bundle, new_info = build_new_pipeline(train_all, dev, tune_mask)

    y_dev = dev["y"]
    report = {
        "target_fpr": TARGET_FPR,
        "old_pipeline": {"info": old_info, "feature_set": "base90", "model": "rf300",
                         "train_files": int(len(train_4k["y"]))},
        "new_pipeline": {"info": new_info, "feature_set": "full", "model": "extra150",
                         "train_files": int(new_info.get("total_rows", len(train_all["y"])))},
        "splits": {},
    }

    X_old_dev = tm.slice_feature_set(dev, "base90")
    X_old_val = tm.slice_feature_set(eval_val, "base90")
    X_old_hold = tm.slice_feature_set(eval_hold, "base90")
    X_new_dev = tm.slice_feature_set(dev, "full")
    X_new_val = tm.slice_feature_set(eval_val, "full")
    X_new_hold = tm.slice_feature_set(eval_hold, "full")

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------
    print("\nMetrics at the 2% dev-tuned false-positive target:")
    print(f"{'split':<22}{'old F1':>9}{'old FPR':>9}{'old FNR':>9}"
          f"{'new F1':>9}{'new FPR':>9}{'new FNR':>9}")
    for name, X_old, X_new, y in (
        ("dev tune", X_old_dev[tune_mask], X_new_dev[tune_mask], y_dev[tune_mask]),
        ("dev test", X_old_dev[test_mask], X_new_dev[test_mask], y_dev[test_mask]),
        ("eval validation", X_old_val, X_new_val, eval_val["y"]),
        ("eval holdout", X_old_hold, X_new_hold, eval_hold["y"]),
    ):
        old = evaluate(old_bundle["model"], old_bundle["calibrator"], X_old, y,
                       old_bundle["spoof_threshold"], old_bundle["bonafide_threshold"])
        new = evaluate(new_bundle["model"], new_bundle["calibrator"], X_new, y,
                       new_bundle["spoof_threshold"], new_bundle["bonafide_threshold"])
        report["splits"][name] = {"old": public(old), "new": public(new)}
        print(f"{name:<22}{old['metrics']['f1']:>9.4f}{old['metrics']['fpr']:>9.4f}"
              f"{old['metrics']['fnr']:>9.4f}{new['metrics']['f1']:>9.4f}"
              f"{new['metrics']['fpr']:>9.4f}{new['metrics']['fnr']:>9.4f}")

    # Per-attack breakdown on the holdout
    new_hold = evaluate(
        new_bundle["model"], new_bundle["calibrator"], X_new_hold, eval_hold["y"],
        new_bundle["spoof_threshold"], new_bundle["bonafide_threshold"],
    )
    report["eval_holdout_per_attack"] = per_attack_report(
        eval_hold["path"], eval_hold["y"], new_hold["probabilities"],
        new_bundle["spoof_threshold"],
    )

    print("\nPer-attack recall on the unseen-attack holdout:")
    for attack, values in report["eval_holdout_per_attack"].items():
        if attack == "bonafide":
            print(f"  bonafide  n={values['n']:<5} false-positive rate {values['false_positive_rate']:.4f}")
        else:
            print(f"  {attack:<8} n={values['n']:<5} recall {values['recall']:.4f}  "
                  f"mean score {values['mean_spoof_score']:.2f}%")

    # ---------------------------------------------------------------
    # Speed: old vs new, same 40 files
    # ---------------------------------------------------------------
    sample_paths = [str(p) for p in eval_hold["path"][:40]]

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "legacy_features", BASE_DIR / "legacy" / "audio_features_v1.py"
    )
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)

    def old_single_file():
        for path in sample_paths:
            legacy.predict_audio_file(path, old_bundle)

    def new_batch():
        af.predict_audio_files(sample_paths, new_bundle)

    old_speed = benchmark(old_single_file, repeats=2)
    new_speed = benchmark(new_batch, repeats=3)

    # Pure feature extraction, 100 files, I/O included.
    extraction_paths = sample_paths[:40]
    legacy.extract_features_from_file(extraction_paths[0])
    started = time.perf_counter()
    for path in extraction_paths:
        legacy.extract_features_from_file(path)
    old_extract_ms = (time.perf_counter() - started) / len(extraction_paths) * 1000

    af.extract_features_from_file(extraction_paths[0], feature_set="full")
    started = time.perf_counter()
    for path in extraction_paths:
        af.extract_features_from_file(path, feature_set="full")
    new_extract_ms = (time.perf_counter() - started) / len(extraction_paths) * 1000

    # Model-only latency.
    one_old = X_old_hold[:1]
    one_new = X_new_hold[:1]
    old_bundle["model"].predict_proba(one_old)
    new_bundle["model"].predict_proba(one_new)
    old_model = benchmark(lambda: old_bundle["model"].predict_proba(one_old), repeats=10)
    new_model = benchmark(lambda: new_bundle["model"].predict_proba(one_new), repeats=10)

    report["speed"] = {
        "files": len(sample_paths),
        "old_end_to_end": old_speed,
        "new_end_to_end_batch": new_speed,
        "old_per_file_ms": round(old_speed["mean_ms"] / len(sample_paths), 2),
        "new_per_file_ms": round(new_speed["mean_ms"] / len(sample_paths), 2),
        "old_feature_extraction_ms": round(old_extract_ms, 2),
        "new_feature_extraction_ms": round(new_extract_ms, 2),
        "old_model_one_row": old_model,
        "new_model_one_row": new_model,
    }

    print(f"\nSpeed on {len(sample_paths)} real holdout files (warm):")
    print(f"  old pipeline : {old_speed['mean_ms']:>8.2f} ms total, "
          f"{report['speed']['old_per_file_ms']:>6.2f} ms/file")
    print(f"  new pipeline : {new_speed['mean_ms']:>8.2f} ms total, "
          f"{report['speed']['new_per_file_ms']:>6.2f} ms/file")
    print(f"  feature only : old {old_extract_ms:.2f} ms -> new {new_extract_ms:.2f} ms per file")
    print(f"  model 1 row  : old {old_model['mean_ms']:.2f} ms -> new {new_model['mean_ms']:.2f} ms")

    # ---------------------------------------------------------------
    # 5 s vs 10 s chunking on long synthetic recordings
    # ---------------------------------------------------------------
    chunk_report = {"recordings": 0, "chunk_seconds": {}}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rng = np.random.default_rng(SEED)
        long_files = []
        for label in (0, 1):
            pool = [str(p) for p, y in zip(eval_hold["path"], eval_hold["y"]) if y == label]
            chosen = [pool[i] for i in rng.choice(len(pool), size=12, replace=False)]
            for index in range(12):
                pieces = chosen[index * 3:(index * 3) + 3] if index * 3 + 3 <= len(chosen) else chosen[:3]
                waves = []
                for path in pieces:
                    y, _ = af.load_audio(path, max_seconds=4.0)
                    waves.append(y)
                combined = np.concatenate(waves)
                target = tmp / f"long_{label}_{index}.wav"
                sf.write(target, combined, af.SAMPLE_RATE)
                long_files.append((str(target), label))
        chunk_report["recordings"] = len(long_files)
        chunk_report["mean_seconds"] = round(
            float(np.mean([af.audio_duration(p) for p, _ in long_files])), 2
        )
        chunk_paths = [p for p, _ in long_files]
        chunk_labels = np.asarray([label for _, label in long_files])

        for seconds in (5.0, 10.0):
            bundle = dict(new_bundle)
            bundle["chunk_seconds"] = seconds
            started = time.perf_counter()
            results = af.predict_audio_files(chunk_paths, bundle)
            elapsed = (time.perf_counter() - started) * 1000

            predictions = np.asarray([
                {"SPOOF": 1, "BONAFIDE": 0}.get(r.get("prediction"), -1)
                for r in results
            ])
            decided = predictions >= 0
            correct = int(np.sum(predictions == chunk_labels))
            chunk_report["chunk_seconds"][str(seconds)] = {
                "accuracy": round(correct / len(chunk_labels), 4),
                "uncertain_rate": round(float(np.mean(~decided)), 4),
                "mean_ms_per_file": round(elapsed / len(chunk_paths), 2),
                "mean_chunks_per_file": round(
                    float(np.mean([r.get("chunks_analysed", 0) for r in results])), 2
                ),
                "results": [
                    {"truth": int(t), "prediction": r.get("prediction"),
                     "score": r.get("spoof_score"), "chunks": r.get("chunks_analysed")}
                    for t, r in zip(chunk_labels, results)
                ],
            }
    report["chunking"] = chunk_report
    print(f"\nChunk length on {chunk_report['recordings']} long recordings "
          f"(mean {chunk_report['mean_seconds']} s):")
    for seconds, values in chunk_report["chunk_seconds"].items():
        print(f"  {seconds:>4}s windows: accuracy {values['accuracy']:.4f}, "
              f"uncertain {values['uncertain_rate']:.2%}, "
              f"{values['mean_ms_per_file']:.1f} ms/file, "
              f"{values['mean_chunks_per_file']:.1f} chunks/file")

    # ---------------------------------------------------------------
    # Formats
    # ---------------------------------------------------------------
    # Use a class-balanced sample so the agreement number is not dominated by
    # whichever class happened to come first in the cache.
    format_paths, format_labels = [], []
    format_rng = np.random.default_rng(SEED)
    for label in (0, 1):
        pool = [
            str(path)
            for path, truth in zip(eval_hold["path"], eval_hold["y"])
            if int(truth) == label
        ]
        chosen = format_rng.choice(len(pool), size=min(10, len(pool)), replace=False)
        format_paths.extend(pool[index] for index in chosen)
        format_labels.extend([label] * len(chosen))
    report["formats"] = format_robustness(new_bundle, format_paths, format_labels)
    print(f"\nFormat round-trip ({len(format_paths)} files encoded with ffmpeg, "
          f"then re-analysed):")
    if "error" in report["formats"]:
        print("  ", report["formats"]["error"])
    else:
        for extension, values in report["formats"]["formats"].items():
            print(f"  {extension:<8} decoded {values['decoded']}/{values['encoded']}, "
                  f"label agreement {values['label_agreement_pct']}%, "
                  f"mean |score change| {values['mean_abs_score_change_pct']}%")

    # ---------------------------------------------------------------
    # Bad files
    # ---------------------------------------------------------------
    checks = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sr = af.SAMPLE_RATE
        t = np.arange(int(1.0 * sr), dtype=np.float32) / sr
        tone = (0.05 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        empty = tmp / "empty.wav"; empty.write_bytes(b"")
        corrupt = tmp / "corrupt.wav"; corrupt.write_bytes(b"this is not audio" * 100)
        short = tmp / "short.wav"; sf.write(short, tone[: int(0.1 * sr)], sr)
        valid = tmp / "valid_1s.wav"; sf.write(valid, tone, sr)
        stereo = tmp / "stereo.wav"; sf.write(stereo, np.stack([tone, tone], axis=1), sr)
        for path in (empty, corrupt, short, valid, stereo):
            result = af.predict_audio_files([path], new_bundle)[0]
            if result.get("error"):
                checks[path.name] = {"ok": False, "error": result["error"][:110]}
            else:
                checks[path.name] = {"ok": True, "prediction": result["prediction"],
                                     "score": result["spoof_score"]}
    report["bad_files"] = checks
    print("\nBad / edge-case files:")
    for name, values in checks.items():
        print(f"  {name:<16} {'analysed' if values['ok'] else 'rejected'}"
              f"  {values.get('prediction') or values.get('error')}")

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    (RESULTS_DIR / "final_evaluation.json").write_text(
        json.dumps(report, indent=2, default=tm._json_default), encoding="utf-8"
    )
    print(f"\nSaved {RESULTS_DIR / 'final_evaluation.json'}")


if __name__ == "__main__":
    main()

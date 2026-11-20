"""
Build codec/resampling-augmented training features.

Why this exists
---------------
A model trained only on native 16 kHz FLAC files learns that a sharp
high-frequency roll-off means "spoof".  A genuine recording that was saved as
44.1 kHz MP3, or simply resampled, has a similar roll-off and is then wrongly
flagged as AI-generated.  The same artefact therefore has to appear on BOTH
classes during training.

This script converts a subset of the training split and extracts the same
features used by the app:

    resample22  : 16 kHz -> 22.05 kHz -> 16 kHz
    resample44  : 16 kHz -> 44.1 kHz -> 16 kHz
    mp3         : real MP3 encode at 44.1 kHz, decoded back to 16 kHz
    ogg         : real OGG encode at 44.1 kHz, decoded back to 16 kHz

The resulting cache is combined with the normal train cache by
`train_augmented.py`.  Dev/eval files are never augmented or used here.

Run:
    python augment_training.py --count 4000
"""

import argparse
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

import audio_features as af
import train_model as tm

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
BENCH_DIR = BASE_DIR / "benchmarks"
BENCH_DIR.mkdir(exist_ok=True)


def find_ffmpeg():
    import shutil

    found = shutil.which("ffmpeg")
    if found:
        return found
    candidate = Path(r"C:\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe")
    return str(candidate) if candidate.exists() else None


def encode(source, target, ffmpeg):
    extension = target.suffix.lower()
    codec = {
        ".mp3": ["-c:a", "libmp3lame", "-b:a", "192k"],
        ".ogg": ["-c:a", "libvorbis", "-q:a", "6"],
    }[extension]
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
         "-ar", "44100", "-ac", "1", *codec, str(target)],
        check=True,
        capture_output=True,
    )


def balanced_rows(rows, per_class, seed):
    """Equal number of bonafide and spoof files.

    The dataset is ~9:1 spoof:bonafide.  Augmenting a random subset would add
    almost only spoof examples, so the model would still treat codec/resampling
    artefacts as evidence of spoofing.  Balance the augmented classes instead.
    """
    rng = np.random.default_rng(seed)
    chosen = []
    for label in (0, 1):
        pool = [row for row in rows if row["label"] == label]
        count = len(pool) if per_class is None else min(per_class, len(pool))
        indices = rng.choice(len(pool), size=count, replace=False)
        chosen.extend(pool[i] for i in indices)
    rng.shuffle(chosen)
    return chosen


def resample_round_trip(samples, target_rate):
    upsampled = librosa.resample(
        samples, orig_sr=af.SAMPLE_RATE, target_sr=target_rate, res_type="soxr_hq"
    )
    return librosa.resample(
        upsampled, orig_sr=target_rate, target_sr=af.SAMPLE_RATE, res_type="soxr_hq"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=4000,
                        help="number of original train files to augment")
    parser.add_argument("--formats", default="resample22,resample44,mp3,ogg",
                        help="comma-separated augmentations")
    parser.add_argument("--feature-set", default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced", action="store_true",
                        help="take --count bonafide AND --count spoof files")
    args = parser.parse_args()

    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    ffmpeg = find_ffmpeg()
    if any(item in {"mp3", "ogg"} for item in formats) and ffmpeg is None:
        raise SystemExit("ffmpeg is required for mp3/ogg augmentation")

    rows = tm.read_split("train")
    if args.balanced:
        rows = balanced_rows(rows, args.count, args.seed)
    else:
        rows = tm.subsample_rows(rows, args.count, seed=args.seed)
    print(f"augmenting {len(rows):,} training files with {formats}")

    X, y, speakers, paths, attacks, variants = [], [], [], [], [], []
    started = time.perf_counter()
    failed = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for index, row in enumerate(tqdm(rows, desc="augment", disable=True)):
            try:
                waveform, _ = af.load_audio(row["path"], max_seconds=af.CHUNK_SECONDS)
            except Exception:
                failed += 1
                continue

            for variant in formats:
                try:
                    if variant == "resample22":
                        samples = resample_round_trip(waveform, 22050)
                        vector = af.extract_features_from_waveform(
                            samples, af.SAMPLE_RATE, feature_set=args.feature_set
                        )
                    elif variant == "resample44":
                        samples = resample_round_trip(waveform, 44100)
                        vector = af.extract_features_from_waveform(
                            samples, af.SAMPLE_RATE, feature_set=args.feature_set
                        )
                    elif variant in {"mp3", "ogg"}:
                        suffix = "." + variant
                        converted = tmp / f"{index}_{variant}{suffix}"
                        encode(row["path"], converted, ffmpeg)
                        vector = af.extract_features_from_file(
                            converted, max_seconds=af.CHUNK_SECONDS,
                            feature_set=args.feature_set,
                        )
                        converted.unlink(missing_ok=True)
                    else:
                        continue

                    X.append(vector)
                    y.append(row["label"])
                    speakers.append(row["speaker"])
                    paths.append(row["path"])
                    attacks.append(row["attack"])
                    variants.append(variant)
                except Exception:
                    failed += 1

            if (index + 1) % 200 == 0:
                elapsed = time.perf_counter() - started
                print(f"  {index + 1}/{len(rows)} files, {len(X):,} rows, "
                      f"{elapsed:.0f}s", flush=True)

    if not X:
        raise SystemExit("No augmented features were created.")

    X = np.vstack(X).astype(np.float32)
    payload = {
        "X": X,
        "y": np.asarray(y, dtype=np.int8),
        "speaker": np.asarray(speakers),
        "path": np.asarray(paths),
        "attack": np.asarray(attacks),
        "variant": np.asarray(variants),
        "feature_names": np.asarray(af.feature_names(args.feature_set)),
    }
    tag = f"balanced_{len(rows)}" if args.balanced else str(len(rows))
    output = BENCH_DIR / f"train_aug_{tag}_{args.feature_set}.npz"
    np.savez_compressed(output, **payload)
    elapsed = time.perf_counter() - started
    print(f"saved {output}  X={X.shape}  {elapsed:.0f}s  ({failed} failed variants)")


if __name__ == "__main__":
    main()

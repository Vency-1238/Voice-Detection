"""
Stage 1 verifier: checks that the Python environment and the ASVspoof 2019 LA
dataset are ready before any training happens.

Run it from the project folder with the virtual environment activated:

    python setup_check.py
"""

import sys
from importlib.metadata import PackageNotFoundError, version as pkg_version

import config

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

failures = 0
warnings = 0


def report(status: str, message: str) -> None:
    global failures, warnings
    if status == FAIL:
        failures += 1
    elif status == WARN:
        warnings += 1
    print(f"{status} {message}")


# ---------------------------------------------------------------------------
# 1. Python version and interpreter
# ---------------------------------------------------------------------------
print("\n=== 1. Python interpreter ===")
major, minor = sys.version_info[:2]
print(f"    executable : {sys.executable}")
print(f"    version    : {sys.version.split()[0]}")

if (major, minor) >= (3, 11):
    report(PASS, f"Python {major}.{minor} detected (3.11+ supported)")
else:
    report(FAIL, f"Python {major}.{minor} is too old, install Python 3.11 or newer")

in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
report(PASS if in_venv else WARN,
       "running inside the virtual environment (.venv)"
       if in_venv else "NOT inside .venv - activate it before continuing")

# ---------------------------------------------------------------------------
# 2. Required packages
# ---------------------------------------------------------------------------
print("\n=== 2. Required packages ===")

packages = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "librosa": "librosa",
    "soundfile": "soundfile",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "joblib": "joblib",
    "tqdm": "tqdm",
    "flask": "Flask",
    "ipykernel": "ipykernel",
}

for module_name, pip_name in packages.items():
    try:
        __import__(module_name)
        report(PASS, f"{pip_name:<12} {pkg_version(pip_name)}")
    except ImportError:
        report(FAIL, f"{pip_name:<12} NOT installed (pip install -r requirements.txt)")
    except PackageNotFoundError:
        report(WARN, f"{pip_name:<12} imported but has no package metadata")

# ---------------------------------------------------------------------------
# 3. Dataset folders and protocol files
# ---------------------------------------------------------------------------
print("\n=== 3. Dataset files ===")
print(f"    dataset root: {config.DATASET_ROOT}")

if not config.DATASET_ROOT.exists():
    report(FAIL, f"dataset folder not found: {config.DATASET_ROOT}")
else:
    report(PASS, "dataset folder found")

for split_name, split in config.SPLITS.items():
    if split["protocol"].is_file():
        report(PASS, f"{split_name:<5} protocol : {split['protocol'].name}")
    else:
        report(FAIL, f"{split_name:<5} protocol MISSING: {split['protocol']}")

    if split["audio_dir"].is_dir():
        report(PASS, f"{split_name:<5} audio dir: {split['audio_dir']}")
    else:
        report(FAIL, f"{split_name:<5} audio dir MISSING: {split['audio_dir']}")

# ---------------------------------------------------------------------------
# 4. Protocol <-> audio file consistency
# ---------------------------------------------------------------------------
print("\n=== 4. Protocol entries vs audio files ===")

for split_name, split in config.SPLITS.items():
    if not (split["protocol"].is_file() and split["audio_dir"].is_dir()):
        report(FAIL, f"{split_name:<5} skipped (missing files above)")
        continue

    with open(split["protocol"], encoding="utf-8") as fh:
        rows = [line.split() for line in fh if line.strip()]

    present = {p.stem for p in split["audio_dir"].iterdir() if p.is_file()}
    missing = [row[1] for row in rows if row[1] not in present]

    bonafide = sum(1 for row in rows if row[-1] == "bonafide")
    spoof = len(rows) - bonafide

    if missing:
        report(FAIL, f"{split_name:<5} {len(missing)} protocol IDs have no .flac file")
    else:
        report(PASS, f"{split_name:<5} {len(rows):6d} entries "
                     f"({bonafide} bonafide / {spoof} spoof), all audio present")

# ---------------------------------------------------------------------------
# 5. Can we actually decode audio? (one real file per split)
# ---------------------------------------------------------------------------
print("\n=== 5. Audio decoding test ===")

try:
    import soundfile as sf

    for split_name, split in config.SPLITS.items():
        if not split["audio_dir"].is_dir():
            continue
        sample = next(iter(sorted(split["audio_dir"].glob("*.flac"))), None)
        if sample is None:
            report(FAIL, f"{split_name:<5} no .flac files to test")
            continue
        data, sr = sf.read(str(sample))
        duration = len(data) / sr
        report(PASS, f"{split_name:<5} {sample.name} -> {sr} Hz, "
                     f"{duration:.2f} s, {len(data)} samples")
except Exception as exc:  # noqa: BLE001 - we want any decoding error reported
    report(FAIL, f"could not decode audio: {exc}")

# ---------------------------------------------------------------------------
# 6. Project folders
# ---------------------------------------------------------------------------
print("\n=== 6. Project folders ===")
for folder in (config.MODELS_DIR, config.DATA_DIR, config.RESULTS_DIR):
    if folder.is_dir():
        report(PASS, str(folder.relative_to(config.PROJECT_ROOT)))
    else:
        folder.mkdir(parents=True, exist_ok=True)
        report(WARN, f"created missing folder {folder.relative_to(config.PROJECT_ROOT)}")

# ---------------------------------------------------------------------------
# 7. Trained model (optional until you finish the notebook)
# ---------------------------------------------------------------------------
print("\n=== 7. Trained model ===")
if config.MODEL_PATH.exists():
    size_mb = config.MODEL_PATH.stat().st_size / (1024 * 1024)
    try:
        from audio_features import load_model

        bundle = load_model(config.MODEL_PATH)
        report(PASS, f"{config.MODEL_PATH.name} ({size_mb:.1f} MB)")
        report(PASS, f"  model={bundle.get('model_name')} "
                     f"features={bundle.get('feature_set')} "
                     f"({bundle.get('n_features')}) "
                     f"threshold={bundle.get('spoof_threshold', 0.5):.3f}")
    except Exception as exc:  # noqa: BLE001 - report the exact problem
        report(FAIL, f"model exists but cannot be loaded: {exc}")
else:
    report(WARN, "no trained model yet - run: python train_model.py --mode all")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
if failures == 0:
    print(f"RESULT: ENVIRONMENT READY  ({warnings} warning(s))")
    print("Next: train with  python train_model.py --mode all")
    print("      then start the web app with  python app.py")
else:
    print(f"RESULT: {failures} problem(s) found, {warnings} warning(s). Fix them above.")
print("=" * 62 + "\n")

sys.exit(1 if failures else 0)

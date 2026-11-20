"""
Central configuration for the Audio Deepfake Detection project.

All file paths live here so the notebook, the Flask app and the setup checker
never hard-code paths in several places. Change a path once, and everything
(notebook, backend) picks it up.

Folder layout of the project:

    audio_deepfake_vscode/
        Audio_Deepfake_Detector.ipynb   <- training notebook (Stage 2)
        audio_features.py               <- shared feature extraction (Stage 2)
        app.py                          <- Flask backend (Stage 3)
        config.py                       <- THIS FILE
        setup_check.py                  <- environment/dataset verifier (Stage 1)
        requirements.txt
        models/                         <- saved model artifacts
        data/                           <- notes + your own test audio
        results/                        <- evaluation plots and reports
        templates/index.html            <- frontend (Stage 3)
        static/style.css, script.js     <- frontend (Stage 3)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root = the folder that contains this file
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Saved model artifact: created by the notebook, loaded by app.py
MODEL_PATH = MODELS_DIR / "audio_deepfake_model.joblib"

# ---------------------------------------------------------------------------
# Dataset: ASVspoof 2019 Logical Access (LA)
# ---------------------------------------------------------------------------
# The dataset is already present on this machine. If you move it, edit ONE line.
DATASET_ROOT = Path(r"F:\minor project\LA")

CM_PROTOCOL_DIR = DATASET_ROOT / "ASVspoof2019_LA_cm_protocols"

# Each split: where the .flac files are + which protocol file describes them
SPLITS = {
    "train": {
        "audio_dir": DATASET_ROOT / "ASVspoof2019_LA_train" / "flac",
        "protocol": CM_PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt",
    },
    "dev": {
        "audio_dir": DATASET_ROOT / "ASVspoof2019_LA_dev" / "flac",
        "protocol": CM_PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt",
    },
    "eval": {
        "audio_dir": DATASET_ROOT / "ASVspoof2019_LA_eval" / "flac",
        "protocol": CM_PROTOCOL_DIR / "ASVspoof2019.LA.cm.eval.trl.txt",
    },
}


def get_split(name: str) -> dict:
    """Return {'audio_dir': Path, 'protocol': Path} for 'train', 'dev' or 'eval'."""
    if name not in SPLITS:
        raise ValueError(f"Unknown split '{name}'. Choose from {list(SPLITS)}.")
    return SPLITS[name]

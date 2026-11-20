# AI-Based Audio Deepfake Detection System

Detects whether a voice recording is **REAL / BONAFIDE**, **AI-GENERATED / SPOOF**,
or **UNCERTAIN** when the model is not confident enough to make a safe call.

**Dataset:** ASVspoof 2019 Logical Access (LA) · **Stack:** Python · Librosa ·
scikit-learn · Flask

> This is a machine-learning prediction, not proof of authenticity. It is trained
> on ASVspoof 2019 LA and is not a universal deepfake detector. Phone recordings,
> music, heavy noise, and audio unlike the training data can be misclassified.
> `UNCERTAIN` means “needs review”, not a verdict.

---

## Quick start: just run the app

```powershell
cd audio_deepfake_vscode

# Optional: VS Code already uses .venv; activate it in a terminal if needed
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python setup_check.py     # checks Python, libraries, dataset paths, model
python app.py             # loads the shipped model once, then serves the UI
```

Open **http://127.0.0.1:5000**.

* The model is loaded **once at startup**; opening the page never retrains anything.
* Upload up to 20 files (100 MB total) in WAV, FLAC, MP3 or OGG, **or record your
  voice directly in the browser**.
* A bad or corrupted file gets its own error message; the rest of the batch still runs.
* `GET /health` shows which model is loaded, its thresholds, and its held-out metrics.

### VeriHuman website

The root URL serves the VeriHuman website. Create an account to use the protected
dashboard, upload workspace, private SQLite-backed analysis history, and PDF/CSV
reports. Website records store the values returned by the existing prediction
interface; uploaded audio is removed after analysis unless private retention is
explicitly enabled in Settings.

The website can be started with the same command:

```powershell
python app.py
```

The model files and feature pipeline are not modified by the website layer.

If `setup_check.py` says the model is missing, train it first (below).

---

## How to train the model

Training is done by ordinary Python scripts, not by a notebook. The notebook
(`Audio_Deepfake_Detector.ipynb`) is a guided walkthrough that calls these scripts.
Every script uses the same `audio_features.py` as the Flask app, so the feature
vector at training time is exactly the one used at prediction time.

### Recommended full pipeline

```powershell
# 0. Environment, dataset paths and dataset file counts
python setup_check.py

# 1. Build feature caches, compare feature sets / classifiers on speaker-disjoint
#    dev data, calibrate, tune the threshold for a 2% false-positive target,
#    evaluate on unseen attacks, and save models/audio_deepfake_model.joblib
python train_model.py --mode all

# 2. Add codec/resampling robustness (see the explanation below)
python augment_training.py --count 2580 --balanced --formats resample44,mp3

# 3. Retrain the production bundle on native + augmented features and save it
python train_augmented.py

# 4. Re-measure everything used in this README
python final_evaluation.py
```

Then start the app:

```powershell
python app.py
```

**Typical timings on this machine**

| Step | First run | Re-run |
|---|---|---|
| `setup_check.py` | a few seconds | a few seconds |
| `train_model.py --mode all` | ~30–60 min (decodes ~52,000 files) | a few minutes (caches reused) |
| `augment_training.py ...` | ~10 min for 5,160 files × 2 variants | same |
| `train_augmented.py` | ~1–2 min | same |
| `final_evaluation.py` | ~3–5 min | same |

Feature caches are written to `benchmarks/*.npz`; delete a cache file to rebuild it.

### Faster experiment loop

```powershell
# compare feature sets and models on a 4,000-file training subset
python train_model.py --mode compare --max-train-files 4000 --skip-svc

# train only the final bundle from the caches already on disk
python train_model.py --mode final
```

### Why the augmentation step exists

A model trained only on native 16 kHz FLAC files learns that a sharp
high-frequency roll-off means “spoof”. A genuine voice saved as a 44.1 kHz MP3 has
a very similar roll-off, so the unaugmented model marked **every** resampled or
MP3-encoded genuine voice as AI-generated. That is the worst possible error for
this project, so it was fixed properly:

`augment_training.py` converts a **class-balanced** subset of the *training* split
to 44.1 kHz and to real MP3, then extracts the same features. Dev and eval files
are never augmented. `train_augmented.py` combines native and augmented rows,
retrains, recalibrates, retunes the threshold, and saves the bundle.

---

## Project layout

```
audio_deepfake_vscode/
├── audio_features.py            # feature extraction, chunking, prediction, calibration
├── train_model.py               # caching, model comparison, calibration, final training
├── augment_training.py          # balanced 44.1 kHz / MP3 augmentation cache
├── train_augmented.py           # production training on native + augmented features
├── evaluate_models.py           # feature-set and classifier experiments
├── select_final_model.py        # final candidate comparison + latency
├── final_evaluation.py          # every measured number in this README
├── benchmark.py                 # diagnostics: cache, feature timings, model latency
├── setup_check.py               # environment / dataset / model verifier
├── app.py                       # Flask backend (batch prediction, health endpoint)
├── config.py                    # all paths and dataset locations
├── models/audio_deepfake_model.joblib
├── benchmarks/                  # feature caches and JSON measurements
├── results/                     # evaluation reports and logs
├── templates/index.html         # UI
├── static/style.css, script.js  # UI
├── Audio_Deepfake_Detector.ipynb
└── legacy/                      # original v1 extractor, model and notebook (reference)
```

The dataset is **not** copied into the project. It lives at the path in
`config.py` (`DATASET_ROOT`); edit that single line if you move it.

---

## How detection works

1. **Decode** the upload to 16 kHz mono with `soundfile`, falling back to `librosa`
   (and therefore `audioread`/ffmpeg) for formats soundfile cannot read.
2. **Chunk** long files into 5-second windows. A tail shorter than 1 second is
   merged into the previous chunk; analysis is capped at 10 minutes.
3. **Extract 172 features per chunk** from a *single shared STFT*:
   mean/std of MFCCs, delta and delta-delta MFCCs, spectral centroid, bandwidth,
   roll-off, flatness, flux, spectral contrast, chroma, zero-crossing rate and RMS.
4. **Classify** all chunks from all uploaded files in one `predict_proba` batch
   using **ExtraTrees150** (`n_jobs=1`, chosen because it had the best unseen-attack
   F1 and ~21 ms one-row latency).
5. **Calibrate** the raw score with Platt scaling, fitted on dev-tune speakers only.
6. **Aggregate** chunk scores with duration × energy weights. Strong disagreement
   between chunks is reported and can force `UNCERTAIN`.
7. **Decide** with two thresholds:
   * score ≥ spoof threshold → `SPOOF`
   * score ≤ bonafide threshold → `BONAFIDE`
   * otherwise → `UNCERTAIN`

The score shown in the UI is the **calibrated** score. The raw model score is also
returned, so it is never presented as if it were a true probability when it is not.

---

## Recording your voice in the browser

The page has a built-in recorder, so you can test a live voice without creating a
file first.

1. Press **Start recording** and allow microphone access.
2. Speak for 0.5–60 seconds. A level meter and timer show that audio is arriving.
3. Press **Stop recording**, listen back to the preview, then press
   **Analyse this recording**.

How it is implemented:

* Capture uses `navigator.mediaDevices.getUserMedia` with `echoCancellation`,
  `noiseSuppression` and `autoGainControl` turned **off**, so the browser does not
  change the spectrum before the model sees it.
* The `MediaRecorder` blob (WebM/Opus in Chrome, Edge and Firefox; MP4/AAC in
  Safari) is decoded with Web Audio, downmixed to mono and resampled to
  **16 kHz**, then encoded as 16-bit PCM WAV **in the browser**. No ffmpeg is
  needed for the normal recording path.
* The resulting `recording-YYYYMMDD-HHMMSS.wav` is submitted through exactly the
  same `POST /predict` endpoint as an uploaded file, so it gets the same chunking,
  calibration, thresholds and `UNCERTAIN` logic.
* Uploaded files are copied to a temporary file and deleted immediately after
  analysis; recordings are not stored on the server.

Because it is the same pipeline, a normal live recording of your own voice should
come back as **REAL**. Note that playing an AI clip through loudspeakers and
re-recording it with the microphone changes the signal a lot (room acoustics,
speaker response, microphone response), so that is **not** a valid spoof test.
Use an actual spoof file for that.

Recording containers can also be uploaded directly (`.webm`, `.m4a`, `.oga`) as a
fallback. Those need `ffmpeg` on the server to decode; the in-browser recorder
does not.

---

## Evaluation protocol

The numbers below are honest about generalisation:

| Data | Role |
|---|---|
| train, attacks A01–A06 | model fitting only |
| dev, **tune speakers** | calibration + threshold tuning only |
| dev, **test speakers** (disjoint) | speaker-independent report |
| eval, 2,000-file validation sample (attacks A07–A19) | model-selection validation |
| eval, separate 2,000-file holdout (attacks A07–A19, no file overlap) | final report only |

* Dev accuracy is high because train and dev share attack types. That is **not**
  the headline result.
* The headline result is the **eval holdout**, which contains attacks the model
  has never seen.
* The spoof threshold is tuned for a **2% false-positive target** on dev-tune
  speakers, because wrongly accusing a real person is the costlier error here.

### Headline result: unseen attacks (eval holdout)

| Metric | Value |
|---|---|
| Accuracy | 0.9050 |
| Precision | 0.9787 |
| Recall (spoof detection) | 0.8280 |
| F1 | 0.8971 |
| ROC-AUC | 0.9556 |
| False-positive rate | 0.0180 |
| False-negative rate | 0.1720 |

Measured with `python final_evaluation.py`; raw numbers are in
`results/final_evaluation.json`.

### Old vs optimized pipeline

Both are measured at the same 2% false-positive target on dev-tune speakers.
“Old” is the original recipe (Random Forest 300 + 90 features + 4,000 training
files). “New” is the shipped model (ExtraTrees150 + 172 features + 35,700 rows,
including balanced 44.1 kHz/MP3 augmentation).

| Split | Old F1 | New F1 | Old FPR | New FPR | Old FNR | New FNR | New AUC |
|---|---|---|---|---|---|---|---|
| dev tune (calibration) | 0.8649 | 0.9969 | 0.0196 | 0.0196 | 0.2363 | 0.0039 | 0.9994 |
| dev test (new speakers) | 0.8721 | **0.9977** | 0.0157 | 0.0126 | 0.2254 | 0.0031 | 0.9998 |
| eval validation (unseen attacks) | 0.8487 | **0.9059** | 0.0160 | 0.0120 | 0.2510 | 0.1620 | 0.9622 |
| eval holdout (unseen attacks, separate files) | 0.8275 | **0.8971** | 0.0130 | 0.0180 | 0.2850 | 0.1720 | 0.9556 |

The old baseline could not be unpickled under the current scikit-learn version
(its `SimpleImputer` pickles are incompatible), so it was **reconstructed from its
exact recipe** and retrained on the same machine rather than loaded from the old
file. That makes the comparison like-for-like.

### Per-attack recall on the unseen-attack holdout

| Attack | Recall | Attack | Recall |
|---|---|---|---|
| A07 | 1.0000 | A14 | 1.0000 |
| A08 | 1.0000 | A15 | 0.9867 |
| A09 | 1.0000 | A16 | 1.0000 |
| A10 | 0.9836 | **A17** | **0.0494** |
| A11 | 1.0000 | **A18** | **0.1856** |
| A12 | 0.9254 | A19 | 0.8902 |
| A13 | 1.0000 | genuine voices | FPR 0.0180 |

A17 and A18 remain the hard cases. This is reported rather than hidden: a single
overall accuracy number would hide it.

### Three-way decisions and calibration (holdout)

| True class | Decided BONAFIDE | UNCERTAIN | Decided SPOOF |
|---|---|---|---|
| genuine (n=1,000) | 97.2% | 1.0% | 1.8% |
| spoof (n=1,000) | 16.2% | 1.0% | 82.8% |

Platt calibration on the holdout: Brier score 0.0818, expected calibration error
0.0705, log loss 0.4575. The displayed score is calibrated; the raw score is
returned separately for transparency.

### Speed (40 real holdout files, warm process)

| Measurement | Old | New | Speed-up |
|---|---|---|---|
| End-to-end, per file | 106.28 ms | **24.71 ms** | 4.3× |
| Feature extraction only | 37.50 ms | **20.03 ms** | 1.9× |
| Model, one row | 48.03 ms | **12.87 ms** | 3.7× |
| Batch prediction | — | **13.63 ms/file** | — |

“End-to-end” includes decoding, chunking, feature extraction, calibration and the
decision. Batch prediction sends every chunk from every uploaded file through
`predict_proba` in one call.

### 5-second vs 10-second chunks (24 long recordings, mean 9.41 s)

| Window | Accuracy | UNCERTAIN | Time per file | Chunks per file |
|---|---|---|---|---|
| 5 s | 0.9583 | 4.17% | 64.6 ms | 2.0 |
| 10 s | 0.9583 | 0.00% | 44.4 ms | 1.0 |

Accuracy is identical; 10-second chunks are about 31% faster on long files but
give coarser detail. The shipped default stays at **5 seconds** so the UI can show
a meaningful chunk breakdown.

### Format robustness (20 files: 10 genuine + 10 spoof, encoded with ffmpeg)

| Variant | Decoded | Label agreement | Mean absolute score change |
|---|---|---|---|
| WAV 16 kHz (control) | 20/20 | 100.0% | 0.00% |
| WAV 44.1 kHz | 20/20 | 90.0% | 6.93% |
| FLAC 44.1 kHz | 20/20 | 90.0% | 6.93% |
| MP3 44.1 kHz, 192 kbps | 20/20 | 90.0% | 7.00% |
| OGG 44.1 kHz, q6 | 20/20 | 85.0% | 9.43% |

Every disagreement was a **spoof** file that became BONAFIDE or UNCERTAIN after
conversion. **No genuine voice became SPOOF** in any format. This is the critical
fix: before balanced augmentation, every genuine 22.05/44.1/48 kHz or MP3 upload
was scored as 100% SPOOF. On a separate 30-genuine/30-spoof check, the single
genuine false positive was the same file before and after conversion, so the
remaining error is a model limitation, not a codec artefact.

### Corrupted and edge-case files

| File | Result |
|---|---|
| empty file | rejected: “could not decode … corrupted, empty, or not a real audio file” |
| random bytes named `.wav` | rejected with the same readable message |
| 0.1 s valid WAV | rejected: “at least 0.25 seconds are required” |
| 1 s stereo WAV | downmixed to mono and analysed |
| unsupported extension (`.txt`) | rejected per file; the rest of the batch continues |

### Environment used for these numbers

Python 3.11.8 · numpy 2.4.3 · scipy 1.17.1 · librosa 0.11.0 ·
scikit-learn 1.8.0 · soundfile 0.14.0 · Flask 3.1.3 · Windows · ffmpeg 8.0
for the format conversions.

---

## Using the model from Python

```python
from audio_features import load_model, predict_audio_files

bundle = load_model("models/audio_deepfake_model.joblib")
results = predict_audio_files(["clip1.wav", "clip2.mp3"], bundle)

for r in results:
    if r.get("error"):
        print(r["filename"], "->", r["error"])
    else:
        print(r["filename"], "->", r["prediction"],
              r["spoof_score"], "%", r["confidence_label"])
```

Each result contains `prediction`, `display_label`, `spoof_score`,
`raw_spoof_score`, `confidence`, `confidence_label`, `calibrated`,
`chunks_analysed`, `chunk_seconds`, `chunk_results`, `chunk_disagreement`,
`duration_seconds`, `truncated`, `threshold`, `bonafide_threshold`, `reason`,
`model_name` and `feature_set`.

---

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | web UI |
| GET | `/health` | model name, feature set, thresholds, held-out metrics |
| POST | `/predict` | multipart form field `files` (one or more audio files) |

`POST /predict` returns:

```json
{
  "results": [ { "filename": "...", "prediction": "BONAFIDE", "spoof_score": 0.11, "...": "..." } ],
  "summary": { "bonafide": 1, "spoof": 1, "uncertain": 0, "errors": 0 },
  "count": 2,
  "elapsed_seconds": 0.32,
  "model": { "model_name": "extra150", "feature_set": "full", "n_features": 172 }
}
```

Errors are per file: one unreadable upload does not fail the request. Unsupported
extensions are rejected with a readable message; empty or corrupted audio is
decoded safely and reported as an error.

---

## Limitations

* **Not universal.** The model knows ASVspoof 2019 LA and related synthesis
  artefacts. It has not been validated on arbitrary internet audio, music,
  singing, or non-speech.
* **Hard attacks remain.** Per-attack recall in `results/final_evaluation.json`
  shows that A17 and A18 are still much harder than the other eval attacks.
* **Calibration is dataset-specific.** Platt scaling makes scores more meaningful,
  but they are still estimates. Treat `UNCERTAIN` as “needs human review”.
* **Telephone / heavy compression.** Augmentation covers 44.1 kHz resampling and
  MP3. Very low-bitrate or telephony-band audio is outside the tested range.
* **Live microphone audio.** Recording in the browser is normalised to 16 kHz mono,
  which matches the model's training format, but room noise and microphone
  colouration are still unlike ASVspoof. A spoof played through loudspeakers and
  re-recorded with a microphone is not a valid test of the detector.
* **Adversarial spoofs.** A deliberately crafted attack that mimics bonafide
  statistics can still evade the classifier.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `setup_check.py` cannot find the dataset | edit `DATASET_ROOT` in `config.py` |
| “no trained model yet” | run the training pipeline above |
| MP3/OGG augmentation fails | install ffmpeg and put it on `PATH` (or in `C:\ffmpeg-8.0-essentials_build\bin`) |
| First prediction is slow | the app warms the pipeline at startup; a direct Python script pays a one-time import cost |
| Model will not load after changing libraries | retrain with the versions in `requirements.txt`; joblib bundles are tied to the scikit-learn version that wrote them |
| Very slow feature extraction on Windows | the code avoids OpenBLAS matmuls for tiny matrices; do not replace `np.einsum` with `@` in the mel/spectral paths |

The bundled `.venv` is set up with exactly these versions and is the interpreter
VS Code uses. It was verified end-to-end: it loads the shipped bundle and predicts
a real dev file correctly. If you ever rebuild the environment, install from
`requirements.txt` so the joblib bundle stays compatible.
#   V o i c e - D e t e c t i o n  
 
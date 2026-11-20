"""Upload genuine recordings converted to real-world formats and check that
the Flask app does not call them AI-generated.

Run AFTER restarting app.py with the augmented model:
    python app.py
    python <this script>
"""

import json
import subprocess
import tempfile
import urllib.request
import uuid
from pathlib import Path

import numpy as np

import config
import final_evaluation as fe

BASE = "http://127.0.0.1:5000"
HOLDOUT = Path(config.PROJECT_ROOT) / "benchmarks" / "eval_holdout_2000_full.npz"


def multipart(files):
    boundary = "----opencode" + uuid.uuid4().hex
    body = []
    for name, path in files:
        body.append(f"--{boundary}\r\n".encode())
        body.append(
            f'Content-Disposition: form-data; name="files"; filename="{name}"\r\n'.encode()
        )
        body.append(b"Content-Type: application/octet-stream\r\n\r\n")
        body.append(Path(path).read_bytes())
        body.append(b"\r\n")
    body.append(f"--{boundary}--\r\n".encode())
    return b"".join(body), f"multipart/form-data; boundary={boundary}"


def main():
    with np.load(HOLDOUT, allow_pickle=False) as data:
        paths = [str(p) for p in data["path"]]
        labels = data["y"].astype(int)

    rng = np.random.default_rng(7)
    picks = []
    for label in (0, 1):
        pool = [p for p, y in zip(paths, labels) if y == label]
        chosen = [pool[i] for i in rng.choice(len(pool), size=2, replace=False)]
        picks.extend((p, label) for p in chosen)

    ffmpeg = fe.find_ffmpeg()
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found")

    recipes = {
        "wav44": (["-ar", "44100", "-c:a", "pcm_s16le"], "wav"),
        "flac44": (["-ar", "44100", "-c:a", "flac"], "flac"),
        "mp3": (["-ar", "44100", "-c:a", "libmp3lame", "-b:a", "192k"], "mp3"),
        "ogg": (["-ar", "44100", "-c:a", "libvorbis", "-q:a", "6"], "ogg"),
    }

    uploads = []
    truth = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for index, (source, label) in enumerate(picks):
            original = Path(source)
            uploads.append((original.name, original))
            truth[original.name] = label
            for tag, (codec, extension) in recipes.items():
                target = tmp / f"{index}_{tag}.{extension}"
                subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error", "-i", source, "-ac", "1",
                     *codec, str(target)],
                    check=True, capture_output=True,
                )
                name = f"{original.stem}_{tag}.{extension}"
                uploads.append((name, target))
                truth[name] = label

        payload, ctype = multipart(uploads)
        request = urllib.request.Request(
            BASE + "/predict", data=payload,
            headers={"Content-Type": ctype}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.load(response)

    print("summary:", data["summary"], "elapsed:", data["elapsed_seconds"], "s")

    by_name = {}
    for result in data["results"]:
        by_name[result["filename"]] = result

    critical = 0
    misses = 0
    for result in data["results"]:
        name = result["filename"]
        expected = "BONAFIDE" if truth[name] == 0 else "SPOOF"
        if result.get("error"):
            print(f"  ERROR  {name:<34} {result['error'][:60]}")
            critical += 1
            continue

        prediction = result["prediction"]
        if truth[name] == 0:
            ok = prediction == "BONAFIDE"
            if not ok:
                critical += 1
            tag = "OK " if ok else "CRITICAL FALSE POSITIVE"
        else:
            ok = prediction == "SPOOF"
            if not ok:
                misses += 1
            tag = "OK " if ok else "spoof miss"
        print(f"  {tag:<24} {name:<34} truth={expected:<9} "
              f"pred={prediction:<9} score={result['spoof_score']:>6}%")

    # A conversion must not change a genuine decision.
    changed_genuine = 0
    for name, label in truth.items():
        if label != 0 or "_" not in name:
            continue
        stem = name.rsplit("_", 1)[0]
        originals = by_name.get(f"{stem}.flac") or by_name.get(f"{stem}.wav")
        converted = by_name.get(name)
        if not originals or not converted or converted.get("error"):
            continue
        if originals["prediction"] == "BONAFIDE" and converted["prediction"] != "BONAFIDE":
            changed_genuine += 1
            print(f"  FORMAT FLIP            {name:<34} "
                  f"{originals['prediction']} -> {converted['prediction']}")

    print(f"\ncritical errors (genuine flagged / unreadable): {critical}")
    print(f"spoof misses (native or converted): {misses}")
    print(f"genuine decisions changed by format conversion: {changed_genuine}")
    raise SystemExit(1 if critical or changed_genuine else 0)


if __name__ == "__main__":
    main()

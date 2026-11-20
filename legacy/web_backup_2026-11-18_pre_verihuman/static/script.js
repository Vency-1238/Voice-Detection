/* ------------------------------------------------------------------
   Audio Deepfake Detector - front-end logic
   Talks to Flask with the Fetch API. No Python runs in the browser.
   ------------------------------------------------------------------ */

const form = document.getElementById("upload-form");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const fileList = document.getElementById("file-list");
const analyseBtn = document.getElementById("analyse-btn");
const resetBtn = document.getElementById("reset-btn");
const statusBox = document.getElementById("status");
const statusText = document.getElementById("status-text");
const alertBox = document.getElementById("alert");
const resultsSection = document.getElementById("results");
const resultsList = document.getElementById("results-list");
const summaryBox = document.getElementById("summary");
const elapsedBox = document.getElementById("elapsed");
const modelPill = document.getElementById("model-pill");

const recordBtn = document.getElementById("record-btn");
const recordBtnLabel = document.getElementById("record-btn-label");
const recordTimer = document.getElementById("record-timer");
const levelFill = document.getElementById("level-fill");
const recorderStatus = document.getElementById("recorder-status");
const recordingPreview = document.getElementById("recording-preview");
const recordingAudio = document.getElementById("recording-audio");
const analyseRecordingBtn = document.getElementById("analyse-recording-btn");
const discardRecordingBtn = document.getElementById("discard-recording-btn");

const ALLOWED = [".wav", ".flac", ".mp3", ".ogg", ".webm", ".m4a", ".oga"];
const MAX_FILES = 20;
const MAX_BYTES = 100 * 1024 * 1024;

/* Recorder settings. Recordings are converted to mono PCM WAV at the model's
   native sample rate in the browser, so no server-side ffmpeg is required. */
const RECORD_TARGET_RATE = 16000;
const RECORD_MAX_SECONDS = 60;
const RECORD_MIN_SECONDS = 0.5;
const RECORDING_MIME_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
  "audio/mp4",
];

/** Files currently selected, one entry per file object. */
let selectedFiles = [];

/* ----------------------------- helpers ----------------------------- */

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function hasAllowedExtension(name) {
  const lower = name.toLowerCase();
  return ALLOWED.some((extension) => lower.endsWith(extension));
}

function fileKey(file) {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

function showAlert(message, type = "error") {
  alertBox.textContent = message;
  alertBox.className = type === "warning" ? "alert warning" : "alert";
  alertBox.hidden = false;
}

function hideAlert() {
  alertBox.hidden = true;
  alertBox.textContent = "";
}

function decisionClass(result) {
  if (result.error) return "failed";
  const prediction = String(result.prediction || "").toLowerCase();
  if (prediction === "spoof") return "spoof";
  if (prediction === "uncertain") return "uncertain";
  return "bonafide";
}

function badgeText(result) {
  if (result.error) return "Could not analyse";
  if (result.prediction === "SPOOF") return "AI-generated";
  if (result.prediction === "UNCERTAIN") return "Uncertain";
  return "Real";
}

/* ---------------------------- file list ---------------------------- */

function renderFileList() {
  fileList.innerHTML = "";

  selectedFiles.forEach((file, index) => {
    const item = document.createElement("li");
    item.className = "file-item";

    const name = document.createElement("span");
    name.className = "file-name";
    name.textContent = file.name;

    const size = document.createElement("span");
    size.className = "file-size";
    size.textContent = formatSize(file.size);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "file-remove";
    remove.setAttribute("aria-label", `Remove ${file.name}`);
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      selectedFiles.splice(index, 1);
      renderFileList();
    });

    item.append(name, size, remove);
    fileList.appendChild(item);
  });

  fileList.hidden = selectedFiles.length === 0;
  analyseBtn.disabled = selectedFiles.length === 0;
}

function addFiles(fileListLike) {
  const incoming = Array.from(fileListLike);
  const rejected = [];
  const keys = new Set(selectedFiles.map(fileKey));
  const messages = [];

  incoming.forEach((file) => {
    if (!hasAllowedExtension(file.name)) {
      rejected.push(file.name);
      return;
    }
    if (selectedFiles.length >= MAX_FILES) {
      messages.push(`Only the first ${MAX_FILES} files are analysed at once.`);
      return;
    }
    const duplicate = keys.has(fileKey(file));
    if (duplicate) return;          // ignore duplicates
    keys.add(fileKey(file));
    selectedFiles.push(file);
  });

  const totalBytes = selectedFiles.reduce((sum, file) => sum + file.size, 0);
  if (totalBytes > MAX_BYTES) {
    messages.push("Selected files exceed the 100 MB total upload limit.");
  }

  renderFileList();

  if (rejected.length) {
    messages.unshift(
      `Skipped unsupported file(s): ${rejected.join(", ")}. Use WAV, FLAC, MP3 or OGG.`
    );
  }

  if (messages.length) {
    showAlert(messages.join(" "), "warning");
  } else {
    hideAlert();
  }
}

/* ------------------------- picking files -------------------------- */

fileInput.addEventListener("change", () => {
  addFiles(fileInput.files);
  fileInput.value = "";                 // allows picking the same file again
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
});

["dragleave", "dragend"].forEach((eventName) => {
  dropzone.addEventListener(eventName, () => dropzone.classList.remove("dragging"));
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
  if (event.dataTransfer && event.dataTransfer.files.length) {
    addFiles(event.dataTransfer.files);
  }
});

/* Drop anywhere on the page, not only on the dropzone. */
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => {
  event.preventDefault();
  if (event.dataTransfer && event.dataTransfer.files.length) {
    addFiles(event.dataTransfer.files);
  }
});

/* ---------------------------- analysing ---------------------------- */

function setLoading(isLoading, message = "Analysing…") {
  statusText.textContent = message;
  statusBox.hidden = !isLoading;
  analyseBtn.disabled = isLoading || selectedFiles.length === 0;
  analyseBtn.textContent = isLoading ? "Analysing…" : "Analyse audio";
}

function addChunkBreakdown(card, result) {
  const chunks = result.chunk_results || [];
  if (!chunks.length) return;

  const details = document.createElement("details");
  details.className = "chunk-details";

  const summary = document.createElement("summary");
  summary.textContent =
    `Chunk breakdown (${result.chunks_analysed} × ${result.chunk_seconds || 5}s window)`;
  details.appendChild(summary);

  const table = document.createElement("table");
  table.className = "chunk-table";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["Start", "Length", "Spoof score", "Weight"].forEach((label) => {
    const th = document.createElement("th");
    th.textContent = label;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);

  const tbody = document.createElement("tbody");
  chunks.forEach((chunk) => {
    const row = document.createElement("tr");
    const cells = [
      `${chunk.start_seconds}s`,
      `${chunk.duration_seconds}s`,
      `${chunk.spoof_score}%`,
      chunk.weight,
    ];
    cells.forEach((text) => {
      const td = document.createElement("td");
      td.textContent = text;
      row.appendChild(td);
    });
    tbody.appendChild(row);
  });

  table.append(thead, tbody);
  details.appendChild(table);
  card.appendChild(details);
}

function buildResultCard(result) {
  const card = document.createElement("article");
  const failed = Boolean(result.error);
  const kind = decisionClass(result);

  card.className = `result ${kind}`;

  const head = document.createElement("div");
  head.className = "result-head";

  const name = document.createElement("span");
  name.className = "result-name";
  name.textContent = result.filename || "Unknown file";

  const badge = document.createElement("span");
  badge.className = `badge ${kind}`;
  badge.textContent = badgeText(result);

  head.append(name, badge);
  card.appendChild(head);

  if (failed) {
    const error = document.createElement("p");
    error.className = "result-error";
    error.textContent = result.error;
    card.appendChild(error);
    return card;
  }

  const scoreRow = document.createElement("div");
  scoreRow.className = "score-row";

  const bar = document.createElement("div");
  bar.className = "score-bar";

  const fill = document.createElement("div");
  fill.className = `score-fill ${kind}`;
  bar.appendChild(fill);

  const value = document.createElement("span");
  value.className = "score-value";
  value.textContent = `${result.spoof_score}%`;

  scoreRow.append(bar, value);
  card.appendChild(scoreRow);

  // Animate the bar after the card is in the DOM.
  requestAnimationFrame(() => {
    fill.style.width = `${Math.max(2, Math.min(100, result.spoof_score))}%`;
  });

  const meta = document.createElement("p");
  meta.className = "result-meta";
  meta.textContent =
    `${result.display_label} · spoof score ${result.spoof_score}% · ` +
    `${result.confidence_label} confidence · ${result.chunks_analysed} chunk(s) · ` +
    `${result.duration_seconds}s of audio`;
  card.appendChild(meta);

  const detail = document.createElement("p");
  detail.className = "result-detail";
  const threshold = Math.round((result.threshold ?? 0.5) * 100);
  const raw = result.calibrated
    ? `raw model score ${result.raw_spoof_score}% → calibrated ${result.spoof_score}%`
    : `model score ${result.spoof_score}% (not calibrated)`;
  detail.textContent =
    `Spoof threshold ${threshold}% · ${raw}` +
    (result.chunk_disagreement
      ? ` · chunk disagreement ${Math.round(result.chunk_disagreement * 100)}%`
      : "");
  card.appendChild(detail);

  if (result.reason) {
    const reason = document.createElement("p");
    reason.className = "result-reason";
    reason.textContent = result.reason;
    card.appendChild(reason);
  }

  if (result.truncated) {
    const note = document.createElement("p");
    note.className = "result-note";
    note.textContent = "Only the first 10 minutes of this file were analysed.";
    card.appendChild(note);
  }

  addChunkBreakdown(card, result);
  return card;
}

function renderSummary(summary, count, elapsedSeconds) {
  const entries = [
    ["Real", summary.bonafide || 0, "bonafide"],
    ["AI-generated", summary.spoof || 0, "spoof"],
    ["Uncertain", summary.uncertain || 0, "uncertain"],
    ["Errors", summary.errors || 0, "failed"],
  ];

  summaryBox.innerHTML = "";
  entries.forEach(([label, value, kind]) => {
    const chip = document.createElement("span");
    chip.className = `summary-chip ${kind}`;
    chip.textContent = `${value} ${label}`;
    summaryBox.appendChild(chip);
  });
  summaryBox.hidden = false;

  elapsedBox.textContent =
    `${count} file${count === 1 ? "" : "s"} · ${elapsedSeconds.toFixed(2)} s`;
}

function renderResults(payload) {
  const results = payload.results || [];
  resultsList.innerHTML = "";
  results.forEach((result) => resultsList.appendChild(buildResultCard(result)));

  if (payload.summary) {
    renderSummary(payload.summary, payload.count ?? results.length,
                  Number(payload.elapsed_seconds || 0));
  }

  resultsSection.hidden = results.length === 0;
  resetBtn.hidden = results.length === 0;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!selectedFiles.length) {
    showAlert("Please choose at least one audio file.", "warning");
    return;
  }

  hideAlert();
  resultsSection.hidden = true;
  summaryBox.hidden = true;
  elapsedBox.textContent = "";
  setLoading(true, `Analysing ${selectedFiles.length} file(s)…`);

  const payload = new FormData();
  selectedFiles.forEach((file) => payload.append("files", file, file.name));

  try {
    const response = await fetch("/predict", { method: "POST", body: payload });
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(data.error || `Server returned ${response.status}.`);
    }

    renderResults(data);
  } catch (error) {
    showAlert(error.message || "Could not reach the server. Is app.py running?", "error");
  } finally {
    setLoading(false);
  }
});

/* ------------------------------ reset ------------------------------ */

resetBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    stopRecording();
  }
  discardRecording();
  selectedFiles = [];
  renderFileList();
  resultsList.innerHTML = "";
  resultsSection.hidden = true;
  summaryBox.hidden = true;
  elapsedBox.textContent = "";
  resetBtn.hidden = true;
  hideAlert();
  fileInput.value = "";
  dropzone.focus();
});

/* ---------------------------- recorder ---------------------------- */

let mediaRecorder = null;
let mediaStream = null;
let recordedChunks = [];
let recordingStartedAt = 0;
let recordingSeconds = 0;
let recordingTimer = null;
let meterFrame = null;
let meterContext = null;
let pendingRecordingBlob = null;
let pendingRecordingUrl = null;

function formatDuration(seconds) {
  const whole = Math.max(0, Math.floor(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

function setRecorderStatus(message, isError = false) {
  recorderStatus.textContent = message;
  recorderStatus.className = isError ? "recorder-status error" : "recorder-status";
  recorderStatus.hidden = !message;
}

function clearRecorderStatus() {
  setRecorderStatus("");
}

function pickRecordingMimeType() {
  if (typeof window.MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) {
    return "";
  }
  return RECORDING_MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

/** Encode mono float samples as a 16-bit PCM WAV Blob. */
function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeText = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  };

  writeText(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);            // fmt chunk size
  view.setUint16(20, 1, true);             // PCM
  view.setUint16(22, 1, true);             // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);             // block align
  view.setUint16(34, 16, true);            // bits per sample
  writeText(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }
  return new Blob([view], { type: "audio/wav" });
}

async function renderMono(decoded, targetRate) {
  const OfflineContextClass = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  if (!OfflineContextClass) {
    throw new Error("this browser cannot resample audio");
  }
  const duration = decoded.duration || decoded.length / decoded.sampleRate;
  const frames = Math.max(1, Math.ceil(duration * targetRate));

  const offline = new OfflineContextClass(1, frames, targetRate);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();

  const rendered = await offline.startRendering();
  return { samples: rendered.getChannelData(0), sampleRate: targetRate };
}

/** Decode the recorder's WebM/Ogg/MP4 blob and render it as 16 kHz mono WAV. */
async function recordingToWav(blob) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error("this browser cannot decode audio");
  }

  const encoded = await blob.arrayBuffer();
  const decodeContext = new AudioContextClass();
  let decoded;
  try {
    decoded = await decodeContext.decodeAudioData(encoded);
  } finally {
    decodeContext.close().catch(() => {});
  }
  if (!decoded || !decoded.length) throw new Error("the recording was empty");

  let mono;
  try {
    mono = await renderMono(decoded, RECORD_TARGET_RATE);
  } catch {
    // Some browsers refuse unusual custom rates. Keep the capture rate instead;
    // the server resamples to 16 kHz in the normal pipeline.
    mono = await renderMono(decoded, decoded.sampleRate);
  }
  if (!mono.samples.length) throw new Error("the recording was empty");
  return encodeWav(mono.samples, mono.sampleRate);
}

function startTimer() {
  stopTimer();
  recordingSeconds = 0;
  recordTimer.textContent = "0:00";
  recordingTimer = window.setInterval(() => {
    recordingSeconds = (Date.now() - recordingStartedAt) / 1000;
    recordTimer.textContent = formatDuration(recordingSeconds);
    if (recordingSeconds >= RECORD_MAX_SECONDS) {
      setRecorderStatus(`Reached the ${RECORD_MAX_SECONDS}-second limit.`);
      stopRecording();
    }
  }, 200);
}

function stopTimer() {
  if (recordingTimer) {
    window.clearInterval(recordingTimer);
    recordingTimer = null;
  }
}

function startMeter(stream) {
  stopMeter();
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return;
  try {
    meterContext = new AudioContextClass();
    const source = meterContext.createMediaStreamSource(stream);
    const analyser = meterContext.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);
    const data = new Uint8Array(analyser.fftSize);

    const tick = () => {
      analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i += 1) {
        const value = (data[i] - 128) / 128;
        sum += value * value;
      }
      const rms = Math.sqrt(sum / data.length);
      levelFill.style.width = `${Math.min(100, Math.round(rms * 320))}%`;
      meterFrame = window.requestAnimationFrame(tick);
    };
    tick();
  } catch {
    /* the level meter is optional */
  }
}

function stopMeter() {
  if (meterFrame) {
    window.cancelAnimationFrame(meterFrame);
    meterFrame = null;
  }
  if (meterContext) {
    meterContext.close().catch(() => {});
    meterContext = null;
  }
  levelFill.style.width = "0%";
}

function cleanupStream() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
}

function setRecordingUi(isRecording) {
  recordBtn.classList.toggle("recording", isRecording);
  recordBtn.setAttribute("aria-pressed", String(isRecording));
  recordBtnLabel.textContent = isRecording ? "Stop recording" : "Start recording";
}

async function startRecording() {
  clearRecorderStatus();

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia
      || typeof window.MediaRecorder === "undefined") {
    setRecorderStatus(
      "Recording is not supported in this browser. Use Chrome, Edge, Firefox or " +
      "Safari over http://127.0.0.1 or https, or upload a file instead.",
      true
    );
    return;
  }

  try {
    // Browser "enhancements" alter the spectrum, so ask for the raw signal.
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
  } catch (error) {
    const name = error && error.name;
    let message = `Could not start recording: ${error && error.message ? error.message : error}.`;
    if (name === "NotAllowedError" || name === "SecurityError") {
      message = "Microphone permission was denied. Allow microphone access in the " +
                "browser, then try again.";
    } else if (name === "NotFoundError" || name === "DevicesNotFoundError") {
      message = "No microphone was found. Connect a microphone, or upload a file instead.";
    }
    setRecorderStatus(message, true);
    cleanupStream();
    return;
  }

  const mimeType = pickRecordingMimeType();
  let recorder;
  try {
    recorder = mimeType
      ? new MediaRecorder(mediaStream, { mimeType })
      : new MediaRecorder(mediaStream);
  } catch (error) {
    setRecorderStatus(`Could not start the recorder: ${error.message || error}`, true);
    cleanupStream();
    return;
  }

  recordedChunks = [];
  mediaRecorder = recorder;

  recorder.addEventListener("dataavailable", (event) => {
    if (event.data && event.data.size > 0) recordedChunks.push(event.data);
  });
  recorder.addEventListener("stop", () => {
    finalizeRecording(recorder.mimeType || mimeType || "audio/webm");
  });
  recorder.addEventListener("error", (event) => {
    const error = event && event.error;
    setRecorderStatus(`Recording failed: ${error && error.message ? error.message : "unknown error"}`, true);
    stopRecording();
  });

  recorder.start(250);
  recordingStartedAt = Date.now();
  setRecordingUi(true);
  startTimer();
  startMeter(mediaStream);
  setRecorderStatus(
    `Recording… you can record up to ${RECORD_MAX_SECONDS} seconds. ` +
    "Speak normally, about one hand-span from the microphone."
  );
}

function stopRecording() {
  setRecordingUi(false);
  stopTimer();
  stopMeter();
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  // The microphone stream is released once the recorder's final chunk has been
  // handed over (see finalizeRecording), so the last word is never cut off.
}

function finalizeRecording(mimeType) {
  cleanupStream();
  const chunks = recordedChunks;
  recordedChunks = [];
  mediaRecorder = null;
  recordTimer.textContent = formatDuration(recordingSeconds);

  const blob = new Blob(chunks, { type: mimeType || "audio/webm" });
  if (!blob.size) {
    setRecorderStatus("No audio was captured. Check the microphone and try again.", true);
    return;
  }
  if (recordingSeconds < RECORD_MIN_SECONDS) {
    setRecorderStatus(
      `That recording was too short (${recordingSeconds.toFixed(1)} s). ` +
      `Record at least ${RECORD_MIN_SECONDS} second and try again.`,
      true
    );
    return;
  }

  setRecorderStatus("Converting the recording to WAV…");
  recordBtn.disabled = true;

  recordingToWav(blob)
    .then((wavBlob) => {
      showRecording(wavBlob);
    })
    .catch((error) => {
      setRecorderStatus(
        `This browser could not convert the recording (${error.message || error}). ` +
        "Try uploading the file instead.",
        true
      );
    })
    .finally(() => {
      recordBtn.disabled = false;
    });
}

function showRecording(wavBlob) {
  pendingRecordingBlob = wavBlob;
  if (pendingRecordingUrl) URL.revokeObjectURL(pendingRecordingUrl);
  pendingRecordingUrl = URL.createObjectURL(wavBlob);
  recordingAudio.src = pendingRecordingUrl;
  recordingPreview.hidden = false;

  setRecorderStatus(
    `Recording ready: ${formatDuration(recordingSeconds)}, ` +
    `${formatSize(wavBlob.size)}. Listen back, then press Analyse this recording.`
  );
}

function discardRecording() {
  pendingRecordingBlob = null;
  if (pendingRecordingUrl) {
    URL.revokeObjectURL(pendingRecordingUrl);
    pendingRecordingUrl = null;
  }
  recordingAudio.removeAttribute("src");
  recordingAudio.load();
  recordingPreview.hidden = true;
  recordTimer.textContent = "0:00";
  recordingSeconds = 0;
  clearRecorderStatus();
}

recordBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    stopRecording();
    return;
  }
  startRecording();
});

analyseRecordingBtn.addEventListener("click", () => {
  if (!pendingRecordingBlob) return;

  const now = new Date();
  const pad = (value) => String(value).padStart(2, "0");
  const name =
    `recording-${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
    `-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}.wav`;
  const file = new File([pendingRecordingBlob], name, {
    type: "audio/wav",
    lastModified: Date.now(),
  });

  addFiles([file]);
  form.requestSubmit();
});

discardRecordingBtn.addEventListener("click", () => {
  discardRecording();
});

/* A finished recording is still just a normal WAV file, so it goes through
   exactly the same /predict endpoint as an upload. */
window.addEventListener("beforeunload", () => {
  if (pendingRecordingUrl) URL.revokeObjectURL(pendingRecordingUrl);
});

/* Show which model is actually loaded (best effort, does not block the page). */
fetch("/health")
  .then((response) => response.json())
  .then((data) => {
    const model = data.model || {};
    if (!model.loaded) return;
    modelPill.textContent =
      `ASVspoof 2019 · ${model.model_name || "model"} · ` +
      `${model.feature_set || ""} · calibrated`;
  })
  .catch(() => {});

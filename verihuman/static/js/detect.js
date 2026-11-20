/* VeriHuman — detection workspace.
   Uploads go to /api/analyze, which calls the existing model predictor.
   No prediction is ever computed in the browser. */
(function () {
  "use strict";

  var root = document.getElementById("detect-root");
  if (!root) return;

  var dropzone = document.getElementById("dropzone");
  var input = document.getElementById("file-input");
  var browseBtn = document.getElementById("browse-btn");
  var queueEl = document.getElementById("file-queue");
  var queueEmpty = document.getElementById("queue-empty");
  var countEl = document.getElementById("file-count");
  var analyzeBtn = document.getElementById("analyze-btn");
  var clearBtn = document.getElementById("clear-btn");
  var resultsEl = document.getElementById("results");
  var summaryEl = document.getElementById("results-summary");
  var errorEl = document.getElementById("detect-error");
  var resultsPanel = document.getElementById("results-panel");

  var playerCard = document.getElementById("player-card");
  var playerFile = document.getElementById("player-file");
  var playerSub = document.getElementById("player-sub");
  var playBtn = document.getElementById("play-btn");
  var timeEl = document.getElementById("player-time");
  var metaEl = document.getElementById("player-meta");

  var MAX_FILES = parseInt(root.getAttribute("data-max-files"), 10) || 10;
  var MAX_BYTES = (parseFloat(root.getAttribute("data-max-mb")) || 50) * 1024 * 1024;
  var ALLOWED = ["wav", "flac", "mp3", "ogg", "webm", "m4a", "oga"];

  var entries = [];
  var activeId = null;
  var wavesurfer = null;
  var audioCtx = null;
  var busy = false;
  var nextId = 1;

  var recordBtn = document.getElementById("record-btn");
  var recordTime = document.getElementById("record-time");
  var recorderStatus = document.getElementById("recorder-status");
  var levelFill = document.getElementById("level-fill");
  var recordingPreview = document.getElementById("recording-preview");
  var recordingAudio = document.getElementById("recording-audio");
  var analyseRecordingBtn = document.getElementById("analyse-recording-btn");
  var downloadRecordingBtn = document.getElementById("download-recording-btn");
  var discardRecordingBtn = document.getElementById("discard-recording-btn");
  var mediaRecorder = null;
  var mediaStream = null;
  var recordedChunks = [];
  var recordingStartedAt = 0;
  var recordingSeconds = 0;
  var recordingTimer = null;
  var meterFrame = null;
  var meterContext = null;
  var pendingRecordingBlob = null;
  var pendingRecordingUrl = null;

  var VERDICTS = {
    BONAFIDE: { slug: "bonafide", title: "Human Voice Detected", icon: "shield-check" },
    SPOOF: { slug: "spoof", title: "Potential AI-Generated or Manipulated Voice", icon: "shield-alert" },
    UNCERTAIN: { slug: "uncertain", title: "Inconclusive Audio Analysis", icon: "shield-question" }
  };

  function esc(value) {
    return VH.escape(value);
  }

  function showError(message) {
    if (!errorEl) return;
    errorEl.innerHTML =
      VH.iconMarkup("triangle-alert", 16) + "<span>" + esc(message) + "</span>";
    errorEl.hidden = false;
    VH.icons(errorEl);
  }

  function clearError() {
    if (errorEl) {
      errorEl.hidden = true;
      errorEl.innerHTML = "";
    }
  }

  function extension(name) {
    var parts = String(name || "").split(".");
    return parts.length > 1 ? parts.pop().toLowerCase() : "";
  }

  function prettyFormat(ext) {
    return (ext || "").toUpperCase() || "—";
  }

  document.querySelectorAll("[data-detect-mode]").forEach(function (tab) {
    tab.addEventListener("click", function () {
      var mode = tab.getAttribute("data-detect-mode");
      document.querySelectorAll("[data-detect-mode]").forEach(function (item) {
        var active = item === tab;
        item.classList.toggle("active", active);
        item.setAttribute("aria-selected", active ? "true" : "false");
      });
      document.querySelectorAll("[data-detect-panel]").forEach(function (panel) {
        panel.hidden = panel.getAttribute("data-detect-panel") !== mode;
      });
    });
  });

  function clock(seconds) {
    var value = Math.max(0, Math.floor(Number(seconds) || 0));
    return Math.floor(value / 60) + ":" + String(value % 60).padStart(2, "0");
  }

  function pickRecordingMimeType() {
    if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) return "";
    var types = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
    for (var i = 0; i < types.length; i += 1) {
      if (MediaRecorder.isTypeSupported(types[i])) return types[i];
    }
    return "";
  }

  function encodeWav(samples, sampleRate) {
    var buffer = new ArrayBuffer(44 + samples.length * 2);
    var view = new DataView(buffer);
    function text(offset, value) {
      for (var i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i));
    }
    text(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); text(8, "WAVE");
    text(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    text(36, "data"); view.setUint32(40, samples.length * 2, true);
    var offset = 44;
    for (var n = 0; n < samples.length; n += 1) {
      var sample = Math.max(-1, Math.min(1, samples[n]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
      offset += 2;
    }
    return new Blob([buffer], { type: "audio/wav" });
  }

  function recordingToWav(blob) {
    var Ctx = window.AudioContext || window.webkitAudioContext;
    var Offline = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    if (!Ctx || !Offline) return Promise.reject(new Error("This browser cannot convert microphone audio."));
    var context = new Ctx();
    return blob.arrayBuffer().then(function (encoded) {
      return context.decodeAudioData(encoded);
    }).then(function (decoded) {
      var targetRate = 16000;
      var duration = decoded.duration || decoded.length / decoded.sampleRate;
      var offline = new Offline(1, Math.max(1, Math.ceil(duration * targetRate)), targetRate);
      var source = offline.createBufferSource();
      source.buffer = decoded;
      source.connect(offline.destination);
      source.start();
      return offline.startRendering();
    }).then(function (rendered) {
      context.close && context.close().catch(function () {});
      return encodeWav(rendered.getChannelData(0), 16000);
    }).catch(function (error) {
      context.close && context.close().catch(function () {});
      throw error;
    });
  }

  function setRecorderStatus(message, isError) {
    if (!recorderStatus) return;
    recorderStatus.textContent = message;
    recorderStatus.className = isError ? "text-spoof" : "text-muted";
  }

  function stopMeter() {
    if (meterFrame) cancelAnimationFrame(meterFrame);
    meterFrame = null;
    if (meterContext) { meterContext.close().catch(function () {}); meterContext = null; }
    if (levelFill) levelFill.style.width = "0%";
  }

  function stopRecording() {
    if (recordBtn) { recordBtn.classList.remove("recording"); recordBtn.setAttribute("aria-pressed", "false"); }
    if (recordingTimer) { clearInterval(recordingTimer); recordingTimer = null; }
    stopMeter();
    if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
  }

  function startMeter(stream) {
    var Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    try {
      meterContext = new Ctx();
      var source = meterContext.createMediaStreamSource(stream);
      var analyser = meterContext.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      var data = new Uint8Array(analyser.fftSize);
      function tick() {
        analyser.getByteTimeDomainData(data);
        var sum = 0;
        for (var i = 0; i < data.length; i += 1) {
          var value = (data[i] - 128) / 128;
          sum += value * value;
        }
        if (levelFill) levelFill.style.width = Math.min(100, Math.round(Math.sqrt(sum / data.length) * 320)) + "%";
        meterFrame = requestAnimationFrame(tick);
      }
      tick();
    } catch (error) { stopMeter(); }
  }

  function cleanupStream() {
    if (mediaStream) {
      mediaStream.getTracks().forEach(function (track) { track.stop(); });
      mediaStream = null;
    }
  }

  function finalizeRecording(mimeType) {
    cleanupStream();
    var blob = new Blob(recordedChunks, { type: mimeType || "audio/webm" });
    recordedChunks = [];
    mediaRecorder = null;
    if (!blob.size) { setRecorderStatus("No audio was captured.", true); return; }
    if (recordingSeconds < 0.5) { setRecorderStatus("Record for at least 1 second.", true); return; }
    setRecorderStatus("Converting microphone audio to 16 kHz mono WAV…");
    if (recordBtn) recordBtn.disabled = true;
    recordingToWav(blob).then(function (wavBlob) {
      pendingRecordingBlob = wavBlob;
      if (pendingRecordingUrl) URL.revokeObjectURL(pendingRecordingUrl);
      pendingRecordingUrl = URL.createObjectURL(wavBlob);
      if (recordingAudio) recordingAudio.src = pendingRecordingUrl;
      if (recordingPreview) recordingPreview.hidden = false;
      if (downloadRecordingBtn) {
        downloadRecordingBtn.href = pendingRecordingUrl;
        downloadRecordingBtn.download = "verihuman-recording-" + new Date().toISOString().replace(/[:.]/g, "-") + ".wav";
      }
      setRecorderStatus("Recording ready — listen back, download, or analyze.");
    }).catch(function (error) {
      setRecorderStatus("Could not convert the recording: " + (error.message || error), true);
    }).then(function () {
      if (recordBtn) recordBtn.disabled = false;
    });
  }

  if (recordBtn) {
    recordBtn.addEventListener("click", function () {
      if (mediaRecorder && mediaRecorder.state === "recording") { stopRecording(); return; }
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || typeof MediaRecorder === "undefined") {
        setRecorderStatus("Microphone recording requires a supported browser and secure local access.", true);
        return;
      }
      navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
      }).then(function (stream) {
        mediaStream = stream;
        var mimeType = pickRecordingMimeType();
        mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType: mimeType }) : new MediaRecorder(stream);
        recordedChunks = [];
        mediaRecorder.addEventListener("dataavailable", function (event) {
          if (event.data && event.data.size > 0) recordedChunks.push(event.data);
        });
        mediaRecorder.addEventListener("stop", function () { finalizeRecording(mediaRecorder.mimeType || mimeType || "audio/webm"); });
        mediaRecorder.start(250);
        recordingStartedAt = Date.now();
        recordingSeconds = 0;
        recordBtn.classList.add("recording");
        recordBtn.setAttribute("aria-pressed", "true");
        if (recordTime) recordTime.textContent = "0:00";
        setRecorderStatus("Recording…");
        startMeter(stream);
        recordingTimer = setInterval(function () {
          recordingSeconds = (Date.now() - recordingStartedAt) / 1000;
          if (recordTime) recordTime.textContent = clock(recordingSeconds);
          if (recordingSeconds >= 60) stopRecording();
        }, 200);
      }).catch(function (error) {
        setRecorderStatus("Could not start the microphone: " + (error.message || error), true);
      });
    });
  }

  if (analyseRecordingBtn) {
    analyseRecordingBtn.addEventListener("click", function () {
      if (!pendingRecordingBlob) return;
      var stamp = new Date().toISOString().replace(/[:.]/g, "-");
      var file = new File([pendingRecordingBlob], "verihuman-recording-" + stamp + ".wav", { type: "audio/wav" });
      clearError();
      addFiles([file]);
      document.querySelectorAll("[data-detect-mode]").forEach(function (item) {
        if (item.getAttribute("data-detect-mode") === "upload") item.click();
      });
      setTimeout(function () {
        if (analyzeBtn && !analyzeBtn.disabled) analyzeBtn.click();
      }, 100);
    });
  }

  if (discardRecordingBtn) {
    discardRecordingBtn.addEventListener("click", function () {
      pendingRecordingBlob = null;
      if (pendingRecordingUrl) URL.revokeObjectURL(pendingRecordingUrl);
      pendingRecordingUrl = null;
      if (recordingAudio) recordingAudio.removeAttribute("src");
      if (recordingPreview) recordingPreview.hidden = true;
      setRecorderStatus("Recording discarded.");
    });
  }

  /* ----------------------------------------------------------- metadata -- */

  function audioContext() {
    if (audioCtx) return audioCtx;
    var Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    try {
      audioCtx = new Ctx();
    } catch (error) {
      audioCtx = null;
    }
    return audioCtx;
  }

  function probe(file) {
    var entry = {
      duration: null,
      sampleRate: null,
      channels: null
    };

    var decode = function () {
      var ctx = audioContext();
      if (!ctx || typeof file.arrayBuffer !== "function") return Promise.reject();
      return file
        .arrayBuffer()
        .then(function (buffer) {
          return new Promise(function (resolve, reject) {
            // Callback form works on older Safari too.
            var promise = ctx.decodeAudioData(buffer.slice(0), resolve, reject);
            if (promise && typeof promise.then === "function") promise.then(resolve, reject);
          });
        })
        .then(function (decoded) {
          entry.duration = decoded.duration;
          entry.sampleRate = decoded.sampleRate;
          entry.channels = decoded.numberOfChannels;
          return entry;
        });
    };

    var fallback = function () {
      return new Promise(function (resolve) {
        var audio = document.createElement("audio");
        audio.preload = "metadata";
        var url = URL.createObjectURL(file);
        var finish = function () {
          if (audio.duration && isFinite(audio.duration)) entry.duration = audio.duration;
          URL.revokeObjectURL(url);
          resolve(entry);
        };
        audio.onloadedmetadata = finish;
        audio.onerror = finish;
        audio.src = url;
        setTimeout(finish, 4000);
      });
    };

    return decode().catch(fallback);
  }

  /* ---------------------------------------------------------------- queue */

  function status(entry, kind, label) {
    entry.status = kind;
    var node = queueEl.querySelector('[data-entry="' + entry.id + '"] .file-status');
    if (node) {
      node.className = "file-status " + (kind || "");
      node.innerHTML =
        (kind === "busy" ? '<span class="spinner"></span>' : "") + esc(label || "");
    }
  }

  function renderQueue() {
    if (!entries.length) {
      queueEl.innerHTML = "";
      queueEl.hidden = true;
      if (queueEmpty) queueEmpty.hidden = false;
    } else {
      queueEl.hidden = false;
      if (queueEmpty) queueEmpty.hidden = true;
      queueEl.innerHTML = entries
        .map(function (entry) {
          var metaBits = [
            VH.formatBytes(entry.file.size),
            prettyFormat(entry.ext),
            entry.duration != null ? VH.formatDuration(entry.duration) : null,
            entry.sampleRate ? entry.sampleRate.toLocaleString() + " Hz" : null,
            entry.channels ? entry.channels + " ch" : null
          ].filter(Boolean);

          return (
            '<div class="file-row' + (entry.id === activeId ? " is-active" : "") +
            (entry.error ? " is-error" : "") + '" data-entry="' + entry.id + '">' +
            '<span class="file-badge">' + VH.iconMarkup("audio-lines", 16) + "</span>" +
            '<div class="file-row-main">' +
            '<div class="file-row-top">' +
            '<span class="file-row-name" title="' + esc(entry.file.name) + '">' + esc(entry.file.name) + "</span>" +
            "</div>" +
            '<div class="file-row-meta">' + metaBits.map(esc).join("<span>·</span>") + "</div>" +
            '<div class="progress-line' + (entry.status === "busy" ? " indeterminate" : "") +
            '" data-progress="' + entry.id + '"' + (entry.status === "busy" ? "" : " hidden") +
            '><span></span></div>' +
            "</div>" +
            '<div class="file-row-actions">' +
            '<span class="file-status ' + (entry.status || "") + '">' +
            (entry.status === "busy" ? '<span class="spinner"></span>' : "") +
            esc(entry.error ? "Failed" : entry.status === "ok" ? "Analysed" : "") +
            "</span>" +
            '<button type="button" class="icon-btn" data-download="' + entry.id + '" title="Download audio" aria-label="Download ' + esc(entry.file.name) + '">' +
            VH.iconMarkup("download", 16) + "</button>" +
            '<button type="button" class="icon-btn" data-preview="' + entry.id + '" title="Preview waveform" aria-label="Preview ' + esc(entry.file.name) + '">' +
            VH.iconMarkup("play", 16) + "</button>" +
            '<button type="button" class="icon-btn" data-remove="' + entry.id + '" title="Remove file" aria-label="Remove ' + esc(entry.file.name) + '">' +
            VH.iconMarkup("x", 16) + "</button>" +
            "</div></div>"
          );
        })
        .join("");
      VH.icons(queueEl);
    }

    if (countEl) countEl.textContent = String(entries.length);
    analyzeBtn.disabled = busy || !entries.length;
    clearBtn.disabled = busy || !entries.length;
  }

  function addFiles(fileList) {
    clearError();
    var files = Array.prototype.slice.call(fileList || []);
    if (!files.length) return;

    var rejected = [];

    files.forEach(function (file) {
      var ext = extension(file.name);
      if (ALLOWED.indexOf(ext) === -1) {
        rejected.push(file.name + " — unsupported format");
        return;
      }
      if (!file.size) {
        rejected.push(file.name + " — empty file");
        return;
      }
      if (file.size > MAX_BYTES) {
        rejected.push(file.name + " — larger than " + Math.round(MAX_BYTES / (1024 * 1024)) + " MB");
        return;
      }
      var duplicate = entries.some(function (entry) {
        return (
          entry.file.name === file.name &&
          entry.file.size === file.size &&
          entry.file.lastModified === file.lastModified
        );
      });
      if (duplicate) return;
      if (entries.length >= MAX_FILES) {
        rejected.push(file.name + " — batch limit is " + MAX_FILES + " files");
        return;
      }

      entries.push({
        id: nextId++,
        file: file,
        ext: ext,
        url: URL.createObjectURL(file),
        duration: null,
        sampleRate: null,
        channels: null,
        status: null,
        error: null,
        result: null
      });
    });

    if (rejected.length) {
      showError(
        rejected.slice(0, 3).join(" · ") +
          (rejected.length > 3 ? " · and " + (rejected.length - 3) + " more" : "")
      );
    }

    entries
      .filter(function (entry) { return entry.duration == null; })
      .forEach(function (entry) {
        probe(entry.file).then(function (meta) {
          entry.duration = meta.duration;
          entry.sampleRate = meta.sampleRate;
          entry.channels = meta.channels;
          renderQueue();
          if (entry.id === activeId) showPlayer(entry);
        });
      });

    renderQueue();

    var latest = entries[entries.length - 1];
    if (latest && activeId == null) loadPreview(latest, false);
    else if (latest && entries.length === 1) loadPreview(latest, false);
  }

  /* -------------------------------------------------------------- player -- */

  function destroyWave() {
    if (wavesurfer) {
      try { wavesurfer.destroy(); } catch (error) {}
      wavesurfer = null;
    }
    var container = document.getElementById("waveform");
    if (container) container.innerHTML = "";
  }

  function showPlayer(entry) {
    if (!playerCard || !entry) return;
    playerCard.hidden = false;
    playerFile.textContent = entry.file.name;
    playerSub.textContent =
      [prettyFormat(entry.ext), VH.formatBytes(entry.file.size)]
        .concat(entry.duration != null ? [VH.formatDuration(entry.duration)] : [])
        .join(" · ");

    if (metaEl) {
      metaEl.innerHTML = [
        ["Filename", entry.file.name],
        ["Format", prettyFormat(entry.ext)],
        ["File size", VH.formatBytes(entry.file.size)],
        ["Duration", entry.duration != null ? VH.formatDuration(entry.duration) : "Reading…"],
        ["Sample rate", entry.sampleRate ? entry.sampleRate.toLocaleString() + " Hz" : "Not available"],
        ["Channels", entry.channels ? String(entry.channels) : "Not available"]
      ]
        .map(function (pair) {
          return "<div><span>" + esc(pair[0]) + "</span><b>" + esc(pair[1]) + "</b></div>";
        })
        .join("");
    }
  }

  function loadPreview(entry, autoplay) {
    activeId = entry.id;
    showPlayer(entry);
    renderQueue();
    destroyWave();

    var container = document.getElementById("waveform");
    if (!container) return;

    if (!window.WaveSurfer) {
      container.innerHTML = '<p class="vh-hint" style="padding:24px;text-align:center">Waveform preview is unavailable in this browser. The audio can still be analysed.</p>';
      return;
    }

    var styles = getComputedStyle(document.documentElement);
    var accent = styles.getPropertyValue("--text").trim() || "#FAFAFA";
    var muted = styles.getPropertyValue("--muted").trim() || "#A1A1AA";

    try {
      wavesurfer = window.WaveSurfer.create({
        container: container,
        height: 96,
        waveColor: muted,
        progressColor: accent,
        cursorColor: accent,
        cursorWidth: 1,
        barWidth: 2,
        barGap: 2,
        barRadius: 2,
        normalize: true,
        url: entry.url
      });
    } catch (error) {
      container.innerHTML = '<p class="vh-hint" style="padding:24px;text-align:center">Preview could not be drawn for this file.</p>';
      return;
    }

    wavesurfer.on("ready", function () {
      timeEl.textContent = "0:00 / " + clock(wavesurfer.getDuration());
      if (autoplay) wavesurfer.play();
    });
    wavesurfer.on("timeupdate", function (current) {
      timeEl.textContent = clock(current) + " / " + clock(wavesurfer ? wavesurfer.getDuration() : 0);
    });
    wavesurfer.on("play", function () { setPlayIcon(true); });
    wavesurfer.on("pause", function () { setPlayIcon(false); });
    wavesurfer.on("finish", function () { setPlayIcon(false); });
    wavesurfer.on("error", function () {
      container.innerHTML = '<p class="vh-hint" style="padding:24px;text-align:center">This format cannot be previewed here, but it can still be analysed.</p>';
    });
  }

  function clock(seconds) {
    var value = Math.max(0, Math.floor(Number(seconds) || 0));
    var mins = Math.floor(value / 60);
    var rest = value % 60;
    return mins + ":" + String(rest).padStart(2, "0");
  }

  function setPlayIcon(playing) {
    if (!playBtn) return;
    playBtn.innerHTML = VH.iconMarkup(playing ? "pause" : "play", 16);
    playBtn.setAttribute("aria-label", playing ? "Pause preview" : "Play preview");
    VH.icons(playBtn);
  }

  if (playBtn) {
    playBtn.addEventListener("click", function () {
      if (!wavesurfer) {
        var entry = entries.find(function (item) { return item.id === activeId; });
        if (entry) loadPreview(entry, true);
        return;
      }
      wavesurfer.playPause();
    });
  }

  /* ---------------------------------------------------------------- events */

  if (browseBtn && input) {
    browseBtn.addEventListener("click", function (event) {
      event.stopPropagation();
      input.click();
    });
  }

  if (dropzone && input) {
    dropzone.addEventListener("click", function (event) {
      if (event.target.closest("button")) return;
      input.click();
    });
    dropzone.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        input.click();
      }
    });
    input.addEventListener("change", function () {
      addFiles(input.files);
      input.value = "";
    });

    ["dragenter", "dragover"].forEach(function (name) {
      dropzone.addEventListener(name, function (event) {
        event.preventDefault();
        dropzone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach(function (name) {
      dropzone.addEventListener(name, function (event) {
        event.preventDefault();
        dropzone.classList.remove("dragover");
      });
    });
    dropzone.addEventListener("drop", function (event) {
      if (event.dataTransfer && event.dataTransfer.files) addFiles(event.dataTransfer.files);
    });
  }

  queueEl.addEventListener("click", function (event) {
    var download = event.target.closest("[data-download]");
    if (download) {
      var downloadId = parseInt(download.getAttribute("data-download"), 10);
      var downloadEntry = entries.find(function (item) { return item.id === downloadId; });
      if (downloadEntry) {
        var url = URL.createObjectURL(downloadEntry.file);
        var link = document.createElement("a");
        link.href = url;
        link.download = downloadEntry.file.name;
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
      }
      return;
    }

    var remove = event.target.closest("[data-remove]");
    if (remove) {
      var removeId = parseInt(remove.getAttribute("data-remove"), 10);
      var entry = entries.find(function (item) { return item.id === removeId; });
      if (!entry || busy) return;
      URL.revokeObjectURL(entry.url);
      entries = entries.filter(function (item) { return item.id !== removeId; });
      if (activeId === removeId) {
        activeId = null;
        destroyWave();
        if (playerCard) playerCard.hidden = true;
        if (entries.length) loadPreview(entries[0], false);
      }
      renderQueue();
      return;
    }

    var preview = event.target.closest("[data-preview]");
    if (preview) {
      var previewId = parseInt(preview.getAttribute("data-preview"), 10);
      var picked = entries.find(function (item) { return item.id === previewId; });
      if (picked) loadPreview(picked, true);
    }
  });

  if (clearBtn) {
    clearBtn.addEventListener("click", function () {
      if (busy) return;
      entries.forEach(function (entry) { URL.revokeObjectURL(entry.url); });
      entries = [];
      activeId = null;
      destroyWave();
      if (playerCard) playerCard.hidden = true;
      if (resultsEl) {
        resultsEl.innerHTML = "";
        resultsEl.hidden = true;
      }
      if (resultsPanel) resultsPanel.hidden = true;
      clearError();
      renderQueue();
    });
  }

  /* --------------------------------------------------------------- results */

  function verdictOf(result) {
    return VERDICTS[String(result.prediction || "").toUpperCase()] || VERDICTS.UNCERTAIN;
  }

  function pct(value) {
    if (value == null || value === "") return null;
    var number = Number(value);
    return isFinite(number) ? number : null;
  }

  function scoreRow(label, value, cls, note) {
    if (value == null) {
      return (
        '<div class="score-row"><div class="score-row-head"><span>' + esc(label) +
        '</span><b class="text-muted">Not available</b></div>' +
        '<div class="score-bar"><span style="width:0%"></span></div></div>'
      );
    }
    var clamped = Math.max(0, Math.min(100, value));
    return (
      '<div class="score-row">' +
      '<div class="score-row-head"><span>' + esc(label) + "</span><b>" + value.toFixed(2) + "%</b></div>" +
      '<div class="score-bar"><span class="' + (cls || "") + '" style="width:' + clamped.toFixed(2) + '%"></span></div>' +
      (note ? '<span class="vh-hint">' + esc(note) + "</span>" : "") +
      "</div>"
    );
  }

  function chunkMarkup(result) {
    var chunks = Array.isArray(result.chunk_results) ? result.chunk_results : [];
    if (!chunks.length) return "";

    var bars = chunks
      .slice(0, 40)
      .map(function (chunk, index) {
        var score = Number(chunk.spoof_score) || 0;
        var start = Number(chunk.start_seconds);
        var label = isFinite(start) ? start.toFixed(1) + "s" : "#" + (chunk.index != null ? chunk.index : index + 1);
        return (
          '<div class="chunk-bar-row">' +
          "<span>" + esc(chunk.index != null ? chunk.index : index + 1) + "</span>" +
          "<span>" + esc(label) + "</span>" +
          '<span class="track"><span style="width:' + Math.max(0, Math.min(100, score)).toFixed(2) + '%"></span></span>' +
          '<span class="val">' + score.toFixed(1) + "%</span>" +
          "</div>"
        );
      })
      .join("");

    return (
      '<div class="chunk-section">' +
      '<button type="button" class="chunk-toggle" aria-expanded="false">' +
      "<span>Chunk-level results (" + chunks.length + ")</span>" +
      VH.iconMarkup("chevron-down", 16) +
      "</button>" +
      '<div class="chunk-body" hidden><div class="chunk-bars">' + bars + "</div>" +
      (chunks.length > 40 ? '<p class="vh-hint" style="margin-top:8px">Showing the first 40 chunks. The full list is in the PDF report.</p>' : "") +
      "</div></div>"
    );
  }

  function resultMarkup(result) {
    var verdict = verdictOf(result);
    var score = pct(result.spoof_score);
    var raw = pct(result.raw_spoof_score);
    var confidence = pct(result.confidence);
    var calibrated = Boolean(result.calibrated);
    var chunks = pct(result.chunks_analysed);
    var stamp = new Date().toLocaleString(undefined, {
      day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit"
    });
    var ring = score == null ? 0 : Math.max(0, Math.min(100, score));

    var actions = [];
    if (result.report_url) actions.push('<a class="vh-btn vh-btn-outline vh-btn-sm" href="' + esc(result.report_url) + '">' + VH.iconMarkup("file-text", 15) + " Full report</a>");
    actions.push('<button type="button" class="vh-btn vh-btn-subtle vh-btn-sm" data-download-result="' +
      esc(result.filename || "") + '">' + VH.iconMarkup("download", 15) + " Download original audio</button>");
    if (result.pdf_url) actions.push('<a class="vh-btn vh-btn-subtle vh-btn-sm" href="' + esc(result.pdf_url) + '">' + VH.iconMarkup("download", 15) + " PDF</a>");
    if (result.csv_url) actions.push('<a class="vh-btn vh-btn-subtle vh-btn-sm" href="' + esc(result.csv_url) + '">' + VH.iconMarkup("table", 15) + " CSV</a>");
    actions.push('<button type="button" class="vh-btn vh-btn-ghost vh-btn-sm" data-analyze-another>' + VH.iconMarkup("upload", 15) + " Upload another</button>");

    var confidenceNote = calibrated
      ? result.confidence_label
        ? result.confidence_label + " · calibrated score"
        : "Calibrated score"
      : "Model is not calibrated; confidence hidden";

    return (
      '<article class="result-card">' +
      '<div class="result-head">' +
      '<div class="r-file">' +
      '<span class="file-badge">' + VH.iconMarkup("audio-lines", 18) + "</span>" +
      "<div><div class=\"result-sub\">" + esc(result.filename || "Audio file") + "</div>" +
      '<h3 class="result-title">' + esc(verdict.title) + "</h3>" +
      '<div class="result-sub">' + esc(result.model_name || "Existing VeriHuman model") +
      (result.feature_set ? " · " + esc(result.feature_set) + " features" : "") + "</div>" +
      "</div></div>" +
      '<span class="verdict verdict-' + verdict.slug + '"><span class="verdict-dot"></span>' +
      esc(result.display_label || result.prediction || "UNCERTAIN") + "</span>" +
      "</div>" +

      '<div class="result-body">' +
      '<div class="score-panel">' +
      '<div class="score-ring" style="--pct:' + ring.toFixed(2) + ";--ring:var(--" + verdict.slug + ')">' +
      '<div class="score-ring-inner"><b>' + (score == null ? "—" : score.toFixed(1) + "%") + "</b><span>Spoof score</span></div>" +
      "</div>" +
      '<p class="score-caption">' + (score == null ? "Score not returned" : "Calibrated probability of synthetic or manipulated speech") + "</p>" +
      "</div>" +

      '<div class="score-list">' +
      scoreRow("Human probability", score == null ? null : 100 - score, "bonafide", "Calculated from the model's AI/SPOOF probability") +
      scoreRow("Spoof score", score, "spoof") +
      scoreRow("AI probability", score, "spoof", "Calibrated SPOOF probability returned by the model") +
      scoreRow("Raw model score", raw, "") +
      scoreRow("Confidence", calibrated ? confidence : null, "uncertain", calibrated ? confidenceNote : "Not available for this model") +
      '<div class="result-metrics">' +
      "<div>Chunks<b>" + (chunks == null ? "—" : chunks) + "</b></div>" +
      "<div>Chunk length<b>" + (result.chunk_seconds != null ? esc(result.chunk_seconds) + "s" : "—") + "</b></div>" +
      "<div>Disagreement<b>" + (result.chunk_disagreement != null ? (Number(result.chunk_disagreement) * 100).toFixed(1) + "%" : "—") + "</b></div>" +
      "<div>Processing<b>" + esc(VH.formatMs(result.processing_ms)) + "</b></div>" +
      "</div>" +
      "</div></div>" +

      chunkMarkup(result) +

      '<div class="result-foot">' +
      '<span class="left">' + VH.iconMarkup("clock", 14) + esc(stamp) +
      (result.report_id ? " · " + esc(result.report_id) : "") + "</span>" +
      '<div class="result-actions">' + actions.join("") + "</div>" +
      "</div></article>"
    );
  }

  function renderResults(payload) {
    if (!resultsEl) return;
    var results = payload.results || [];
    var ok = results.filter(function (result) { return !result.error; });
    var failed = results.filter(function (result) { return result.error; });

    if (!results.length) {
      resultsEl.innerHTML = "";
      resultsEl.hidden = true;
      return;
    }

    var summary = "";
    if (ok.length) {
      var counts = { bonafide: 0, spoof: 0, uncertain: 0 };
      ok.forEach(function (result) {
        var key = String(result.prediction || "").toLowerCase();
        if (counts[key] != null) counts[key] += 1;
      });
      summary =
        '<div class="notice" style="margin-bottom:16px">' +
        VH.iconMarkup("info", 16) +
        "<div><strong>" + ok.length + " of " + results.length + " file" + (results.length === 1 ? "" : "s") +
        " analysed.</strong> " +
        counts.bonafide + " bonafide · " + counts.spoof + " spoof · " + counts.uncertain + " uncertain. " +
        "Automated detection is an estimate, not definitive proof of authenticity.</div></div>";
    }

    var failures = failed
      .map(function (result) {
        return (
          '<div class="notice notice-warn" style="margin-bottom:10px">' +
          VH.iconMarkup("triangle-alert", 16) +
          "<div><strong>" + esc(result.filename) + "</strong><br>" + esc(result.error) + "</div></div>"
        );
      })
      .join("");

    if (summaryEl) summaryEl.innerHTML = summary;
    resultsEl.innerHTML = failures + ok.map(resultMarkup).join("");
    resultsEl.hidden = false;
    if (resultsPanel) resultsPanel.hidden = false;
    VH.icons(resultsEl);

    resultsEl.querySelectorAll(".chunk-toggle").forEach(function (toggle) {
      toggle.addEventListener("click", function () {
        var body = toggle.nextElementSibling;
        var open = toggle.getAttribute("aria-expanded") === "true";
        toggle.setAttribute("aria-expanded", open ? "false" : "true");
        if (body) body.hidden = open;
      });
    });

    resultsEl.querySelectorAll("[data-analyze-another]").forEach(function (button) {
      button.addEventListener("click", resetWorkspace);
    });

    resultsEl.querySelectorAll("[data-download-result]").forEach(function (button) {
      button.addEventListener("click", function () {
        var filename = button.getAttribute("data-download-result");
        var entry = entries.find(function (item) {
          return item.file.name === filename;
        });
        if (!entry) {
          showError("The original audio is no longer in this browser session.");
          return;
        }
        var url = URL.createObjectURL(entry.file);
        var link = document.createElement("a");
        link.href = url;
        link.download = entry.file.name;
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
      });
    });

    resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function resetWorkspace() {
    if (busy) return;
    entries.forEach(function (entry) { URL.revokeObjectURL(entry.url); });
    entries = [];
    activeId = null;
    destroyWave();
    if (playerCard) playerCard.hidden = true;
    if (resultsEl) {
      resultsEl.innerHTML = "";
      resultsEl.hidden = true;
    }
    if (resultsPanel) resultsPanel.hidden = true;
    if (summaryEl) summaryEl.innerHTML = "";
    clearError();
    renderQueue();
    if (dropzone) dropzone.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  /* --------------------------------------------------------------- analyse */

  function setBusy(value) {
    busy = value;
    analyzeBtn.disabled = value || !entries.length;
    clearBtn.disabled = value || !entries.length;
    analyzeBtn.innerHTML = value
      ? '<span class="spinner"></span> Analysing…'
      : VH.iconMarkup("sparkles", 16) + " Analyze Audio";
    VH.icons(analyzeBtn);
    entries.forEach(function (entry) {
      var progress = queueEl.querySelector('[data-progress="' + entry.id + '"]');
      if (progress) {
        progress.hidden = !value;
        progress.classList.toggle("indeterminate", value);
      }
    });
  }

  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", function () {
      if (busy || !entries.length) return;
      clearError();

      if (entries.length > MAX_FILES) {
        showError("Upload at most " + MAX_FILES + " files per batch.");
        return;
      }

      var form = new FormData();
      entries.forEach(function (entry) {
        form.append("files", entry.file, entry.file.name);
        entry.error = null;
        entry.result = null;
        entry.status = "busy";
      });
      renderQueue();
      setBusy(true);

      VH.apiFetch(root.getAttribute("data-api"), { method: "POST", body: form })
        .then(function (payload) {
          var results = payload.results || [];
          entries.forEach(function (entry, index) {
            var result = results[index];
            if (!result) {
              entry.status = "fail";
              entry.error = "No result was returned.";
              return;
            }
            if (result.error) {
              entry.status = "fail";
              entry.error = result.error;
            } else {
              entry.status = "ok";
              entry.result = result;
            }
          });
          renderQueue();
          renderResults(payload);
        })
        .catch(function (error) {
          entries.forEach(function (entry) { entry.status = "fail"; });
          renderQueue();
          showError(error.message || "The analysis request failed.");
        })
        .then(function () {
          setBusy(false);
          renderQueue();
        });
    });
  }

  renderQueue();
})();

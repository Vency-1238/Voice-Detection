# data/

This folder is for **your own audio files** — recordings you want to test manually,
outside the ASVspoof dataset.

The ASVspoof 2019 LA dataset itself is *not* copied here. It is 20+ GB and already lives at:

    F:\minor project\LA

`config.py` points to it via `DATASET_ROOT`.

## Suggested layout for your own test clips

```
data/
├── real/          # your own voice recordings (record with Windows Voice Recorder)
│   ├── my_voice_1.wav
│   └── my_voice_2.wav
└── ai/            # AI-generated clips (e.g. from a TTS website)
    ├── tts_1.mp3
    └── tts_2.ogg
```

Recording your own voice and testing it is the honest "completely new audio, not part of
the training set" test required at the end of Stage 2.

Supported formats for the finished app: **WAV, FLAC, MP3, OGG**.

# BSSE23029 — Multimodal HCI Assistant

**Muhammad Mukarram Raza | SE305T Human Computer Interaction | Spring 2026**

Offline-first multimodal assistant running on the exam stack:
**llama3 (Ollama) · Whisper · MediaPipe / OpenCV · gradio**

No internet required on exam day. Every feature degrades gracefully if a component is missing.

---

## Quick start

```bash
# Terminal 1 — keep open the whole time
ollama serve

# Terminal 2
python app.py        # → http://localhost:7860
```

---

## What it does

| Modality | Input | How |
|---|---|---|
| Text | Typed message | Chat tab → full pipeline |
| Voice | Microphone | Whisper STT → same pipeline |
| Vision | Webcam | MediaPipe/OpenCV → label dict → LLM context |
| Offline replay | Log lines in config | `replay.log` in config.yaml → run without mic |

The pipeline per turn: **sentiment → tier → classify → respond → log → report**

Each stage can be turned on/off independently from `config.yaml`. With all stages off → pure LLM.

---

## Architecture in one picture

```
config.yaml  ─────────────────────────────────── single source of truth
     │
     ├── personas        full system prompts for every task type
     ├── stages          ordered pipeline definition (13 stage types)
     ├── tabs            which stages each gradio tab runs
     ├── categories      keyword→label classifier
     ├── sentiment       lexicon scorer (VADER-free)
     ├── scale           N-tier emotional state mapper
     ├── responses       rule table (CATEGORY|TIER → reply)
     └── vision          all detector thresholds + HUD layout

LIVE (core/provider.py)  ────── global singleton, always current
     ├── LIVE.vision     latest frame + label dict (written by camera thread)
     ├── LIVE.audio      latest recording + transcript
     └── LIVE.session    SessionState (full turn history)

pipeline/stages.py  ─────────── 13 stage implementations
     voice_input → text_input → vision_input → fuser
     → context_inject → sentiment → scale → classify
     → trajectory → respond → display → tts → log

app.py  ─────────────────────── gradio UI (config-driven tabs)
exam_adapter.py  ────────────── exam-day function name wrappers ONLY
```

---

## File map

```
app.py                  gradio entry point  →  python app.py
config.yaml             ALL tunables (edit this, not code)
config.json             auto-generated mirror (fallback if PyYAML missing)
exam_adapter.py         exact-name wrappers for exam-mandated function names
requirements.txt        official install list
models/
  yunet.onnx            YuNet neural face detector (227 KB)
  emotion_ferplus.onnx  FER+ 8-class emotion model (33 MB)

core/
  conf.py               config loader  (YAML → JSON → hardcoded defaults)
  llm.py                Ollama/llama3 client  (chat, json, classify, score, stream)
  state.py              SessionState  —  accumulates all turn data
  provider.py           LIVE singleton  —  global shared state
  executor.py           ordered stage runner with fallbacks

pipeline/
  stages.py             13 stage implementations (voice_input … logger)
  turn.py               run_turn() / process_turn() — public API
  replay.py             run_replay_from_config()  — offline log → report

text/
  sentiment.py          lexicon scorer → compound float [-1, 1]
  scale.py              score → tier dict  (assess_wellbeing equivalent)
  classify.py           keyword multi-label classifier  (classify_support_need equiv)
  trajectory.py         trend, transitions, escalations, frequency rank
  respond.py            rule table → LLM fallback → default  (nexus_respond equiv)
  report.py             risk formula + banner report  (generate_intelligence_report equiv)
  preprocess.py         tokenize, normalize, word_count, count_keywords

voice/
  capture.py            capture_audio()  — mic → float32 array
  stt.py                transcribe()  — array → {text, language, confidence}
  audio_io.py           format plumbing  (from_gradio, to_float32_mono_16k, …)

vision/
  backend.py            auto-selects: mediapipe > dnn/yunet > lbp/opencv
  faces.py              detect_faces, blink, EAR, mood, head pose (all backends)
  hands.py              count_fingers, classify_gesture, multi-hand (all backends)
  lips.py               MAR features, vowel classification  (MediaPipe only)
  smoothing.py          EMA, Kalman, MajorityBuffer, HoldTimer
  draw.py               status bar, face mesh, hand mesh, chips, gauges
  bridge.py             frame → label dict → LLM context string
  camera.py             VideoSource, run_loop, draw_text
  color_motion.py       HSV colour tracking, frame-diff / MOG2 motion
```

---

## Offline setup (do at home)

```bash
# 1. packages
uv pip install -r requirements.txt
# Apple Silicon: uv pip install "mediapipe-silicon>=0.9.2.1"

# 2. system
brew install ffmpeg          # mac
winget install ffmpeg        # windows

# 3. models
ollama pull llama3           # 4.7 GB
python -c "import whisper; whisper.load_model('base')"
python -c "import whisper; whisper.load_model('small')"   # optional backup

# 4. verify everything works
python -c "
import gradio, cv2, whisper, speech_recognition, openai, PIL, pydub, numpy
import mediapipe, requests
print('packages OK')
"
ollama run llama3 "Reply: ready"
python -m pipeline.replay     # should print a sample report
python exam_adapter.py         # should print adapter self-check
```

---

## Swap domain in 60 seconds (config only)

1. Change `active_persona` to a different persona key
2. Change `categories` keyword lists
3. Change `responses.rules` using `CATEGORY|TIER` pattern
4. Change `sentiment.positive_words` / `sentiment.negative_words`

No code changes. See [MODIFICATION_GUIDE.md](MODIFICATION_GUIDE.md) for step-by-step examples.

---

## Documentation

| File | Purpose |
|---|---|
| `README.md` | This file — overview and quick start |
| `CHEATSHEET.md` | Exam-day 60-second playbook, recipes, gotchas |
| `MODIFICATION_GUIDE.md` | Where to look and what to change for each scenario |
| `PROJECT_SUMMARY.md` | Architecture, design decisions, pattern names |

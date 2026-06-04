# Project Summary — Multimodal HCI Assistant

**BSSE23029 Muhammad Mukarram Raza · SE305T HCI Spring 2026**

---

## What was built

An offline-first multimodal assistant that accepts text, voice (Whisper STT), and camera (MediaPipe / OpenCV) input simultaneously, processes them through a configurable staged pipeline, responds using a local LLM (llama3 via Ollama), and generates structured session intelligence reports — all without internet access.

The system is designed as an **offline exam weapon**: swap domains by editing one YAML file, wire the exam's mandated function names in one adapter file, and run.

---

## Architecture overview

The system has three layers:

### Layer 1 — Live state (LIVE provider)

```
core/provider.py   LIVE singleton

LIVE.vision        ← camera thread writes at 15fps
LIVE.audio         ← mic capture + STT writes
LIVE.session       ← SessionState accumulates all turn history
```

Any module imports `LIVE` and reads directly. No data passed as function parameters across call boundaries. The camera thread, gradio handlers, and pipeline stages all see the same state.

### Layer 2 — Ordered stage pipeline

```
config.yaml defines:
  stages:          global definitions (13 stage types, each with fallback chain)
  tabs:            which stages run in which gradio tab, in what order

core/executor.py runs the pipeline:
  for each stage in pipeline_order:
    if disabled → write null+_skipped to ctx, continue
    try → run → write result to ctx
    except → try fallback method → write fallback value

pipeline/stages.py implements each stage type
```

Stages communicate through a shared `TurnContext` dict. Each stage reads what it needs, writes what it produces. Disabled stages write a `_skipped: True` marker so dependent stages can detect and adapt.

### Layer 3 — Personas as full system prompts

```
config.yaml → llm.personas

Each persona is a complete system prompt:
  [SYSTEM CONTEXT]
  Role description.
  {vision_context}   ← filled at call time from LIVE.vision

  TASK: what the LLM must do
  OUTPUT: exact format required
  RULES: constraints and edge cases
```

`persona_system()` in `core/llm.py` reads `LIVE.vision.snapshot()`, fills `{vision_context}`, and returns the complete system prompt. Swapping `active_persona` in config changes the entire LLM behavior.

---

## Data flow for one turn

```
User types "Book a table for 2 at an Italian restaurant"
         ↓
[text_input stage]   → ctx["text_raw"] = "Book a table for 2..."
         ↓
[fuser stage]        → ctx["text"] = "Book a table for 2..."
                        ctx["source"] = "text"
         ↓
[context_inject]     → ctx["vision_context_str"] = "A person visible, mood happy"
                        (from LIVE.vision if camera running)
         ↓
[sentiment stage]    → ctx["sentiment"] = {compound: 0.1, pos:..., neg:...}
                        + ctx["vision_score_delta"] if fusion=score
         ↓
[scale stage]        → ctx["scale"] = {tier: "NEUTRAL", score: 0.1, emoji: "😐"}
                        (reads sentiment compound + adjusts if sentiment._skipped)
         ↓
[classify stage]     → ctx["classify"] = {primary: "BOOKING", all_detected: [...]}
         ↓
[trajectory stage]   → trend analysis from LIVE.session history
         ↓
[respond stage]      → rule lookup: "BOOKING|NEUTRAL" → response string
                        or LLM if no rule matches
         ↓
[log stage]          → LIVE.session updated (turns, wellbeing_log, support_log, ...)
         ↓
gradio handler       → appends to chatbot history, shows meta: "😐 NEUTRAL · BOOKING · turn 1"
```

---

## Key design decisions

### 1. Config-first, code-second

Every behavior that might change on exam day is a config value. Code implements the mechanism; config controls the policy.

- Domain vocabulary → `categories.CATEGORY.keywords`
- Tier thresholds → `scale.tiers` rows
- Response rules → `responses.rules` string keys
- Persona / task type → `llm.personas` + `llm.active_persona`
- Pipeline stages → `tabs.TABNAME.pipeline_order`

### 2. Fallback chain on every stage

Every stage has three failure modes:
```
enabled: false     → write null + _skipped=True → pipeline continues
method fails       → try fallback.method
fallback fails     → inject fallback.value → pipeline continues
```
The pipeline never crashes. The user always gets a response.

### 3. Legacy flags change method, not presence

`sentiment.enabled: false` does **not** skip the sentiment stage. It switches the method from `lexicon` to `llm`. The stage still runs and produces a compound score. This ensures dependent stages (scale reads sentiment compound) always have something to work with.

Only a **tab override** `{enabled: false}` truly skips a stage (writes `_skipped: True`).

### 4. Personas own context; respond passes text

The `respond` prompt template is simply `"{text}"`. The persona defines role, task, output format, and constraints. This means:
- Changing the LLM task = change the persona in config
- The respond stage is just the delivery mechanism
- No `{tier}` or `{category}` injected into the prompt when those features are disabled

### 5. Vision to LLM via context injection

llama3 is text-only. Vision reaches it through `context_inject` stage:
```
LIVE.vision.snapshot() → vision_context_string() → prepended to user text
```
Self-disabling: if camera is off or frame is stale, the context string is empty — no noise injected. The stage checks `LIVE.vision.last_updated` timestamp.

---

## Module responsibilities

| Module | Single responsibility |
|---|---|
| `core/conf.py` | Load config (YAML → JSON → hardcoded defaults), provide `get()`, `reload()`, `sync_json()` |
| `core/llm.py` | Ollama client: chat, JSON, classify, score, extract, summarize + `persona_system()` |
| `core/state.py` | Accumulate turn history: turns, wellbeing_log, support_log, transition_log, responses |
| `core/provider.py` | LIVE singleton: vision frame, audio buffer, session reference |
| `core/executor.py` | Run ordered stage list, handle enabled/disabled/fallback, write to TurnContext |
| `pipeline/stages.py` | 13 stage implementations (one function per stage type) + STAGE_REGISTRY |
| `pipeline/turn.py` | `run_turn()` public API + `process_turn()` legacy compatibility wrapper |
| `pipeline/replay.py` | `run_replay_from_config()` — offline log → full pipeline → report |
| `text/sentiment.py` | Lexicon scorer → compound float in [-1, 1] (VADER-free, no nltk) |
| `text/scale.py` | Compound score → tier dict; alert on at-risk tiers |
| `text/classify.py` | Keyword multi-label → {primary, all_detected, scores} |
| `text/trajectory.py` | Session trend, transition detection, escalation flags, frequency rank |
| `text/respond.py` | Rule lookup (CATEGORY\|TIER → CATEGORY\|* → CATEGORY) → LLM fallback |
| `text/report.py` | Risk formula + banner report + LLM narrative |
| `voice/capture.py` | Mic → float32 array with countdown, PyAudio fallback |
| `voice/stt.py` | Whisper STT with cached model; SpeechRecognition path as backup |
| `voice/audio_io.py` | Format conversion: int16/float32, stereo/mono, any SR → float32 16kHz |
| `vision/backend.py` | Auto-select: mediapipe > yunet+dnn > lbp/opencv; capability checks |
| `vision/faces.py` | Three-backend face detection, EAR blink, drowsy, geometric mood, head pose |
| `vision/hands.py` | Three-backend gesture: MediaPipe 21-landmark, OpenCV skin-mask fallback |
| `vision/lips.py` | MAR features → vowel (simple threshold OR fuzzy Gaussian), LipAnalyser |
| `vision/smoothing.py` | EMA, KalmanScalar, MajorityBuffer, HoldTimer (reference guide params) |
| `vision/draw.py` | Elegant overlays: status bar, face mesh, hand mesh, chips, EAR gauge |
| `vision/bridge.py` | Frame → label dict → human-readable context string for LLM |
| `vision/camera.py` | VideoSource, run_loop, draw_text, distance |
| `vision/color_motion.py` | HSV colour blob tracking, frame-diff / MOG2 motion detection |
| `app.py` | Gradio UI: config-driven tabs, camera worker thread, event handlers |
| `exam_adapter.py` | Exam-mandated function name wrappers; task switcher shortcuts |

---

## Vision backend priority

```
mediapipe-silicon (Apple) or mediapipe (Intel/Linux)
    → FaceMesh 478 landmarks: EAR blink, geometric mood, head pose
    → Hands 21 landmarks: accurate finger count, thumb handedness-aware
    → Special gestures: Thumbs Up, High Five

YuNet (ONNX, 227KB) + FER+ (ONNX, 33MB)
    → Neural face detection: works at angles, low light
    → 7-class emotion: happy/sad/angry/surprised/disgust/fear/contempt

LBP cascades (always available, ships in opencv-python)
    → 3-5× faster than Haar, similar accuracy
    → Eye cascade for blink (presence detection)
    → Smile cascade for mood (binary)
```

Backend is selected automatically at startup and cached. Override: `vision.backend: "opencv"`.

---

## Pattern names

| What we built | Architecture / Design Pattern |
|---|---|
| LIVE singleton | Service Locator / Application Context |
| Ordered stage list + fallbacks | Pipeline/Filter + Chain of Responsibility |
| `enabled: true/false` per stage | Feature Toggles |
| Tab = pipeline_order + overrides | Template Method |
| YAML drives structure and behavior | Configuration-Driven / Declarative Architecture |
| Persona as full system prompt | Strategy Pattern for LLM behavior |
| exam_adapter.py | Facade Pattern |
| Camera thread → LIVE → pipeline | Observer / Publish-Subscribe |
| Taken together | Hybrid Compound AI System (Berkeley, 2024) |

The system is a **Hybrid Deterministic-Neural Pipeline**: deterministic stages for graded, exact-output requirements; LLM-native architecture for conversational and generative requirements; unified by a global state provider and declarative configuration.

---

## What "pure LLM mode" looks like

With all five text stages disabled:
```yaml
sentiment.enabled:   false
scale.enabled:       false
categories.enabled:  false
responses.enabled:   false
scoring.enabled:     false
```

Every turn is: `persona_system() + {text}` → `llama3` → response. No keyword matching, no lexicon scoring, no rule table, no formula. The pipeline still runs but every stage is a pass-through that writes its fallback value. The `log` stage still captures history. The report still generates (using LLM to estimate risk).

This is the architecture the system converges toward as domain complexity increases. The deterministic components exist because the exam requires predictable, auto-gradable output — not because they are architecturally necessary.

---

## Files not to edit (and why)

| File | Edit only if… |
|---|---|
| `core/executor.py` | adding a new stage type with new fallback behavior |
| `core/provider.py` | adding a new live data source (e.g. biometric sensor) |
| `core/llm.py` | adding a new Ollama API capability |
| `core/state.py` | adding a new per-turn data field |
| `text/*.py` | implementing a new algorithm (new sentiment method, new trajectory calculation) |
| `voice/*.py` | supporting a new STT backend |
| `vision/backend.py` | adding a new vision backend (e.g. TensorFlow Lite) |

Everything else should be achievable through `config.yaml` + `exam_adapter.py`.

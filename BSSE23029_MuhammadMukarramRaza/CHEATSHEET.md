# 🗡️ EXAM CHEATSHEET

> **Goal:** professor describes a system → you assemble it from pre-built functions + config.yaml in minutes.
> You almost never write new logic.

---

## ⏱️ 60-second exam playbook

```
1. ollama serve          (leave open in a second terminal)
2. Read the paper        identify: theme, exact function names, dict keys, formula
3. config.yaml           paste STUDENT_LOG into replay.log | swap active_persona | edit categories/responses
4. exam_adapter.py       add 2-line wrappers for every mandated function name
5. python app.py         confirm it runs, tabs work, Ollama is green
6. Run the asserts       paste the paper's assert snippets into the REPL
```

---

## ✅ Pre-exam checklist (do at HOME)

```bash
# packages
uv pip install -r requirements.txt
# Apple Silicon: uv pip install "mediapipe-silicon>=0.9.2.1"

# models
ollama pull llama3
python -c "import whisper; whisper.load_model('base')"
python -c "import whisper; whisper.load_model('small')"   # if past paper asked for small

# verify
python -c "import gradio, cv2, whisper, speech_recognition, openai, PIL, pydub, numpy, mediapipe, requests; print('OK')"
ollama run llama3 "Say: ready"
python -m pipeline.replay          # should print a full sample report
python exam_adapter.py             # should print adapter self-check
```

---

## 🔁 Exact-name mapping — past papers → our library

The exam mandates exact names. **Never rename library functions.** Add 2-line wrappers in `exam_adapter.py`.

| Mandated name | Our function | Returns |
|---|---|---|
| `nexus_capture_audio()` | `voice.capture.capture_audio` | `float32 array` |
| `nexus_transcribe()` | `voice.stt.transcribe` | `{text, language, confidence}` |
| `nexus_get_input(turn)` | `exam_adapter.nexus_get_input` | `{text, source, turn, word_count}` |
| `assess_wellbeing(text)` | `text.scale.assess` | `{tier, score, emoji, is_at_risk}` |
| `check_and_alert(wb, turn)` | `text.scale.check_and_alert` | `bool` |
| `compute_trajectory(log)` | `text.trajectory.compute_trajectory` | `{trend, lowest_tier, at_risk_turns}` |
| `classify_support_need(text)` | `text.classify.classify` | `{primary, all_detected, scores}` |
| `log_support_transition(log, new, turn)` | `exam_adapter.log_support_transition` | event dict |
| `nexus_respond(text, cat, tier)` | `text.respond.respond` | `str` |
| `generate_intelligence_report(…)` | `exam_adapter.generate_intelligence_report` | `str` |
| blink detection | `vision.faces.run_blink` | `int` (count) |
| gesture recognition | `vision.hands.run_gesture` | live window |
| combined face analysis | `vision.faces.run_all_face` | live window |
| lip / vowel reading | `vision.lips.run_lips` | live window |
| NER extraction | `exam_adapter.run_ner(text)` | `{person, location, date, time}` |
| translation | `exam_adapter.run_translation(text, src, tgt)` | `str` |
| question answering | `exam_adapter.run_qa(question, context)` | `str` |

**Pattern for any new name:**
```python
# in exam_adapter.py — this is all you write
def whatever_the_exam_calls_it(text):
    return our_function(text)
```

---

## 🎯 Task switchers (exam_adapter.py)

One line switches the entire LLM persona + disables irrelevant pipeline stages:

```python
from exam_adapter import *

use_ner()            # NER — disables: sentiment, scale, categories, scoring
use_translator()     # Translation — same disabled sections
use_qa()             # Question Answering — disables: sentiment, scale, categories
use_classifier()     # Text Classification — disables: categories
use_intent_detector()# Intent Detection — disables: categories
use_generator()      # Text Generation — disables ALL deterministic stages
restore_default()    # Wellbeing assistant — all sections back ON
```

---

## 📋 Copy-paste recipes

### Record + transcribe
```python
from voice.capture import capture_audio
from voice.stt import transcribe
audio = capture_audio(seconds=7)
result = transcribe(audio)          # {text, language, confidence}
```

### Full text pipeline (one turn)
```python
from pipeline.turn import run_turn
from core.provider import LIVE
from core.state import SessionState
LIVE.init_session(SessionState())

ctx = run_turn("chat", text="I failed my exam and can't sleep")
print(ctx["tier"])       # DISTRESSED
print(ctx["category"])   # ACADEMIC
print(ctx["response"])   # the reply
```

### Offline replay (the surprise stage)
```python
from pipeline.replay import run_replay, run_replay_from_config
# option A: pass a list directly
run_replay(["Hi I need help", "I failed my exam", ...])
# option B: use config.yaml replay.log (paste exam log there, no code change)
run_replay_from_config()
```

### Sentiment + tier
```python
from text.scale import assess, check_and_alert
wb = assess("I feel completely hopeless")
# {'tier':'CRISIS','score':-0.8,'emoji':'🆘','is_at_risk':True}
check_and_alert(wb, turn_number=3)   # prints alert if at-risk
```

### Multi-label classify
```python
from text.classify import classify
classify("I failed my exam and can't pay rent")
# {'primary':'ACADEMIC','all_detected':['ACADEMIC','FINANCIAL'],'scores':{...}}
```

### Ask llama3 directly
```python
from core import llm
llm.chat("Explain recursion in one sentence.")
llm.complete_json('Return {"mood":"happy|sad","score":0-10} for: I love this')
llm.classify("my portal won't load", ["ACADEMIC","TECHNICAL","FINANCIAL"])
llm.score("I am terrified", lo=0, hi=10, criterion="anxiety level")
llm.extract("Call Ali at 03xx on Monday", fields=["name","phone","day"])
```

### NER extraction
```python
from exam_adapter import run_ner
run_ner("Ali met Sara in Lahore on Monday at 3pm")
# {'person':['Ali','Sara'],'location':['Lahore'],'date':['Monday'],'time':['3pm']}
```

### Translation
```python
from exam_adapter import run_translation
run_translation("How are you?", source_lang="English", target_lang="Spanish")
# "¿Cómo estás?"
```

### Question answering
```python
from exam_adapter import run_qa
run_qa(
    question="What is HCI?",
    context="Human-Computer Interaction (HCI) is the study of how people interact with computers."
)
```

### Risk report
```python
from text.report import generate_report
from core.provider import LIVE
generate_report(LIVE.session)    # prints + returns banner string
```

### Vision standalone windows
```bash
python -c "from vision.faces import run_blink;     run_blink()"
python -c "from vision.faces import run_drowsy;    run_drowsy()"
python -c "from vision.faces import run_mood;      run_mood()"
python -c "from vision.faces import run_head_pose; run_head_pose()"
python -c "from vision.faces import run_all_face;  run_all_face()"
python -c "from vision.hands import run_gesture;   run_gesture()"
python -c "from vision.lips  import run_lips;      run_lips()"
python -c "from vision.color_motion import run_color;  run_color('red')"
python -c "from vision.color_motion import run_motion; run_motion()"
```

---

## 🎛️ Re-theme in config.yaml — no code needed

### Swap domain (wellbeing → flight booking)
```yaml
active_persona: "flight_assistant"   # → new persona in personas section
categories:
  BOOKING:  {keywords: [book, reserve, ticket, flight, ...], response_priority: 3}
  INQUIRE:  {keywords: [price, schedule, when, route, ...], response_priority: 2}
responses:
  rules:
    "BOOKING|STRESSED":  "Let me get you sorted. What's your destination?"
    "INQUIRE|*":         "Happy to help. What info do you need?"
```

### Change the scale (6-tier → 3-tier)
```yaml
scale:
  tiers:
    - ["POSITIVE",  0.20,  100.0, "😊"]
    - ["NEUTRAL",  -0.20,   0.20, "😐"]
    - ["NEGATIVE", -100.0, -0.20, "😟"]
```

### Add a new response rule
```yaml
responses:
  rules:
    "NEWCATEGORY|TIER":  "Your canned reply here."
    "NEWCATEGORY|*":     "Wildcard tier reply."
    "NEWCATEGORY":       "Category-only (no tier needed)."
```

### Paste the exam's STUDENT_LOG
```yaml
replay:
  log:
    - "First student message"
    - "Second student message"
    - "..."
```
Then: `python -c "from pipeline.replay import run_replay_from_config; run_replay_from_config()"`

### Quick reference table

| Want to change | Edit this |
|---|---|
| Active persona | `llm.active_persona` |
| Add/edit persona | `llm.personas` block |
| Category keywords | `categories.CATEGORY.keywords` |
| Add new category | New block + new rules in `responses.rules` |
| Disable pipeline section | `sentiment.enabled: false` / `scale.enabled: false` / etc. |
| Response rules | `responses.rules` (`CATEGORY\|TIER`, `CATEGORY\|*`, `CATEGORY`) |
| Risk formula weights | `scoring.tier_weights`, `scoring.wellbeing_weight` |
| Action thresholds | `scoring.thresholds` |
| Scale tiers | `scale.tiers` (remove rows to simplify) |
| Replay log | `replay.log` |
| Session length | `session.max_turns` |
| Whisper size | `whisper.model_size` (must be pre-cached) |
| Vision backend | `vision.backend` ("auto"\|"mediapipe"\|"dnn"\|"opencv") |
| Lip/vowel on/off | `vision.lip.enabled` |
| Head pose thresholds | `vision.head_zones.up_ratio`, `vision.head_zones.down_ratio` |
| EAR blink threshold | `vision.blink.ear_closed_threshold` (0.25 per reference guide) |
| Gesture hold time | `vision.gesture.hold_seconds` |
| HUD layout | `vision.hud.layout` (add/remove/move items between corners) |

---

## ⚠️ Gotchas that cost marks

| Gotcha | Fix |
|---|---|
| Ollama not running | `ollama serve` in a separate terminal, leave it open |
| First llama3 call is slow (5-10s) | `llm.warm_up()` fires one 1-token call at startup |
| Whisper `small` fails offline | pre-cache: `python -c "import whisper; whisper.load_model('small')"` |
| Gradio gives RGB, OpenCV wants BGR | `cv2.cvtColor(img, cv2.COLOR_RGB2BGR)` |
| Audio from gradio mic needs conversion | `from voice.audio_io import from_gradio; data, sr = from_gradio(audio)` |
| `recognize_google` needs internet | only ever use `recognize_whisper` (offline) |
| YAML `no`/`yes`/`on`/`off` → booleans | quote them: `- "no"` |
| config.json stale after editing YAML | `python -c "from core.conf import sync_json; sync_json()"` |
| category `GENERAL` on vague turns | expected — a GENERAL category exists with generic keywords |
| `use_ner()` then no restore | always call `restore_default()` after task-specific functions |

---

## 🗂️ ON/OFF switches (section-level)

| Key | `true` | `false` |
|---|---|---|
| `sentiment.enabled` | lexicon scores text | LLM scores text |
| `scale.enabled` | score → tier | LLM names the state |
| `categories.enabled` | keyword matching | LLM classifies |
| `responses.enabled` | rule table consulted | every response from LLM |
| `scoring.enabled` | formula computes risk | LLM estimates risk |
| `vision.lip.enabled` | vowel/lip detection | skip lip analysis |
| All five text off | — | **pure LLM pipeline** |

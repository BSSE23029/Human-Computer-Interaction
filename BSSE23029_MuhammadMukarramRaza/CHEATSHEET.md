# 🗡️ EXAM CHEATSHEET — offline playbook

> Goal: the professor describes a system; you **assemble** it from these
> pre-built functions + `config.yaml` in minutes. You rarely write new logic.

---

## ⏱️ 60-second exam playbook

1. **Start the brain:** open a terminal → `ollama serve` (leave it open).
2. **Read the paper.** Identify: theme, the exact **function names** required, the
   **dict keys** required, the **scale/categories/formula**.
3. **Re-skin config.yaml** — edit tiers, `categories`, `responses`, `scoring`. No code.
4. **Wire exact names** in `exam_adapter.py` (2-line wrappers, examples already there).
5. **Run:** `python app.py` (gradio) — or `python exam_adapter.py` for a console check.
6. **Verify** with the paper's assert snippets.

---

## ✅ Pre-exam checklist (do at HOME — no internet on exam day)

```bash
pip install gradio openai opencv-python Pillow openai-whisper SpeechRecognition \
            PyAudio sounddevice soundfile pydub numpy requests python-dotenv
# ffmpeg (system):  winget install ffmpeg   |   mac: brew install ffmpeg
ollama pull llama3
python -c "import whisper; whisper.load_model('base')"      # cache base
python -c "import whisper; whisper.load_model('small')"     # OPTIONAL: a past paper used 'small'
# Final verify:
python -c "import gradio, cv2, whisper, speech_recognition, openai, PIL, pydub, numpy, requests; print('OK')"
python -c "import yaml; print('yaml ok')"                   # comes with gradio
ollama run llama3 "Say: I am ready"
```

Test your **camera** and **mic** before exam day.

---

## 🔁 Exact-name mapping (past papers → our library)

| Mandated name (example) | Our function | Returns |
|---|---|---|
| `nexus_capture_audio()` | `voice.capture.capture_audio` | float32 array |
| `nexus_transcribe()` | `voice.stt.transcribe` | `{text,language,confidence}` |
| `nexus_get_input()` | `exam_adapter.nexus_get_input` | `{text,source,turn,word_count}` |
| `assess_wellbeing()` | `text.scale.assess` | `{tier,score,emoji,is_at_risk}` |
| `compute_trajectory()` | `text.trajectory.compute_trajectory` | `{trend,lowest_tier,at_risk_turns}` |
| `check_and_alert()` | `text.scale.check_and_alert` | bool |
| `classify_support_need()` | `text.classify.classify` | `{primary,all_detected,scores}` |
| `log_support_transition()` | `exam_adapter.log_support_transition` | event dict |
| `nexus_respond()` | `text.respond.respond` | str |
| `generate_intelligence_report()` | `text.report.generate_report` | str |
| blink challenge | `vision.faces.run_blink` | int (count) |
| gesture challenge | `vision.hands.run_gesture` | live window |

**Rule:** never rename library functions. Add a wrapper in `exam_adapter.py`:
```python
def whatever_name_they_want(x):
    return our_function(x)
```

---

## 📋 Copy-paste recipes

**Record + transcribe**
```python
from voice.capture import capture_audio
from voice.stt import transcribe
audio = capture_audio(seconds=7)
print(transcribe(audio))          # {'text':..., 'language':'en', 'confidence':'high'}
```

**Sentiment → tier (no VADER needed)**
```python
from text.scale import assess
assess("I feel completely hopeless")   # {'tier':'CRISIS','score':-0.8,'emoji':'🆘','is_at_risk':True}
```

**Multi-label classify**
```python
from text.classify import classify
classify("I failed my exam and can't pay rent")
# {'primary':'ACADEMIC','all_detected':['ACADEMIC','FINANCIAL'],'scores':{...}}
```

**Ask llama3 (chat / JSON / classify / score)**
```python
from core import llm
llm.chat("Explain recursion in one sentence.")
llm.complete_json('Return {"mood":"happy|sad","score":0-10} for: I love this')
llm.classify("my portal won't load", ["ACADEMIC","TECHNICAL","FINANCIAL"])
llm.score("I am terrified", lo=0, hi=10, criterion="anxiety")
```

**Full offline replay + report (the surprise stage)**
```python
from pipeline.replay import run_replay
run_replay(["msg one", "msg two", ...])              # rule responses
run_replay([...], use_llm_response=True)             # llama3 responses
```

**Live vision windows**
```python
from vision.faces import run_blink, run_mood
from vision.hands import run_gesture
from vision.color_motion import run_color, run_motion
run_blink()        # ESC to exit, returns blink count
run_gesture()
```

---

## ⚠️ Gotchas that cost marks

- **YAML `no`/`yes`/`on`/`off` → booleans.** Always quote them: `- "no"`.
- **Ollama not running** → every LLM call returns an offline message. Run `ollama serve`.
- **First llama3 call is slow** (model loads to RAM). Call `llm.warm_up()` once at startup.
- **Whisper format**: feed float32 mono 16 kHz. From gradio mic use `voice.audio_io.from_gradio()`.
- **gradio gives RGB, OpenCV wants BGR** → `cv2.cvtColor(img, cv2.COLOR_RGB2BGR)`.
- **`recognize_google` needs internet** → only ever use `recognize_whisper` (offline).
- **Whisper `small` must be pre-cached** or `load_model('small')` fails offline.

---

## 🎛️ Re-theme in `config.yaml` (no code)

| Want to change… | Edit this key |
|---|---|
| 6-tier → 3-tier scale | `scale.tiers` |
| Category keywords | `categories` |
| Canned replies | `responses.rules` |
| Risk formula / thresholds | `scoring` |
| Assistant persona | `llm.active_persona` / `llm.personas` |
| Whisper size, record length | `whisper.model_size`, `audio.seconds` |
| Gesture names, skin colour, blink sensitivity | `vision.*` |

After editing, the next `python app.py` picks it up. To hot-reload in a REPL:
`from core.conf import reload; reload()`.

# Modification Guide

> **Where do I look? What do I change?**
> This guide answers every common modification scenario — config-only first, code second.

---

## Decision tree: where does my change live?

```
What do you want to change?
│
├─ LLM behavior / persona / task type      → config.yaml  llm.personas
├─ What the LLM responds with              → config.yaml  responses.rules
├─ Category keywords / domain              → config.yaml  categories
├─ Emotional tiers / scale                 → config.yaml  scale.tiers
├─ Risk formula weights                    → config.yaml  scoring
├─ Vision thresholds (EAR, head pose…)    → config.yaml  vision
├─ Replay log                              → config.yaml  replay.log
├─ Which pipeline stages run               → config.yaml  tabs
│
├─ Exam-mandated function name             → exam_adapter.py
│
├─ Gradio tab layout / new tab             → app.py  build_app()
├─ Camera worker / HUD drawing             → app.py  _vision_worker()
│
├─ New stage type                          → pipeline/stages.py  STAGE_REGISTRY
├─ Stage execution order                   → config.yaml  tabs.TABNAME.pipeline_order
│
├─ New vision detector                     → vision/faces.py or vision/hands.py
│                                            + config.yaml vision section
│
└─ New text analysis capability            → text/ module
                                             + pipeline/stages.py new stage fn
                                             + config.yaml stages section
```

---

## Scenario 1: Change the domain (most common on exam day)

**Wellbeing → flight booking, customer service, HR system, etc.**
All config.yaml, zero code.

### Step 1 — write the persona

```yaml
# config.yaml → llm.personas
flight_assistant: |
  [SYSTEM CONTEXT]
  You are a helpful flight booking assistant.
  {vision_context}

  Reply in 1–3 sentences. Be direct and practical.
  Do not ask more than one question per reply.
```

### Step 2 — replace categories

```yaml
# config.yaml → categories
categories:
  enabled: true
  scoring_method: "count"

  BOOKING:
    escalation_target: false
    response_priority: 3
    keywords: [book, reserve, buy, ticket, flight, seat, travel]

  INQUIRE:
    escalation_target: false
    response_priority: 2
    keywords: [price, when, schedule, route, available, how long]

  CANCEL:
    escalation_target: false
    response_priority: 3
    keywords: [cancel, refund, change, modify, reschedule]

  URGENT:
    escalation_target: true
    response_priority: 5
    keywords: [missed, stranded, urgent, emergency, stuck, help]
```

### Step 3 — replace response rules

```yaml
# config.yaml → responses.rules
"URGENT|*":          "This sounds urgent. I'm escalating this now. Please hold."
"CANCEL|STRESSED":   "I understand. Let me process that cancellation. Booking reference?"
"CANCEL|*":          "I can help you cancel or modify. What's your booking reference?"
"BOOKING|*":         "Happy to help you book. What's your destination and travel date?"
"INQUIRE|*":         "Let me check that for you. What specific information do you need?"
```

### Step 4 — update active_persona

```yaml
active_persona: "flight_assistant"
```

### Step 5 — update sentiment word lists (optional)

```yaml
sentiment:
  positive_words:
    affordable: 2.0
    smooth: 1.5
    on_time: 2.0
    upgraded: 2.0
  negative_words:
    delayed: 2.5
    cancelled: 3.0
    overbooked: 2.5
    missed: 2.8
    expensive: 1.8
```

---

## Scenario 2: Add a new NLP task (NER, translation, QA, etc.)

### Step 1 — add persona to config.yaml

```yaml
# config.yaml → llm.personas
my_new_task: |
  [SYSTEM CONTEXT]
  You are a [describe role].
  {vision_context}

  TASK: [describe what to do]
  OUTPUT: [exact format]
  RULES:
  - [constraint 1]
  - [constraint 2]
```

### Step 2 — add to persona_disable_map

```yaml
# config.yaml → llm.persona_disable_map
my_new_task: {sentiment: true, scale: true, categories: true, scoring: true}
```

### Step 3 — add to exam_adapter.py

```python
# exam_adapter.py

def use_my_task():
    use_persona("my_new_task")

def run_my_task(text: str) -> dict:
    use_my_task()
    from core.llm import complete_json, persona_system
    result = complete_json(text, system=persona_system())
    restore_default()
    return result

# Wrap with exam-mandated name:
def whatever_exam_calls_it(text):
    return run_my_task(text)
```

---

## Scenario 3: Change what appears in the gradio UI

### Hide a tab

```yaml
# config.yaml → tabs
voice:
  enabled: false     # voice tab disappears from UI
```

### Change what features show in the Vision tab live feed

```yaml
# config.yaml → vision.hud.layout
# Add/remove items in any corner list — only those items render
layout:
  top_left:    ["blink", "drowsy"]
  top_right:   ["fps", "tier"]
  bottom_left: ["gesture"]
  bottom_right:["mood"]
# to hide FPS: remove "fps" from top_right
# to move gesture: move "gesture" to top_left
```

### Disable a vision detector

```yaml
vision:
  gesture: {enabled: false}    # gesture detection + hand mesh stop running
  lip:     {enabled: false}    # vowel/lip analysis stops running
  drowsy:  {enabled: false}    # drowsiness alert stops
```

### Change tab pipeline (which stages run in which tab)

```yaml
# config.yaml → tabs.chat.pipeline_order
# remove a stage to skip it entirely; reorder to change sequence
chat:
  pipeline_order:
    - text_input
    - fuser
    - context_inject
    - sentiment           # ← remove this line to skip sentiment entirely
    - scale
    - classify
    - respond
    - display
    - log
```

---

## Scenario 4: Add a new gradio tab

### Step 1 — add to config.yaml

```yaml
# config.yaml → tabs
my_tab:
  enabled: true
  label: "🔧 My Tab"
  pipeline_order:
    - text_input
    - fuser
    - respond
    - display
    - log
  overrides:
    stt:    {enabled: false}
    tts:    {enabled: false}
```

### Step 2 — add handler to app.py

```python
# app.py — add before build_app()

def my_tab_handler(message, history):
    ctx = run_turn("my_tab", text=message)
    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": ctx.get("response", "") + _make_meta(ctx)},
    ]
    return history, ""
```

### Step 3 — add rendering to build_app()

```python
# app.py → build_app() → inside with gr.Blocks()

if _tab_enabled("my_tab"):
    with gr.Tab(get("tabs.my_tab.label", "My Tab")):
        with gr.Row():
            msg = gr.Textbox(placeholder="Type...", scale=8, show_label=False)
        msg.submit(my_tab_handler, [msg, chatbot], [chatbot, msg])
```

---

## Scenario 5: Add a new pipeline stage type

### Step 1 — implement the stage function in pipeline/stages.py

```python
# pipeline/stages.py

def run_my_stage(cfg: dict, ctx: dict) -> dict:
    """
    Reads from ctx, does something, writes result back.
    Always return a dict. Never raise — catch exceptions.
    """
    text = ctx.get("text", "")
    try:
        result = do_something(text)
        return {"my_field": result}
    except Exception as e:
        return {"my_field": None, "_error": str(e)}

# Add to the registry at the bottom of stages.py:
STAGE_REGISTRY = {
    ...
    "my_type": run_my_stage,
}
```

### Step 2 — add to config.yaml stages list

```yaml
# config.yaml → stages (global definitions)
  - name: my_stage
    type: my_type
    enabled: true
    method: default
    fallback:
      on: failure
      value: {my_field: null, _skipped: true}
```

### Step 3 — add to any tab's pipeline_order

```yaml
tabs:
  chat:
    pipeline_order:
      - text_input
      - fuser
      - my_stage      # ← add here at the position you need
      - sentiment
      - ...
```

---

## Scenario 6: Add a new vision detector

### Step 1 — implement in the right vision file

```python
# vision/faces.py (for face-based) or vision/hands.py (for hand-based)

def my_detector(frame) -> dict:
    """Returns a label dict."""
    try:
        # your OpenCV / MediaPipe logic
        return {"my_label": value}
    except Exception:
        return {"my_label": None}
```

### Step 2 — add config section

```yaml
# config.yaml → vision
my_detector:
  enabled: true
  threshold: 0.5     # whatever params your detector needs
```

### Step 3 — call it in the camera worker

```python
# app.py → _vision_worker() → inside the detection block

if get("vision.my_detector.enabled", False):
    _det["my_label"] = my_detector(frame).get("my_label")
```

### Step 4 — add to HUD status bar and LIVE label

```python
# app.py → _vision_worker() → slots list
if _det.get("my_label"):
    slots.append((f"My: {_det['my_label']}", _CYAN))

# app.py → label dict
label = {
    ...
    "my_label": _det.get("my_label"),
}
```

---

## Scenario 7: Debug a broken turn

### Check what the pipeline actually produced

```python
from core.provider import LIVE
from core.state import SessionState
from pipeline.turn import run_turn
LIVE.init_session(SessionState())

ctx = run_turn("chat", text="your test message")
print("stt:       ", ctx.get("stt"))
print("text:      ", ctx.get("text"))
print("sentiment: ", ctx.get("sentiment"))
print("scale:     ", ctx.get("scale"))
print("classify:  ", ctx.get("classify"))
print("response:  ", ctx.get("response"))
print("errors:    ", ctx.get("errors"))   # ← check this first
```

### Check if a stage was skipped

```python
ctx.get("sentiment", {}).get("_skipped")   # True = stage was disabled/hard-skipped
ctx.get("classify",  {}).get("_skipped")
```

### Check which backend is running

```python
from vision.backend import BACKEND, print_backend_summary
print_backend_summary()
```

### Hot-reload config without restarting

```python
from core.conf import reload
reload()    # all modules see the new values immediately
```

---

## Where NOT to edit

| File | Why not edit directly |
|---|---|
| `core/executor.py` | stage execution logic — only change if adding a new stage type |
| `core/provider.py` | LIVE singleton — only change if adding a new live data source |
| `text/*.py` | pipeline stage implementations — only change for new algorithms |
| `voice/*.py` | audio pipeline — only change for new STT backends |
| `vision/backend.py` | backend auto-detection — only change to add a new backend |

If you find yourself editing these files for something that "should just be config" — it should be config. Add the knob to config.yaml first.

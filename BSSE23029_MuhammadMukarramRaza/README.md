# BSSE23029 — Multimodal HCI Assistant (offline)

Text + Voice + Vision assistant on the official exam stack:
**llama3 (Ollama) + Whisper + OpenCV + gradio**. No internet required.

## Run
```bash
ollama serve            # terminal 1 (leave open)
python app.py           # terminal 2  -> opens gradio on http://localhost:7860
```

## Layout
```
app.py            gradio entry point (4 tabs: Chat / Voice / Vision / Report)
config.yaml       ALL tunables (edit this to re-theme; config.json = fallback)
exam_adapter.py   exam-day wrappers (exact mandated function names go here)
CHEATSHEET.md     the exam playbook — READ THIS FIRST
core/   llm (Ollama), conf (config), state (session)
text/   sentiment, scale, classify, trajectory, respond, report
voice/  capture, stt (whisper), audio_io (format plumbing)
vision/ camera, faces (Haar), hands (gestures), color_motion, bridge
pipeline/ turn (multimodal fusion), replay (offline log -> report)
```

## Quick checks (no Ollama/mic/cam needed)
```bash
python -c "from core.conf import get; print(get('llm.model'))"
python -m pipeline.replay          # full text pipeline + report on a sample log
python exam_adapter.py             # adapter wiring self-test
```

See **CHEATSHEET.md** for the 60-second playbook, exact-name mapping, recipes,
and gotchas.

"""
app.py -- the gradio entry point.  Run:  python app.py
(Make sure Ollama is running first:  ollama serve)

Four tabs wire the three modalities into one assistant:
  Chat   : textbox  -> text engine + llama3 (with memory)
  Voice  : mic      -> Whisper STT -> same pipeline
  Vision : webcam   -> OpenCV label -> context for the engine
  Report : button   -> session intelligence report

This file stays THIN: the real work lives in core/, text/, voice/, vision/,
pipeline/. Swap behaviour by editing config.yaml, not this file.
"""
import gradio as gr

from core.conf import get
from core.state import SessionState
from core import llm
from pipeline.turn import process_turn
from text.report import generate_report

# One shared session for the demo (single user during the exam).
SESSION = SessionState()


def ollama_status() -> str:
    return ("🟢 **Ollama connected** — llama3 ready"
            if llm.is_alive() else
            "🔴 **Ollama NOT running** — open a terminal and run `ollama serve`")


# ---- Chat tab -------------------------------------------------------------
def chat_fn(message, history):
    if not message:
        return history, ""
    r = process_turn(SESSION, text=message, use_llm_response=llm.is_alive())
    wb, sup = r["wellbeing"], r["support"]
    meta = f"\n\n_{wb['emoji']} {wb['tier']} · {sup['primary']} · turn {r['turn']}_"
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": r["response"] + meta},
    ]
    return history, ""


# ---- Voice tab ------------------------------------------------------------
def voice_fn(audio, history):
    if audio is None:
        return history, "🎙️ Record something first."
    from voice.audio_io import from_gradio
    data, sr = from_gradio(audio)
    r = process_turn(SESSION, audio=data, use_llm_response=llm.is_alive())
    wb, sup = r["wellbeing"], r["support"]
    meta = f"\n\n_{wb['emoji']} {wb['tier']} · {sup['primary']} · turn {r['turn']} (voice)_"
    history = history + [
        {"role": "user", "content": f"🎙️ {r['text'] or '(empty)'}"},
        {"role": "assistant", "content": r["response"] + meta},
    ]
    return history, f"Heard: \"{r['text']}\""


# ---- Vision tab -----------------------------------------------------------
def vision_fn(image):
    if image is None:
        return "📷 Capture a frame first."
    import cv2
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)   # gradio gives RGB; OpenCV wants BGR
    from vision.bridge import frame_to_label, vision_context_string
    label = frame_to_label(frame)
    SESSION.log_vision(label)
    return f"**{vision_context_string(label)}**\n\n```\n{label}\n```"


# ---- Report tab -----------------------------------------------------------
def report_fn():
    if SESSION.n_turns == 0:
        return "No turns yet — chat or speak first."
    return "```\n" + generate_report(SESSION, do_print=False) + "\n```"


def reset_fn():
    SESSION.reset()
    return [], "Session reset.", "Session reset."


def build_app():
    with gr.Blocks(title=get("ui.app_title", "HCI Assistant")) as demo:
        gr.Markdown(f"# {get('ui.app_title')}\n{get('ui.app_subtitle','')}")
        status = gr.Markdown(ollama_status())

        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(height=380)
            with gr.Row():
                msg = gr.Textbox(placeholder="Type a message and press Enter...",
                                 scale=8, show_label=False)
                send = gr.Button("Send", scale=1, variant="primary")
            msg.submit(chat_fn, [msg, chatbot], [chatbot, msg])
            send.click(chat_fn, [msg, chatbot], [chatbot, msg])

        with gr.Tab("🎙️ Voice"):
            voice_chat = gr.Chatbot(height=300)
            mic = gr.Audio(sources=["microphone"], type="numpy", label="Speak")
            transcript = gr.Markdown()
            gr.Button("Transcribe + Send", variant="primary").click(
                voice_fn, [mic, voice_chat], [voice_chat, transcript])

        with gr.Tab("📷 Vision"):
            gr.Markdown("Capture a webcam frame; OpenCV summarises it for the engine.")
            cam = gr.Image(sources=["webcam"], type="numpy", label="Webcam")
            vis_out = gr.Markdown()
            gr.Button("Analyse frame", variant="primary").click(vision_fn, cam, vis_out)
            gr.Markdown("_For live blink/gesture windows run e.g._ `python -c "
                        "\"from vision.faces import run_blink; run_blink()\"`")

        with gr.Tab("📋 Report"):
            report_box = gr.Markdown()
            with gr.Row():
                gr.Button("Generate report", variant="primary").click(report_fn, None, report_box)
                gr.Button("Reset session").click(reset_fn, None, [chatbot, transcript, report_box])

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print(" ", get("ui.app_title", "HCI Assistant"))
    print(" ", ollama_status().replace("**", ""))
    print("  Whisper model:", get("whisper.model_size"))
    print("=" * 60)
    build_app().launch(server_port=get("ui.server_port", 7860))

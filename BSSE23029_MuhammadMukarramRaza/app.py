"""
app.py -- gradio entry point.   Run:  python app.py
(Make sure Ollama is running first:  ollama serve)

Four tabs:
  💬 Chat   → textbox → full text engine + llama3 (with session memory)
  🎙 Voice  → mic → Whisper STT → same pipeline
  📷 Vision → live webcam stream with all OpenCV overlays drawn on-frame
              + per-turn label dict fed to the LLM as context
  📋 Report → risk score + trajectory + LLM narrative + replay runner

This file stays THIN. All logic lives in core/, text/, voice/, vision/, pipeline/.
Change behaviour by editing config.yaml, not this file.
"""

import threading
import time

import cv2
import gradio as gr
import numpy as np

from core.conf import get
from core.state import SessionState
from core import llm
from pipeline.turn import process_turn
from text.report import generate_report

# ── shared session (single user during the exam) ──────────────────────────────
SESSION = SessionState()

# ── vision state shared between the generator and the pipeline ───────────────
_vision_state = {
    "running":   False,
    "latest_frame": None,    # latest annotated BGR frame
    "latest_label": None,    # latest label dict from bridge.py
    "lock": threading.Lock(),
}


# ── helpers ───────────────────────────────────────────────────────────────────
def _ollama_status() -> str:
    return ("🟢 **Ollama connected** — llama3 ready"
            if llm.is_alive() else
            "🔴 **Ollama NOT running** — open a terminal and run `ollama serve`")


def _session_info() -> str:
    s = SESSION
    mx = int(get("session.max_turns", 0))
    limit = f" / {mx}" if mx > 0 else ""
    return (f"Turn {s.n_turns}{limit} · "
            f"voice {s.voice_turns} · text {s.text_turns} · "
            f"risk {"??" if s.n_turns == 0 else "--"}")


def _bgr_to_pil(frame):
    """Convert OpenCV BGR frame to PIL/numpy RGB for gradio."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ── vision background thread ──────────────────────────────────────────────────
def _vision_worker():
    """Run the webcam loop in a background thread.
    Draws all HUD overlays on the frame and stores the latest annotated frame
    for the gradio streaming generator to yield.
    """
    from vision.faces import detect_faces, is_smiling, head_zone, _gray
    from vision.hands import count_fingers, classify_gesture
    from vision.bridge import frame_to_label

    idx   = int(get("vision.webcam_index", 0))
    flip  = bool(get("vision.flip_webcam", True))
    w     = int(get("vision.display.width", 640))
    h     = int(get("vision.display.height", 480))
    cfg_h = get("vision.hud") or {}
    layout   = get("vision.hud.layout") or {}
    f_scale  = float(cfg_h.get("font_scale", 0.7))
    thick    = int(cfg_h.get("thickness", 2))
    col_def  = tuple(cfg_h.get("color_default", [0, 255, 0]))
    col_warn = tuple(cfg_h.get("color_warning", [0, 0, 255]))
    col_info = tuple(cfg_h.get("color_info",    [255, 255, 0]))

    # stability buffers
    stab   = get("vision.stability") or {}
    g_hold = int(stab.get("gesture_hold_frames", 5))
    m_hist = int(stab.get("mood_history_frames", 15))
    b_dbnc = int(stab.get("blink_debounce_frames", 3))

    gest_buf  = []
    mood_buf  = []
    blink_state = {"blinks": 0, "closed": False, "closed_frames": 0}
    drowsy_frames = 0
    fps_val = 0.0
    frames_counted = 0
    t0 = time.time()

    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"[vision] could not open webcam {idx}")
        return

    try:
        while _vision_state["running"]:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            if flip:
                frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (w, h))

            # ── run detectors ────────────────────────────────────
            gray      = _gray(frame)
            face_boxes = detect_faces(gray) if get("vision.face_detect.enabled", True) else []
            present   = len(face_boxes) > 0

            # blink
            blink_count = 0
            if get("vision.blink.enabled", True) and present:
                x, y, fw_, fh_ = face_boxes[0]
                from vision.faces import detect_eyes
                eyes = detect_eyes(gray[y:y + fh_//2, x:x + fw_])
                need  = int(get("vision.blink.eyes_closed_frames", 2))
                if not eyes:
                    blink_state["closed_frames"] += 1
                    if blink_state["closed_frames"] >= need:
                        blink_state["closed"] = True
                else:
                    if blink_state["closed"]:
                        blink_state["blinks"] += 1
                    blink_state["closed"] = False
                    blink_state["closed_frames"] = 0
                blink_count = blink_state["blinks"]

            # drowsy
            is_drowsy = False
            if get("vision.drowsy.enabled", True) and present:
                x, y, fw_, fh_ = face_boxes[0]
                from vision.faces import detect_eyes
                eyes = detect_eyes(gray[y:y + fh_//2, x:x + fw_])
                if not eyes:
                    drowsy_frames += 1
                else:
                    drowsy_frames = 0
                is_drowsy = drowsy_frames >= int(get("vision.drowsy.closed_frames_alert", 20))

            # mood (majority vote)
            raw_mood = "no_face"
            if get("vision.smile_mood.enabled", True) and present:
                x, y, fw_, fh_ = face_boxes[0]
                roi = gray[y:y + fh_, x:x + fw_]
                raw_mood = "smiling" if is_smiling(roi) else "neutral"
            mood_buf.append(raw_mood)
            if len(mood_buf) > m_hist:
                mood_buf.pop(0)
            from collections import Counter
            mood = Counter(mood_buf).most_common(1)[0][0] if mood_buf else "no_face"

            # head zone
            zone = ""
            if get("vision.head_pose.enabled", True) and present:
                zone = head_zone(face_boxes[0], frame.shape)

            # gesture (hold buffer)
            raw_gest, gest_emoji = "none", ""
            if get("vision.gesture.enabled", True):
                from vision.hands import count_fingers as cf_, classify_gesture as cg_
                count, _ = cf_(frame)
                g_name, g_emoji = cg_(count)
                gest_buf.append((g_name, g_emoji))
                if len(gest_buf) > g_hold:
                    gest_buf.pop(0)
                if len(gest_buf) == g_hold and len(set(n for n,_ in gest_buf)) == 1:
                    raw_gest, gest_emoji = gest_buf[0]

            # fps
            frames_counted += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                fps_val = frames_counted / elapsed
                frames_counted, t0 = 0, time.time()

            # ── draw HUD ─────────────────────────────────────────
            # current wellbeing tier from session (if available)
            tier_str = ""
            if SESSION.wellbeing_log:
                last_wb = SESSION.wellbeing_log[-1]
                tier_str = f"{last_wb.get('emoji','')} {last_wb.get('tier','')}"

            score_str = ""
            if SESSION.wellbeing_log:
                score_str = f"Score: {SESSION.wellbeing_log[-1].get('score',0):+.2f}"

            element_text = {
                "fps":       f"FPS: {fps_val:.1f}",
                "tier":      tier_str,
                "score":     score_str,
                "turn":      f"Turn: {SESSION.n_turns}",
                "blink":     f"Blinks: {blink_count}",
                "drowsy":    (get("vision.drowsy.alert_message","DROWSY!") if is_drowsy else ""),
                "head_zone": f"Head: {zone}" if zone else "",
                "gesture":   f"{raw_gest} {gest_emoji}".strip() if raw_gest != "none" else "",
                "mood":      f"Mood: {mood}" if mood != "no_face" else "",
            }
            element_color = {
                "fps": col_info, "tier": col_def, "score": col_def,
                "turn": col_info, "blink": col_def,
                "drowsy": col_warn, "head_zone": col_info,
                "gesture": col_def, "mood": col_def,
            }

            def _draw_corner(elements, start_x, start_y, right_align=False):
                y_pos = start_y
                for el in elements:
                    txt = element_text.get(el, "")
                    if not txt:
                        continue
                    col = element_color.get(el, col_def)
                    if right_align:
                        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, f_scale, thick)
                        x_pos = start_x - tw - 4
                    else:
                        x_pos = start_x
                    cv2.rectangle(frame, (x_pos-2, y_pos-16), (x_pos+200, y_pos+4), (0,0,0), -1)
                    cv2.putText(frame, txt, (x_pos, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, f_scale, col, thick, cv2.LINE_AA)
                    y_pos += 28

            _draw_corner(layout.get("top_left", []),    10,    28)
            _draw_corner(layout.get("top_right", []),   w-10,  28,  right_align=True)
            _draw_corner(layout.get("bottom_left", []), 10,    h-80)
            _draw_corner(layout.get("bottom_right",[]), w-10,  h-80, right_align=True)

            # draw face boxes
            for (x, y, fw_, fh_) in face_boxes:
                cv2.rectangle(frame, (x, y), (x+fw_, y+fh_), col_def, 2)

            # store label dict
            label = {
                "present":   present,
                "faces":     len(face_boxes),
                "mood":      mood,
                "head_zone": zone or "Center",
                "gesture":   raw_gest,
                "fingers":   0,
                "blinks":    blink_count,
                "is_drowsy": is_drowsy,
            }

            with _vision_state["lock"]:
                _vision_state["latest_frame"] = frame.copy()
                _vision_state["latest_label"] = label

            # optional cv2 window
            if get("vision.display.show_window", False):
                cv2.imshow("HCI Vision", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            time.sleep(1.0 / max(int(get("vision.display.stream_fps", 15)), 1))

    finally:
        cap.release()
        if get("vision.display.show_window", False):
            cv2.destroyAllWindows()
        _vision_state["running"] = False


def _start_vision():
    if not _vision_state["running"]:
        _vision_state["running"] = True
        t = threading.Thread(target=_vision_worker, daemon=True)
        t.start()


def _stop_vision():
    _vision_state["running"] = False


def _vision_stream():
    """Gradio generator: yield annotated frames as RGB numpy arrays."""
    _start_vision()
    target_fps = max(int(get("vision.display.stream_fps", 15)), 1)
    while True:
        with _vision_state["lock"]:
            frame = _vision_state.get("latest_frame")
        if frame is not None:
            yield _bgr_to_pil(frame)
        time.sleep(1.0 / target_fps)


# ── Chat tab ─────────────────────────────────────────────────────────────────
def chat_fn(message, history):
    if not message:
        return history, "", _session_info()

    max_turns = int(get("session.max_turns", 0))
    if max_turns > 0 and SESSION.n_turns >= max_turns:
        history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": f"Session complete ({max_turns} turns). Click **Reset** to start a new session."},
        ]
        return history, "", _session_info()

    # grab latest vision label if vision is running
    frame = None
    with _vision_state["lock"]:
        label = _vision_state.get("latest_label")
    # we pass the label as text context, not the actual frame
    r = process_turn(SESSION, text=message)

    # inject vision context into the response metadata if available
    vis_note = ""
    if label and label.get("present") and get("vision.bridge.include_in_llm_context", True):
        from vision.bridge import vision_context_string
        vis_note = f"\n_📷 {vision_context_string(label)}_"

    wb, sup = r["wellbeing"], r["support"]
    meta = (f"\n\n_{wb.get('emoji','')} {wb.get('tier','?')} · "
            f"{sup.get('primary','?')} · turn {r['turn']}_" + vis_note)

    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": r["response"] + meta},
    ]
    return history, "", _session_info()


# ── Voice tab ─────────────────────────────────────────────────────────────────
def voice_fn(audio, history):
    if audio is None:
        return history, "🎙️ Record something first.", _session_info()

    max_turns = int(get("session.max_turns", 0))
    if max_turns > 0 and SESSION.n_turns >= max_turns:
        return history, "Session complete. Click Reset.", _session_info()

    from voice.audio_io import from_gradio
    data, sr = from_gradio(audio)

    r = process_turn(SESSION, audio=data)
    wb, sup = r["wellbeing"], r["support"]
    meta = (f"\n\n_{wb.get('emoji','')} {wb.get('tier','?')} · "
            f"{sup.get('primary','?')} · turn {r['turn']} (voice)_")

    history = history + [
        {"role": "user",      "content": f"🎙️ {r['text'] or '(empty)'}"},
        {"role": "assistant", "content": r["response"] + meta},
    ]
    return history, f"Heard: \"{r['text']}\"", _session_info()


# ── Vision tab ────────────────────────────────────────────────────────────────
def vision_label_fn():
    """Return a markdown string showing the latest detected labels."""
    with _vision_state["lock"]:
        label = _vision_state.get("latest_label")
    if label is None:
        return "📷 Camera not started. Click **Start camera**."
    from vision.bridge import vision_context_string
    lines = [
        f"**{vision_context_string(label)}**",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
    ] + [f"| `{k}` | {v} |" for k, v in label.items()]
    return "\n".join(lines)


def send_vision_to_chat_fn(history):
    """Capture the latest frame label and run a pipeline turn with it."""
    with _vision_state["lock"]:
        label = _vision_state.get("latest_label")
    if not label:
        return history, "No camera frame yet.", _session_info()

    from vision.bridge import vision_context_string
    ctx_text = vision_context_string(label)
    r = process_turn(SESSION, text=ctx_text)
    wb, sup = r["wellbeing"], r["support"]
    meta = (f"\n\n_{wb.get('emoji','')} {wb.get('tier','?')} · "
            f"{sup.get('primary','?')} · turn {r['turn']} (vision)_")

    history = history + [
        {"role": "user",      "content": f"📷 {ctx_text}"},
        {"role": "assistant", "content": r["response"] + meta},
    ]
    return history, "Vision frame sent to pipeline.", _session_info()


# ── Report tab ────────────────────────────────────────────────────────────────
def report_fn():
    if SESSION.n_turns == 0:
        return "No turns yet — chat or speak first."
    return "```\n" + generate_report(SESSION, do_print=False) + "\n```"


def replay_fn():
    """Run the config.yaml replay log and return the report string."""
    from pipeline.replay import run_replay_from_config
    state = run_replay_from_config(use_llm_response=llm.is_alive())
    return "```\n" + generate_report(state, do_print=False) + "\n```"


def reset_fn():
    SESSION.reset()
    _stop_vision()
    return [], [], "Session reset. Ollama: " + ("online" if llm.is_alive() else "offline"), _session_info()


# ── build gradio app ──────────────────────────────────────────────────────────
def build_app():
    title     = get("ui.app_title", "HCI Assistant")
    subtitle  = get("ui.app_subtitle", "")
    exam_id   = get("exam_meta.student_id", "")
    exam_name = get("exam_meta.student_name", "")

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        if subtitle:
            gr.Markdown(f"_{subtitle}_")
        if exam_id:
            gr.Markdown(f"**{exam_id}** — {exam_name}")

        status    = gr.Markdown(_ollama_status())
        sess_info = gr.Markdown(_session_info())

        # shared chatbot used by both Chat and Voice tabs
        chatbot = gr.Chatbot(height=380, type="messages")

        with gr.Tab("💬 Chat"):
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message and press Enter...",
                    scale=8, show_label=False)
                send_btn = gr.Button("Send", scale=1, variant="primary")
            msg.submit(chat_fn, [msg, chatbot], [chatbot, msg, sess_info])
            send_btn.click(chat_fn, [msg, chatbot], [chatbot, msg, sess_info])

        with gr.Tab("🎙️ Voice"):
            mic = gr.Audio(sources=["microphone"], type="numpy", label="Speak")
            transcript = gr.Markdown()
            gr.Button("Transcribe + Send", variant="primary").click(
                voice_fn, [mic, chatbot], [chatbot, transcript, sess_info])

        with gr.Tab("📷 Vision"):
            with gr.Row():
                with gr.Column(scale=2):
                    live_feed = gr.Image(
                        label="Live webcam (annotated)",
                        streaming=True,
                        height=int(get("vision.display.height", 480)),
                    )
                with gr.Column(scale=1):
                    vis_labels = gr.Markdown("_Labels appear here_")
                    gr.Button("Refresh labels").click(vision_label_fn, None, vis_labels)
                    gr.Button("Send to pipeline", variant="primary").click(
                        send_vision_to_chat_fn, [chatbot], [chatbot, vis_labels, sess_info])

            with gr.Row():
                gr.Button("▶ Start camera", variant="primary").click(
                    lambda: (gr.update(), _start_vision())[0],
                    None, live_feed
                )
                gr.Button("⏹ Stop camera").click(lambda: _stop_vision(), None, None)

            # wire streaming
            live_feed.stream(_vision_stream, None, live_feed)

            gr.Markdown(
                "_For standalone live windows:_ "
                "`python -c \"from vision.faces import run_blink; run_blink()\"`  \n"
                "`python -c \"from vision.hands import run_gesture; run_gesture()\"`"
            )

        with gr.Tab("📋 Report"):
            report_box = gr.Markdown("_Generate a report or run replay._")
            with gr.Row():
                gr.Button("📊 Generate report", variant="primary").click(
                    report_fn, None, report_box)
                gr.Button("▶ Run config replay").click(
                    replay_fn, None, report_box)
                gr.Button("🔄 Reset session").click(
                    reset_fn, None, [chatbot, chatbot, status, sess_info])

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print(f"  {get('ui.app_title', 'HCI Assistant')}")
    print(f"  {get('exam_meta.student_id','')}  {get('exam_meta.student_name','')}")
    print(f"  {_ollama_status().replace('**','')}")
    print(f"  Whisper: {get('whisper.model_size')}  "
          f"| Max turns: {get('session.max_turns')}  "
          f"| Vision: {get('session.input_modes.vision')}")
    print("=" * 60)

    # pre-warm Ollama (hides the first-call lag from the demo)
    if llm.is_alive():
        threading.Thread(target=llm.warm_up, daemon=True).start()

    build_app().launch(
        server_port=int(get("ui.server_port", 7860)),
        share=bool(get("ui.share", False)),
    )

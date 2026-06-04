"""
app.py -- gradio entry point.   Run:  python app.py
(Make sure Ollama is running first:  ollama serve)

Four tabs:
  💬 Chat   → textbox → full text engine + llama3 (with session memory)
  🎙 Voice  → mic → Whisper STT → same pipeline
  📷 Vision → live server-side webcam stream; all OpenCV HUD overlays drawn on-frame
  📋 Report → risk score + trajectory + LLM narrative + replay runner

Architecture:
  • Vision runs in a background thread; annotated BGR frames are stored in
    _vision_state["latest_frame"].  The gradio streaming generator reads from
    that shared variable and yields RGB numpy arrays.
  • Auto-report is owned by the Chat/Voice handlers (not by process_turn), so
    it fires once and only once at the end of a session.
  • All logic lives in core/, text/, voice/, vision/, pipeline/.
    Change behaviour by editing config.yaml, not this file.
"""

import threading
import time

import gradio as gr
import numpy as np

from core.conf import get
from core.state import SessionState
from core import llm
from pipeline.turn import process_turn
from text.report import generate_report

# ── shared session ────────────────────────────────────────────────────────────
SESSION = SessionState()

# ── vision state shared between background thread and gradio ─────────────────
_vision_state: dict = {
    "running":       False,
    "latest_frame":  None,    # latest annotated BGR numpy frame
    "latest_label":  None,    # latest label dict from bridge.py
    "lock":          threading.Lock(),
}


# ── helpers ───────────────────────────────────────────────────────────────────
def _ollama_status() -> str:
    return ("🟢 **Ollama connected** — llama3 ready"
            if llm.is_alive() else
            "🔴 **Ollama NOT running** — open a terminal → run `ollama serve`")


def _session_info() -> str:
    s   = SESSION
    mx  = int(get("session.max_turns", 0))
    lim = f"/{mx}" if mx > 0 else ""
    return (f"Turn **{s.n_turns}**{lim} · "
            f"voice {s.voice_turns} · text {s.text_turns}")


def _bgr_to_rgb(frame):
    import cv2
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _session_over() -> bool:
    mx = int(get("session.max_turns", 0))
    return mx > 0 and SESSION.n_turns >= mx


# ── vision background thread ──────────────────────────────────────────────────
def _vision_worker():
    """Run the webcam capture + OpenCV detection loop in a daemon thread.
    Draws all HUD overlays on the frame; stores it in _vision_state for the
    gradio streaming generator.
    """
    from vision.faces import detect_faces, is_smiling, head_zone, _gray, detect_eyes
    from vision.hands import count_fingers as _count_fingers, classify_gesture

    idx   = int(get("vision.webcam_index", 0))
    flip  = bool(get("vision.flip_webcam", True))
    w     = int(get("vision.display.width", 640))
    h     = int(get("vision.display.height", 480))
    layout    = get("vision.hud.layout") or {}
    f_scale   = float(get("vision.hud.font_scale", 0.7))
    thick     = int(get("vision.hud.thickness", 2))
    col_def   = tuple(get("vision.hud.color_default",  [0, 255, 0]))
    col_warn  = tuple(get("vision.hud.color_warning",  [0, 0, 255]))
    col_info  = tuple(get("vision.hud.color_info",     [255, 255, 0]))
    g_hold    = int(get("vision.stability.gesture_hold_frames",  5))
    m_hist    = int(get("vision.stability.mood_history_frames",  15))
    target_fps = max(int(get("vision.display.stream_fps", 15)), 1)

    gest_buf, mood_buf         = [], []
    blink_state                = {"blinks": 0, "closed": False, "closed_frames": 0}
    drowsy_frames              = 0
    fps_val, frame_cnt, t0     = 0.0, 0, time.time()

    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"[vision] could not open webcam {idx}")
        _vision_state["running"] = False
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
            gray  = _gray(frame)

            # ── face detect ──────────────────────────────────────
            face_boxes = (detect_faces(gray)
                          if get("vision.face_detect.enabled", True) else [])
            present = len(face_boxes) > 0

            # ── blink ────────────────────────────────────────────
            if get("vision.blink.enabled", True) and present:
                x, y, fw_, fh_ = face_boxes[0]
                eyes = detect_eyes(gray[y:y + fh_//2, x:x + fw_])
                need = int(get("vision.blink.eyes_closed_frames", 2))
                if not eyes:
                    blink_state["closed_frames"] += 1
                    if blink_state["closed_frames"] >= need:
                        blink_state["closed"] = True
                else:
                    if blink_state["closed"]:
                        blink_state["blinks"] += 1
                    blink_state["closed"] = False
                    blink_state["closed_frames"] = 0

            # ── drowsy ───────────────────────────────────────────
            is_drowsy = False
            if get("vision.drowsy.enabled", True) and present:
                x, y, fw_, fh_ = face_boxes[0]
                eyes = detect_eyes(gray[y:y + fh_//2, x:x + fw_])
                drowsy_frames = drowsy_frames + 1 if not eyes else 0
                is_drowsy = (drowsy_frames >= int(get("vision.drowsy.closed_frames_alert", 20)))

            # ── mood (majority vote over history) ─────────────────
            raw_mood = "no_face"
            if get("vision.smile_mood.enabled", True) and present:
                x, y, fw_, fh_ = face_boxes[0]
                roi  = gray[y:y + fh_, x:x + fw_]
                raw_mood = "smiling" if is_smiling(roi) else "neutral"
            mood_buf.append(raw_mood)
            if len(mood_buf) > m_hist:
                mood_buf.pop(0)
            from collections import Counter
            mood = Counter(mood_buf).most_common(1)[0][0] if mood_buf else "no_face"

            # ── head zone ─────────────────────────────────────────
            zone = "Center"
            if get("vision.head_pose.enabled", True) and present:
                zone = head_zone(face_boxes[0], frame.shape)

            # ── gesture (stability hold buffer) ──────────────────
            raw_gest, g_emoji = "none", ""
            if get("vision.gesture.enabled", True):
                count, _ = _count_fingers(frame)
                g_name, g_emoji = classify_gesture(count)
                gest_buf.append((g_name, g_emoji, count > 0))
                if len(gest_buf) > g_hold:
                    gest_buf.pop(0)
                # only commit a gesture when it's been stable for hold_frames
                if len(gest_buf) == g_hold and len({n for n, _, _ in gest_buf}) == 1:
                    raw_gest, g_emoji = gest_buf[0][0], gest_buf[0][1]
                    if not gest_buf[0][2]:   # count was 0 — no hand visible
                        raw_gest = "none"

            # ── fps ──────────────────────────────────────────────
            frame_cnt += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                fps_val      = frame_cnt / elapsed
                frame_cnt, t0 = 0, time.time()

            # ── HUD text map ──────────────────────────────────────
            wb_last = SESSION.wellbeing_log[-1] if SESSION.wellbeing_log else {}
            elem_text = {
                "fps":      f"FPS: {fps_val:.1f}",
                "tier":     f"{wb_last.get('emoji','')} {wb_last.get('tier','')}".strip(),
                "score":    f"Score: {wb_last.get('score',0):+.2f}" if wb_last else "",
                "turn":     f"Turn: {SESSION.n_turns}",
                "blink":    f"Blinks: {blink_state['blinks']}",
                "drowsy":   (get("vision.drowsy.alert_message","DROWSY!") if is_drowsy else ""),
                "head_zone":f"Head: {zone}" if zone and zone != "Center" else "",
                "gesture":  f"{raw_gest} {g_emoji}".strip() if raw_gest != "none" else "",
                "mood":     f"Mood: {mood}" if mood != "no_face" else "",
            }
            elem_col = {
                "fps":      col_info, "tier":  col_def,  "score":    col_def,
                "turn":     col_info, "blink": col_def,  "drowsy":   col_warn,
                "head_zone":col_info, "gesture":col_def, "mood":     col_def,
            }

            def _draw_corner(names, sx, sy, right=False):
                y_pos = sy
                for nm in names:
                    txt = elem_text.get(nm, "")
                    if not txt:
                        continue
                    col = elem_col.get(nm, col_def)
                    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, f_scale, thick)
                    xp = sx - tw - 4 if right else sx
                    cv2.rectangle(frame, (xp - 2, y_pos - 18), (xp + tw + 4, y_pos + 4),
                                  (0, 0, 0), -1)
                    cv2.putText(frame, txt, (xp, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, f_scale, col, thick, cv2.LINE_AA)
                    y_pos += 28

            _draw_corner(layout.get("top_left",    []),  10,    28)
            _draw_corner(layout.get("top_right",   []),  w-10,  28,  right=True)
            _draw_corner(layout.get("bottom_left", []),  10,    h-80)
            _draw_corner(layout.get("bottom_right",[]),  w-10,  h-80, right=True)

            # draw face boxes on frame
            for (x, y, fw_, fh_) in face_boxes:
                cv2.rectangle(frame, (x, y), (x+fw_, y+fh_), col_def, 2)

            # store for gradio and pipeline
            label = {
                "present":   present,
                "faces":     len(face_boxes),
                "mood":      mood,
                "head_zone": zone,
                "gesture":   raw_gest,
                "fingers":   0,
                "blinks":    blink_state["blinks"],
                "is_drowsy": is_drowsy,
            }
            with _vision_state["lock"]:
                _vision_state["latest_frame"] = frame.copy()
                _vision_state["latest_label"] = label

            if get("vision.display.show_window", False):
                cv2.imshow("HCI Vision", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            time.sleep(1.0 / target_fps)

    finally:
        cap.release()
        if get("vision.display.show_window", False):
            cv2.destroyAllWindows()
        _vision_state["running"] = False


def _start_vision():
    if not _vision_state["running"]:
        _vision_state["running"] = True
        threading.Thread(target=_vision_worker, daemon=True).start()
        time.sleep(0.3)   # give the thread a moment to open the camera


def _stop_vision():
    _vision_state["running"] = False


def _vision_stream():
    """Generator that yields annotated RGB frames for the gradio Image output.
    Called by the 'Start camera' button click handler (server-side streaming).
    """
    _start_vision()
    target_fps = max(int(get("vision.display.stream_fps", 15)), 1)
    while _vision_state["running"]:
        with _vision_state["lock"]:
            frame = _vision_state.get("latest_frame")
        if frame is not None:
            yield _bgr_to_rgb(frame)
        time.sleep(1.0 / target_fps)


# ── Chat tab ─────────────────────────────────────────────────────────────────
def chat_fn(message, history):
    if not message:
        return history, "", _session_info()

    if _session_over():
        history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": f"Session complete. Click **Reset** to start a new one."},
        ]
        return history, "", _session_info()

    r = process_turn(SESSION, text=message)

    # vision context note (if camera is running)
    vis_note = ""
    with _vision_state["lock"]:
        label = _vision_state.get("latest_label")
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

    # auto-report when session ends (owned HERE, not in process_turn)
    report_md = ""
    if _session_over() and get("session.auto_report", True):
        report_md = "```\n" + generate_report(SESSION, do_print=True) + "\n```"

    return history, "", report_md or "", _session_info()


# ── Voice tab ─────────────────────────────────────────────────────────────────
def voice_fn(audio, history):
    if audio is None:
        return history, "🎙️ Record something first.", _session_info()

    if _session_over():
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

    report_md = ""
    if _session_over() and get("session.auto_report", True):
        report_md = "```\n" + generate_report(SESSION, do_print=True) + "\n```"

    return history, f"Heard: \"{r['text']}\"" + ("\n\n" + report_md if report_md else ""), _session_info()


# ── Vision tab ────────────────────────────────────────────────────────────────
def vision_labels_fn():
    with _vision_state["lock"]:
        label = _vision_state.get("latest_label")
    if label is None:
        return "📷 Camera not started yet."
    from vision.bridge import vision_context_string
    rows  = "\n".join(f"| `{k}` | {v} |" for k, v in label.items() if not k.startswith("_"))
    return f"**{vision_context_string(label)}**\n\n| Field | Value |\n|---|---|\n{rows}"


def send_vision_to_pipeline_fn(history):
    with _vision_state["lock"]:
        label = _vision_state.get("latest_label")
    if not label:
        return history, "No camera frame yet.", _session_info()

    from vision.bridge import vision_context_string
    ctx = vision_context_string(label)
    r   = process_turn(SESSION, text=ctx)
    wb, sup = r["wellbeing"], r["support"]
    meta = (f"\n\n_{wb.get('emoji','')} {wb.get('tier','?')} · "
            f"{sup.get('primary','?')} · turn {r['turn']} (vision)_")

    history = history + [
        {"role": "user",      "content": f"📷 {ctx}"},
        {"role": "assistant", "content": r["response"] + meta},
    ]
    return history, "Vision frame sent to pipeline.", _session_info()


# ── Report tab ────────────────────────────────────────────────────────────────
def report_fn():
    if SESSION.n_turns == 0:
        return "No turns yet — chat or speak first."
    return "```\n" + generate_report(SESSION, do_print=False) + "\n```"


def replay_fn():
    from pipeline.replay import run_replay_from_config
    state = run_replay_from_config(use_llm_response=llm.is_alive())
    return "```\n" + generate_report(state, do_print=False) + "\n```"


def reset_fn():
    SESSION.reset()
    _stop_vision()
    return (
        [],                 # clear chatbot
        _ollama_status(),   # refresh ollama status
        _session_info(),    # refresh session info
    )


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

        chatbot = gr.Chatbot(height=380, type="messages")

        with gr.Tab("💬 Chat"):
            report_inline = gr.Markdown()   # auto-report appears here when session ends
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message and press Enter...",
                    scale=8, show_label=False)
                send_btn = gr.Button("Send", scale=1, variant="primary")
            # msg is in BOTH inputs and outputs so it clears after submit
            msg.submit(chat_fn,  [msg, chatbot], [chatbot, msg, report_inline, sess_info])
            send_btn.click(chat_fn, [msg, chatbot], [chatbot, msg, report_inline, sess_info])

        with gr.Tab("🎙️ Voice"):
            mic        = gr.Audio(sources=["microphone"], type="numpy", label="Speak")
            voice_out  = gr.Markdown()
            gr.Button("Transcribe + Send", variant="primary").click(
                voice_fn, [mic, chatbot], [chatbot, voice_out, sess_info])

        with gr.Tab("📷 Vision"):
            gr.Markdown("Server-side webcam stream — all OpenCV overlays drawn on-frame.")
            with gr.Row():
                with gr.Column(scale=2):
                    # gr.Image() for OUTPUT streaming (NOT streaming=True which is browser input)
                    live_feed = gr.Image(
                        label="Live annotated feed",
                        height=int(get("vision.display.height", 480)),
                    )
                with gr.Column(scale=1):
                    vis_labels = gr.Markdown("_Labels appear here_")
                    gr.Button("🔄 Refresh labels").click(vision_labels_fn, None, vis_labels)
                    gr.Button("📤 Send to pipeline", variant="primary").click(
                        send_vision_to_pipeline_fn, [chatbot], [chatbot, vis_labels, sess_info])

            with gr.Row():
                # Start button triggers the generator → streams frames to live_feed
                start_btn = gr.Button("▶ Start camera", variant="primary")
                stop_btn  = gr.Button("⏹ Stop camera")

            start_btn.click(fn=_vision_stream, inputs=None, outputs=live_feed)
            stop_btn.click(fn=_stop_vision, inputs=None, outputs=None)

            gr.Markdown(
                "_Standalone vision windows (run in a separate terminal):_  \n"
                "`python -c \"from vision.faces import run_blink; run_blink()\"`  \n"
                "`python -c \"from vision.hands import run_gesture; run_gesture()\"`"
            )

        with gr.Tab("📋 Report"):
            report_box = gr.Markdown("_Generate a report or run the config replay._")
            with gr.Row():
                gr.Button("📊 Generate report", variant="primary").click(
                    report_fn, None, report_box)
                gr.Button("▶ Run config replay").click(
                    replay_fn, None, report_box)
                gr.Button("🔄 Reset session").click(
                    reset_fn, None, [chatbot, status, sess_info])

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

    if llm.is_alive():
        threading.Thread(target=llm.warm_up, daemon=True).start()

    build_app().launch(
        server_port=int(get("ui.server_port", 7860)),
        share=bool(get("ui.share", False)),
    )

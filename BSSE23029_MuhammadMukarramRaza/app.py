"""
app.py -- gradio entry point.   Run:  python app.py
(Make sure Ollama is running first:  ollama serve)

Architecture:
  • LIVE (core/provider.py) is the single shared state.
    - Camera thread writes LIVE.vision (frame + label)
    - STT writes LIVE.audio.transcript
    - LIVE.session is the SessionState (full turn history)
  • Tab layout is driven by config.yaml (tabs section).
    - Each tab's pipeline is defined there as an ordered stage list.
    - Disabled tabs do not render.
  • run_turn("tab_name", text=...) is the one entry point for all modalities.
  • Auto-report is owned by the event handlers (not by the pipeline).

To add a new tab: add it to config.yaml tabs section + add a handler below.
To change pipeline behaviour: edit config.yaml stages/tabs, no code change.
"""

import threading
import time

import gradio as gr
import numpy as np

from core.conf import get
from core.state import SessionState
from core.provider import LIVE
from core.executor import validate_tabs
from core import llm
from pipeline.turn import run_turn, process_turn   # process_turn kept for replay compat
from text.report import generate_report

# ── initialise LIVE session ───────────────────────────────────────────────────
_session = SessionState()
LIVE.init_session(_session)


# ── helpers ───────────────────────────────────────────────────────────────────
def _ollama_status() -> str:
    return ("🟢 **Ollama connected** — llama3 ready"
            if llm.is_alive() else
            "🔴 **Ollama NOT running** — open a terminal → run `ollama serve`")


def _session_info() -> str:
    s   = LIVE.session
    mx  = int(get("session.max_turns", 0))
    lim = f"/{mx}" if mx > 0 else ""
    return f"Turn **{s.n_turns}**{lim} · voice {s.voice_turns} · text {s.text_turns}"


def _session_over() -> bool:
    mx = int(get("session.max_turns", 0))
    return mx > 0 and LIVE.session.n_turns >= mx


def _bgr_to_rgb(frame):
    import cv2
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _tab_enabled(tab: str) -> bool:
    return bool(get(f"tabs.{tab}.enabled", False))


def _make_meta(ctx: dict) -> str:
    scale_r = ctx.get("scale", {})
    cls_r   = ctx.get("classify", {})
    turn    = ctx.get("log", {}).get("turn", "?")
    src     = ctx.get("source", "text")
    tier    = scale_r.get("tier", "?") if not scale_r.get("_skipped") else "?"
    emoji   = scale_r.get("emoji", "") if not scale_r.get("_skipped") else ""
    cat     = cls_r.get("primary", "?") if not cls_r.get("_skipped") else "?"
    return f"\n\n_{emoji} {tier} · {cat} · turn {turn} ({src})_"


# ── vision background thread ──────────────────────────────────────────────────
def _vision_worker():
    """
    Camera capture + OpenCV detection loop.
    Writes to LIVE.vision (frame + label) every frame.
    HUD overlays are drawn from config.yaml vision.hud.layout.
    """
    import cv2
    from collections import Counter
    from vision.faces import detect_faces, is_smiling, head_zone, _gray, detect_eyes
    from vision.hands import count_fingers as _count_fingers, classify_gesture

    idx        = int(get("vision.webcam_index", 0))
    flip       = bool(get("vision.flip_webcam", True))
    w          = int(get("vision.display.width",  640))
    h          = int(get("vision.display.height", 480))
    layout     = get("vision.hud.layout") or {}
    f_scale    = float(get("vision.hud.font_scale", 0.7))
    thick      = int(get("vision.hud.thickness", 2))
    col_def    = tuple(get("vision.hud.color_default",  [0, 255, 0]))
    col_warn   = tuple(get("vision.hud.color_warning",  [0, 0, 255]))
    col_info   = tuple(get("vision.hud.color_info",     [255, 255, 0]))
    g_hold     = int(get("vision.stability.gesture_hold_frames", 5))
    m_hist     = int(get("vision.stability.mood_history_frames", 15))
    target_fps = max(int(get("vision.display.stream_fps", 15)), 1)

    gest_buf, mood_buf         = [], []
    blink_state                = {"blinks": 0, "closed": False, "closed_frames": 0}
    drowsy_frames              = 0
    fps_val, frame_cnt, t0     = 0.0, 0, time.time()

    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"[vision] could not open webcam {idx}")
        LIVE.vision.stop()
        return

    try:
        while LIVE.vision.running:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            if flip:
                frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (w, h))
            gray  = _gray(frame)

            face_boxes = (detect_faces(gray)
                          if get("vision.face_detect.enabled", True) else [])
            present    = len(face_boxes) > 0

            # blink
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

            # drowsy
            is_drowsy = False
            if get("vision.drowsy.enabled", True) and present:
                x, y, fw_, fh_ = face_boxes[0]
                eyes = detect_eyes(gray[y:y + fh_//2, x:x + fw_])
                drowsy_frames = drowsy_frames + 1 if not eyes else 0
                is_drowsy = drowsy_frames >= int(get("vision.drowsy.closed_frames_alert", 20))

            # mood
            raw_mood = "no_face"
            if get("vision.smile_mood.enabled", True) and present:
                x, y, fw_, fh_ = face_boxes[0]
                roi      = gray[y:y + fh_, x:x + fw_]
                raw_mood = "smiling" if is_smiling(roi) else "neutral"
            mood_buf.append(raw_mood)
            if len(mood_buf) > m_hist:
                mood_buf.pop(0)
            mood = Counter(mood_buf).most_common(1)[0][0] if mood_buf else "no_face"

            # head zone
            zone = "Center"
            if get("vision.head_pose.enabled", True) and present:
                zone = head_zone(face_boxes[0], frame.shape)

            # gesture
            raw_gest, g_emoji = "none", ""
            if get("vision.gesture.enabled", True):
                count, _ = _count_fingers(frame)
                g_name, g_emoji = classify_gesture(count)
                gest_buf.append((g_name, g_emoji, count > 0))
                if len(gest_buf) > g_hold:
                    gest_buf.pop(0)
                if len(gest_buf) == g_hold and len({n for n, _, _ in gest_buf}) == 1:
                    raw_gest, g_emoji = gest_buf[0][0], gest_buf[0][1]
                    if not gest_buf[0][2]:
                        raw_gest = "none"

            # fps
            frame_cnt += 1
            elapsed    = time.time() - t0
            if elapsed >= 1.0:
                fps_val      = frame_cnt / elapsed
                frame_cnt, t0 = 0, time.time()

            # HUD
            wb_last   = LIVE.session.wellbeing_log[-1] if LIVE.session and LIVE.session.wellbeing_log else {}
            elem_text = {
                "fps":       f"FPS: {fps_val:.1f}",
                "tier":      f"{wb_last.get('emoji','')} {wb_last.get('tier','')}".strip(),
                "score":     f"Score: {wb_last.get('score',0):+.2f}" if wb_last else "",
                "turn":      f"Turn: {LIVE.session.n_turns}" if LIVE.session else "",
                "blink":     f"Blinks: {blink_state['blinks']}",
                "drowsy":    get("vision.drowsy.alert_message","DROWSY!") if is_drowsy else "",
                "head_zone": f"Head: {zone}" if zone != "Center" else "",
                "gesture":   f"{raw_gest} {g_emoji}".strip() if raw_gest != "none" else "",
                "mood":      f"Mood: {mood}" if mood != "no_face" else "",
            }
            elem_col = {
                "fps": col_info, "tier": col_def, "score": col_def,
                "turn": col_info, "blink": col_def, "drowsy": col_warn,
                "head_zone": col_info, "gesture": col_def, "mood": col_def,
            }

            def _draw_corner(names, sx, sy, right=False):
                yp = sy
                for nm in names:
                    txt = elem_text.get(nm, "")
                    if not txt:
                        continue
                    col = elem_col.get(nm, col_def)
                    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, f_scale, thick)
                    xp = sx - tw - 4 if right else sx
                    cv2.rectangle(frame, (xp-2, yp-18), (xp+tw+4, yp+4), (0,0,0), -1)
                    cv2.putText(frame, txt, (xp, yp),
                                cv2.FONT_HERSHEY_SIMPLEX, f_scale, col, thick, cv2.LINE_AA)
                    yp += 28

            _draw_corner(layout.get("top_left",    []),  10,   28)
            _draw_corner(layout.get("top_right",   []),  w-10, 28,  right=True)
            _draw_corner(layout.get("bottom_left", []),  10,   h-80)
            _draw_corner(layout.get("bottom_right",[]),  w-10, h-80, right=True)

            for (x, y, fw_, fh_) in face_boxes:
                cv2.rectangle(frame, (x, y), (x+fw_, y+fh_), col_def, 2)

            # update LIVE provider
            label = {
                "present":   present, "faces": len(face_boxes),
                "mood":      mood,    "head_zone": zone,
                "gesture":   raw_gest,"fingers":   0,
                "blinks":    blink_state["blinks"],
                "is_drowsy": is_drowsy,
            }
            LIVE.vision.update(frame.copy(), label)

            if get("vision.display.show_window", False):
                cv2.imshow("HCI Vision", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            time.sleep(1.0 / target_fps)

    finally:
        cap.release()
        if get("vision.display.show_window", False):
            cv2.destroyAllWindows()
        LIVE.vision.stop()


def _start_vision():
    if not LIVE.vision.running:
        LIVE.vision.running = True
        threading.Thread(target=_vision_worker, daemon=True).start()
        time.sleep(0.3)


def _stop_vision():
    LIVE.vision.stop()


def _vision_stream():
    """Gradio generator — yields RGB frames from LIVE.vision."""
    _start_vision()
    target_fps = max(int(get("vision.display.stream_fps", 15)), 1)
    while LIVE.vision.running:
        frame = LIVE.vision.frame
        if frame is not None:
            yield _bgr_to_rgb(frame)
        time.sleep(1.0 / target_fps)


# ── event handlers (one per enabled tab) ─────────────────────────────────────

def _guard_session_over(history):
    """Return (True, updated_history) if max_turns reached."""
    if _session_over():
        return True, history + [{"role": "assistant", "content": "Session complete. Click **Reset** to start a new one."}]
    return False, history


def _auto_report_if_done() -> str:
    """Generate + return report string if session just ended and auto_report is on."""
    if _session_over() and get("session.auto_report", True):
        return "```\n" + generate_report(LIVE.session, do_print=True) + "\n```"
    return ""


def chat_handler(message, history):
    over, history = _guard_session_over(history)
    if over:
        return history, "", "", _session_info()

    ctx     = run_turn("chat", text=message)
    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": ctx.get("response", "") + _make_meta(ctx)},
    ]
    return history, "", _auto_report_if_done(), _session_info()


def voice_handler(audio, history):
    if audio is None:
        return history, "🎙️ Record something first.", "", _session_info()
    over, history = _guard_session_over(history)
    if over:
        return history, "Session complete.", "", _session_info()

    from voice.audio_io import from_gradio
    data, sr = from_gradio(audio)
    LIVE.audio.buffer     = data
    LIVE.audio.transcript = None

    ctx     = run_turn("voice")
    history = history + [
        {"role": "user",      "content": f"🎙️ {ctx.get('text', '') or '(empty)'}"},
        {"role": "assistant", "content": ctx.get("response", "") + _make_meta(ctx)},
    ]
    return history, f"Heard: \"{ctx.get('text','')}\"", _auto_report_if_done(), _session_info()


def multimodal_handler(audio, text_typed, history):
    """
    Voice + optional text + live camera simultaneously.

    audio       = mic recording (or None if user didn't record)
    text_typed  = textbox content (or None/'' if text_input is disabled or empty)
    history     = chatbot history

    The fuser stage in the multimodal pipeline decides which source wins,
    based on config tabs.multimodal.overrides.fuser.config.priority.
    """
    has_audio  = audio is not None
    has_text   = bool((text_typed or "").strip())
    has_vision = LIVE.vision.running

    if not has_audio and not has_text and not has_vision:
        return history, "Provide audio, type a message, or start the camera first.", "", _session_info()

    over, history = _guard_session_over(history)
    if over:
        return history, "Session complete.", "", _session_info()

    # populate LIVE.audio for the stt stage
    if has_audio:
        from voice.audio_io import from_gradio
        data, _sr             = from_gradio(audio)
        LIVE.audio.buffer     = data
        LIVE.audio.transcript = None

    # text_typed goes into ctx["text_raw"] via run_turn's text= kwarg
    typed = (text_typed or "").strip()
    ctx = run_turn("multimodal", text=typed if has_text else None)

    user_label = f"🎭 {ctx.get('text', '') or '(vision)'}"
    history = history + [
        {"role": "user",      "content": user_label},
        {"role": "assistant", "content": ctx.get("response", "") + _make_meta(ctx)},
    ]
    return history, f"Source: {ctx.get('source','?')}", _auto_report_if_done(), _session_info()


def vision_send_handler(history):
    """Send current camera snapshot through the vision pipeline."""
    over, history = _guard_session_over(history)
    if over:
        return history, "Session complete.", "", _session_info()
    if not LIVE.vision.running:
        return history, "Start the camera first.", "", _session_info()

    ctx     = run_turn("vision")
    history = history + [
        {"role": "user",      "content": f"📷 {ctx.get('text', '')}"},
        {"role": "assistant", "content": ctx.get("response", "") + _make_meta(ctx)},
    ]
    return history, "Vision frame sent.", _auto_report_if_done(), _session_info()


def report_handler():
    if LIVE.session.n_turns == 0:
        return "No turns yet — chat or speak first."
    return "```\n" + generate_report(LIVE.session, do_print=False) + "\n```"


def replay_handler():
    from pipeline.replay import run_replay_from_config
    state = run_replay_from_config(use_llm_response=llm.is_alive())
    return "```\n" + generate_report(state, do_print=False) + "\n```"


def reset_handler():
    LIVE.reset()
    LIVE.init_session(SessionState())
    _stop_vision()
    return [], _ollama_status(), _session_info()


def vision_labels_handler():
    label = LIVE.vision_snapshot()
    if label.get("_stale") or not label.get("present"):
        return "📷 Camera not started or no face detected."
    from vision.bridge import vision_context_string
    rows = "\n".join(f"| `{k}` | {v} |" for k, v in label.items()
                     if not k.startswith("_"))
    return f"**{vision_context_string(label)}**\n\n| Field | Value |\n|---|---|\n{rows}"


# ── gradio app (YAML-driven tab rendering) ────────────────────────────────────

def build_app():
    title     = get("ui.app_title",    "HCI Assistant")
    subtitle  = get("ui.app_subtitle", "")
    exam_id   = get("exam_meta.student_id",   "")
    exam_name = get("exam_meta.student_name", "")

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        if subtitle:
            gr.Markdown(f"_{subtitle}_")
        if exam_id:
            gr.Markdown(f"**{exam_id}** — {exam_name}")

        status    = gr.Markdown(_ollama_status())
        sess_info = gr.Markdown(_session_info())

        # shared chatbot — classic [[user, bot]] tuple format (gradio 6.x)
        chatbot = gr.Chatbot(height=380)

        # ── 💬 Chat tab ────────────────────────────────────────────
        if _tab_enabled("chat"):
            with gr.Tab(get("tabs.chat.label", "💬 Chat")):
                report_inline = gr.Markdown()
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your message and press Enter...",
                        scale=8, show_label=False)
                    send_btn = gr.Button("Send", scale=1, variant="primary")
                msg.submit(chat_handler,
                           [msg, chatbot], [chatbot, msg, report_inline, sess_info])
                send_btn.click(chat_handler,
                               [msg, chatbot], [chatbot, msg, report_inline, sess_info])

        # ── 🎙️ Voice tab ───────────────────────────────────────────
        if _tab_enabled("voice"):
            with gr.Tab(get("tabs.voice.label", "🎙️ Voice")):
                voice_report = gr.Markdown()
                mic      = gr.Audio(sources=["microphone"], type="numpy", label="Speak")
                voice_out = gr.Markdown()
                gr.Button("Transcribe + Send", variant="primary").click(
                    voice_handler, [mic, chatbot],
                    [chatbot, voice_out, voice_report, sess_info])

        # ── 🎭 Multimodal tab ──────────────────────────────────────
        if _tab_enabled("multimodal"):
            # Read which input stages are enabled for THIS tab from config.
            # UI components render/hide based on these values — no hardcoding.
            _mm_ov     = get("tabs.multimodal.overrides") or {}
            _mm_stt    = bool((_mm_ov.get("stt")         or {}).get("enabled", True))
            _mm_text   = bool((_mm_ov.get("text_input")  or {}).get("enabled", False))
            _mm_vision = bool((_mm_ov.get("vision_input")or {}).get("enabled", True))

            _active = " + ".join(
                n for n, on in [("voice", _mm_stt), ("text", _mm_text), ("camera", _mm_vision)] if on
            ) or "no inputs enabled — check config tabs.multimodal.overrides"

            with gr.Tab(get("tabs.multimodal.label", "🎭 Multimodal")):
                gr.Markdown(f"Active inputs: **{_active}**. "
                            f"Fuser picks the best available source each turn.")
                mm_report = gr.Markdown()

                with gr.Row():
                    if _mm_vision:
                        with gr.Column(scale=2):
                            mm_feed = gr.Image(
                                label="Live camera (annotated)",
                                height=int(get("vision.display.height", 480)))

                    with gr.Column(scale=1):
                        # Mic — visible when stt is enabled for this tab
                        mm_mic = gr.Audio(
                            sources=["microphone"], type="numpy",
                            label="Speak",
                            visible=_mm_stt,
                        )
                        # Textbox — visible when text_input is enabled.
                        # Always created (keeps handler signature stable),
                        # visible= controls whether the user can see/use it.
                        mm_txt = gr.Textbox(
                            placeholder="Type a message...",
                            label="Text input",
                            visible=_mm_text,
                            show_label=_mm_text,
                        )
                        mm_out = gr.Markdown()
                        gr.Button("Send", variant="primary").click(
                            multimodal_handler,
                            [mm_mic, mm_txt, chatbot],
                            [chatbot, mm_out, mm_report, sess_info],
                        )

                with gr.Row():
                    if _mm_vision:
                        gr.Button("▶ Start camera", variant="primary").click(
                            fn=_vision_stream, inputs=None, outputs=mm_feed)
                        gr.Button("⏹ Stop camera").click(_stop_vision, None, None)

        # ── 📷 Vision tab ──────────────────────────────────────────
        if _tab_enabled("vision"):
            with gr.Tab(get("tabs.vision.label", "📷 Vision")):
                vis_report = gr.Markdown()
                with gr.Row():
                    with gr.Column(scale=2):
                        live_feed = gr.Image(
                            label="Live annotated feed",
                            height=int(get("vision.display.height", 480)))
                    with gr.Column(scale=1):
                        vis_labels = gr.Markdown("_Labels appear here_")
                        gr.Button("🔄 Refresh labels").click(
                            vision_labels_handler, None, vis_labels)
                        gr.Button("📤 Send frame to pipeline",
                                  variant="primary").click(
                            vision_send_handler, [chatbot],
                            [chatbot, vis_labels, vis_report, sess_info])
                with gr.Row():
                    gr.Button("▶ Start camera", variant="primary").click(
                        fn=_vision_stream, inputs=None, outputs=live_feed)
                    gr.Button("⏹ Stop camera").click(_stop_vision, None, None)

                gr.Markdown(
                    "_Standalone windows:_  \n"
                    "`python -c \"from vision.faces import run_blink; run_blink()\"`  \n"
                    "`python -c \"from vision.hands import run_gesture; run_gesture()\"`"
                )

        # ── 📋 Report tab ──────────────────────────────────────────
        if _tab_enabled("report"):
            with gr.Tab(get("tabs.report.label", "📋 Report")):
                report_box = gr.Markdown("_Generate a report or run the config replay._")
                with gr.Row():
                    gr.Button("📊 Generate report", variant="primary").click(
                        report_handler, None, report_box)
                    gr.Button("▶ Run config replay").click(
                        replay_handler, None, report_box)
                    gr.Button("🔄 Reset session").click(
                        reset_handler, None, [chatbot, status, sess_info])

    return demo


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # startup validation
    warnings = validate_tabs()
    for w in warnings:
        print(f"[config WARNING] {w}")

    print("=" * 60)
    print(f"  {get('ui.app_title', 'HCI Assistant')}")
    print(f"  {get('exam_meta.student_id','')}  {get('exam_meta.student_name','')}")
    print(f"  {_ollama_status().replace('**','')}")
    print(f"  Whisper: {get('whisper.model_size')}  "
          f"| Max turns: {get('session.max_turns')}  "
          f"| Tabs: {[t for t in ['chat','voice','multimodal','vision','report'] if _tab_enabled(t)]}")
    print("=" * 60)

    if llm.is_alive():
        threading.Thread(target=llm.warm_up, daemon=True).start()

    build_app().launch(
        server_port=int(get("ui.server_port", 7860)),
        share=bool(get("ui.share", False)),
    )

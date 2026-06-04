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
    """
    Build the small metadata line shown under each assistant reply.
    Only includes tier and category when those features actually ran.
    When scale or categories is disabled, those slots are omitted — no '?' noise.
    """
    from core.conf import get as _get
    scale_r = ctx.get("scale", {})
    cls_r   = ctx.get("classify", {})
    turn    = ctx.get("log", {}).get("turn", "?")
    src     = ctx.get("source", "text")

    parts = []

    # tier — only show if scale ran and produced a real value
    if (not scale_r.get("_skipped")
            and _get("scale.enabled", True)
            and scale_r.get("tier") not in (None, "UNKNOWN", "?")):
        emoji = scale_r.get("emoji", "")
        tier  = scale_r.get("tier", "")
        parts.append(f"{emoji} {tier}".strip())

    # category — only show if classify ran and found something meaningful
    if (not cls_r.get("_skipped")
            and _get("categories.enabled", True)
            and cls_r.get("primary") not in (None, "GENERAL", "?")):
        parts.append(cls_r["primary"])

    parts.append(f"turn {turn} ({src})")
    return "\n\n_" + " · ".join(parts) + "_"


# ── vision background thread ──────────────────────────────────────────────────
def _vision_worker():
    """
    Camera capture + detection loop.
    Draws face mesh + hand mesh on every frame.
    Elegant status bar at the bottom instead of scattered corner text.
    Frame-skip + downscale for performance.
    """
    import cv2
    from collections import Counter
    from vision.backend import BACKEND, print_backend_summary
    from vision.faces import (detect_faces, mood_of, head_zone,
                               head_zone_landmarks, get_face_landmarks,
                               _gray, detect_eyes, make_blink_processor)
    from vision.hands import count_all_hands
    from vision.lips import LipAnalyser
    from vision.draw import (draw_face_mesh, draw_face_box,
                              draw_status_bar, draw_ear_gauge,
                              draw_corner_badge, draw_chip,
                              hud_color, mood_color,
                              _CYAN, _WHITE, _GREEN, _RED, _AMBER)

    print_backend_summary()

    idx          = int(get("vision.webcam_index", 0))
    flip         = bool(get("vision.flip_webcam", True))
    w            = int(get("vision.display.width",  640))
    h            = int(get("vision.display.height", 480))
    m_hist       = int(get("vision.stability.mood_history_frames", 15))
    target_fps   = max(int(get("vision.display.stream_fps", 15)), 1)
    detect_n     = max(1, int(get("vision.detect_every_n", 3)))
    detect_scale = float(get("vision.detect_scale", 0.5))
    EAR_TH       = float(get("vision.blink.ear_closed_threshold", 0.25))

    _det = {
        "face_boxes": [], "present": False, "mood": "no_face",
        "zone": "Forward", "hands": [], "is_drowsy": False,
        "face_landmarks": None, "vowel": "?", "vowel_conf": 0.0,
    }
    _blink_proc, _blink_state = make_blink_processor()
    _lip = LipAnalyser()   # stateful — keeps EMA + majority-vote buffer across frames
    mood_buf         = []
    drowsy_frames    = 0
    fps_val          = 0.0
    frame_cnt        = 0
    detect_frame_cnt = 0
    t0               = time.time()

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
            frame_cnt        += 1
            detect_frame_cnt += 1

            # ── DETECTION (every N frames, downscaled) ──────────────
            if detect_frame_cnt >= detect_n:
                detect_frame_cnt = 0
                if detect_scale < 1.0:
                    small = cv2.resize(frame, (int(w*detect_scale), int(h*detect_scale)))
                else:
                    small = frame

                raw_boxes = (detect_faces(small)
                             if get("vision.face_detect.enabled", True) else [])
                if detect_scale < 1.0 and raw_boxes:
                    inv = 1.0 / detect_scale
                    raw_boxes = [(int(x*inv), int(y*inv), int(bw*inv), int(bh*inv))
                                 for (x, y, bw, bh) in raw_boxes]
                _det["face_boxes"] = raw_boxes
                _det["present"]    = bool(raw_boxes)

                # face landmarks (full resolution — mesh quality matters)
                _det["face_landmarks"] = get_face_landmarks(frame)

                # vowel / lip reading — reuses already-computed landmarks
                if (BACKEND == "mediapipe"
                        and bool(get("vision.lip.enabled", True))
                        and _det["face_landmarks"]):
                    try:
                        lm = _det["face_landmarks"][0].landmark
                        _det["vowel"]      = _lip.update(lm, w, h)
                        _det["vowel_conf"] = round(_lip.last_confidence, 2)
                    except Exception:
                        pass

                # mood (majority vote)
                raw_mood = mood_of(small) if _det["present"] else "no_face"
                mood_buf.append(raw_mood)
                if len(mood_buf) > m_hist: mood_buf.pop(0)
                _det["mood"] = Counter(mood_buf).most_common(1)[0][0] if mood_buf else "no_face"

                # head zone — landmark-based if MP, bbox-based otherwise
                _det["zone"] = "Forward"
                if _det["present"]:
                    if _det["face_landmarks"]:
                        _det["zone"] = head_zone_landmarks(
                            _det["face_landmarks"][0].landmark, w, h)
                    else:
                        _det["zone"] = head_zone(_det["face_boxes"][0], frame.shape)

                # drowsy (non-MP path only; MP uses EAR inside blink proc)
                if BACKEND != "mediapipe" and get("vision.drowsy.enabled", True):
                    if _det["present"]:
                        gray_ = _gray(frame)
                        x_, y_, fw_, fh_ = _det["face_boxes"][0]
                        eyes_ = detect_eyes(gray_[y_:y_+fh_//2, x_:x_+fw_])
                        drowsy_frames = drowsy_frames + 1 if not eyes_ else 0
                        _det["is_drowsy"] = (drowsy_frames >=
                            int(get("vision.drowsy.closed_frames_alert", 20)))
                    else:
                        drowsy_frames = 0; _det["is_drowsy"] = False

                # hands (full resolution for accuracy)
                if get("vision.gesture.enabled", True):
                    _det["hands"] = count_all_hands(frame)   # also draws hand mesh

            # ── BLINK (every frame) ──────────────────────────────────
            if get("vision.blink.enabled", True):
                _blink_proc(frame)

            # ── FPS ──────────────────────────────────────────────────
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                fps_val       = frame_cnt / elapsed
                frame_cnt, t0 = 0, time.time()

            # ── DRAW FACE MESH (on every frame, full res) ────────────
            if _det["face_landmarks"]:
                for fl in _det["face_landmarks"]:
                    draw_face_mesh(frame, fl, h, w)

            # ── FACE BOXES (corner brackets only, elegant) ───────────
            m_col = mood_color(_det["mood"])
            for box in _det["face_boxes"]:
                draw_face_box(frame, box, m_col)

            # ── EAR GAUGE (top-left corner, small) ───────────────────
            ear = _blink_state.get("ear", 0.3)
            draw_ear_gauge(frame, ear, (8, 8), width=50, height=6, threshold=EAR_TH)

            # ── WELLBEING CHIP (top-right) ────────────────────────────
            wb_last = (LIVE.session.wellbeing_log[-1]
                       if LIVE.session and LIVE.session.wellbeing_log else {})
            if wb_last and not wb_last.get("_skipped"):
                tier_str = f"{wb_last.get('emoji','')} {wb_last.get('tier','')}".strip()
                t_col    = hud_color(wb_last.get("tier",""))
                draw_chip(frame, tier_str, (w - 110, 22), t_col)

            # ── BACKEND BADGE (top-right corner, tiny) ───────────────
            draw_corner_badge(frame, f"{BACKEND} {fps_val:.0f}fps", "tr")

            # ── STATUS BAR (bottom panel) ─────────────────────────────
            hands    = _det.get("hands", [])
            raw_gest = hands[0].get("gesture","none") if hands else "none"
            g_emoji  = hands[0].get("emoji","")       if hands else ""

            drowsy_col = _RED   if _blink_state.get("drowsy") or _det.get("is_drowsy") else _GREEN
            zone_col   = _AMBER if _det["zone"] not in ("Forward","Center","no_face") else _CYAN

            slots = []
            if _det["mood"] != "no_face":
                slots.append((f"Mood: {_det['mood']}", m_col))
            slots.append((f"Head: {_det['zone']}", zone_col))
            slots.append((f"Blinks: {_blink_state['blinks']}", _CYAN))
            if _blink_state.get("drowsy") or _det.get("is_drowsy"):
                slots.append(("⚠ DROWSY", _RED))
            # vowel — only show when confident (not "?" or "neutral"/"smile")
            vowel = _det.get("vowel", "?")
            vconf = _det.get("vowel_conf", 0.0)
            if BACKEND == "mediapipe" and vowel in "AEIOU" and vconf > 0:
                slots.append((f"Vowel: {vowel} ({vconf:.2f})", _AMBER))
            if raw_gest != "none":
                if len(hands) > 1:
                    slots.append((" | ".join(
                        f"{hd['hand']}:{hd['gesture']} {hd['emoji']}"
                        for hd in hands), _WHITE))
                else:
                    slots.append((f"{raw_gest} {g_emoji}", _WHITE))
            if LIVE.session and LIVE.session.n_turns > 0:
                slots.append((f"T{LIVE.session.n_turns}", _CYAN))

            draw_status_bar(frame, slots)

            # ── LIVE PROVIDER ─────────────────────────────────────────
            label = {
                "present":   _det["present"],
                "faces":     len(_det["face_boxes"]),
                "mood":      _det["mood"],
                "head_zone": _det["zone"],
                "vowel":     _det.get("vowel", "?"),
                "gesture":   raw_gest,
                "fingers":   hands[0].get("fingers", 0) if hands else 0,
                "blinks":    _blink_state["blinks"],
                "is_drowsy": _blink_state.get("drowsy") or _det.get("is_drowsy", False),
                "backend":   BACKEND,
                "all_hands": hands,
                "ear":       ear,
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
                    "**Standalone windows** (run in a separate terminal):  \n"
                    "```\n"
                    "# Face modalities\n"
                    "python -c \"from vision.faces import run_blink;     run_blink()\"\n"
                    "python -c \"from vision.faces import run_drowsy;    run_drowsy()\"\n"
                    "python -c \"from vision.faces import run_mood;      run_mood()\"\n"
                    "python -c \"from vision.faces import run_head_pose; run_head_pose()\"\n"
                    "python -c \"from vision.faces import run_all_face;  run_all_face()\"\n"
                    "\n"
                    "# Hand modalities\n"
                    "python -c \"from vision.hands import run_gesture;      run_gesture()\"\n"
                    "python -c \"from vision.hands import run_finger_count; run_finger_count()\"\n"
                    "\n"
                    "# Lip / vowel reading  (MediaPipe required)\n"
                    "python -c \"from vision.lips import run_lips; run_lips()\"\n"
                    "\n"
                    "# Color tracking & motion\n"
                    "python -c \"from vision.color_motion import run_color;  run_color('red')\"\n"
                    "python -c \"from vision.color_motion import run_motion; run_motion()\"\n"
                    "```"
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

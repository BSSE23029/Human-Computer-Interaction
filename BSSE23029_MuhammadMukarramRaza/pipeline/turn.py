"""
pipeline/turn.py -- the single multimodal step that ties everything together.

process_turn() accepts ANY combination of text / audio / frame, resolves it to
text, runs the full engine (sentiment→scale→classify→transition→alert→respond),
logs everything to SessionState, and returns a tidy result dict.

Fallback chain built in:
  text missing + audio given → Whisper STT
  vision frame given         → OpenCV label injected as LLM context
  LLM down                   → rule table or default response
  max_turns reached          → returns {'session_ended': True}

    r = process_turn(state, text="I failed my exam")
    r = process_turn(state, audio=mic_array)            # voice turn
    r = process_turn(state, text="...", frame=cam_bgr)  # text + vision context
"""

from core.conf import get
from text.scale import assess, check_and_alert
from text.classify import classify
from text.trajectory import detect_transition


def process_turn(
    state,
    text: str = None,
    audio=None,
    frame=None,
    use_llm_response: bool = None,   # None = auto-decide based on Ollama status
) -> dict:
    """Run one full multimodal turn.

    Returns a dict:
        turn, text, source, wellbeing, support, transition, vision, response,
        session_ended (True if max_turns was reached before this call)
    """
    # ── max_turns guard ───────────────────────────────────────────
    max_turns = int(get("session.max_turns", 0))
    if max_turns > 0 and state.n_turns >= max_turns:
        return {"session_ended": True, "turn": state.n_turns,
                "text": "", "source": "none", "wellbeing": {}, "support": {},
                "transition": None, "vision": None, "response": ""}

    # ── 1. resolve text + source ──────────────────────────────────
    source = "text"
    transcript_result = None
    if text is None and audio is not None:
        try:
            from voice.stt import transcribe
            transcript_result = transcribe(audio)
            text = transcript_result.get("text", "")
            source = "voice"
        except Exception as e:
            text = ""
            print(f"[turn] STT failed: {e}")

    text = (text or "").strip()
    turn = state.add_turn(
        text,
        source=source,
        confidence=transcript_result.get("confidence", "n/a") if transcript_result else "n/a",
    )

    # ── 2. optional vision context ────────────────────────────────
    vlabel = None
    vision_ctx = ""
    if frame is not None and get("session.input_modes.vision", False):
        try:
            from vision.bridge import frame_to_label, vision_context_string
            vlabel = frame_to_label(frame)
            state.log_vision(vlabel)
            if get("vision.bridge.include_in_llm_context", True):
                template = get("vision.bridge.context_template",
                               "[Vision] Mood: {mood}, Gesture: {gesture}")
                vision_ctx = template.format(
                    face_present=vlabel.get("present", False),
                    mood=vlabel.get("mood", "unknown"),
                    gesture=vlabel.get("gesture", "none"),
                    head_zone=vlabel.get("head_zone", "unknown"),
                )
        except Exception as e:
            print(f"[turn] vision failed: {e}")

    # ── 3. text engine ────────────────────────────────────────────
    wb  = assess(text)
    state.log_wellbeing(wb)

    sup = classify(text)
    state.log_support(sup)

    tr  = detect_transition(state.support_log, turn["turn"])
    state.log_transition(tr)

    check_and_alert(wb, turn["turn"])

    # ── 4. response ───────────────────────────────────────────────
    # auto-decide: use LLM memory if Ollama is alive, else rule/default
    if use_llm_response is None:
        from core.llm import is_alive
        use_llm_response = is_alive()

    if use_llm_response:
        try:
            from core.llm import chat_messages, persona_system
            msgs = state.messages_for_llm(system=persona_system())
            # inject vision context into the last user message
            if vision_ctx and msgs:
                last_user = next((m for m in reversed(msgs) if m["role"] == "user"), None)
                if last_user:
                    last_user["content"] = f"{vision_ctx}\n{last_user['content']}"
            reply = chat_messages(msgs)
            if "[LLM" in reply:     # LLM returned an error string — fall through
                raise RuntimeError(reply)
        except Exception:
            use_llm_response = False   # fall through to rule/default

    if not use_llm_response:
        from text.respond import respond
        reply = respond(text, sup["primary"], wb["tier"])

    state.add_response(reply)

    return {
        "turn":         turn["turn"],
        "text":         text,
        "source":       source,
        "wellbeing":    wb,
        "support":      sup,
        "transition":   tr,
        "vision":       vlabel,
        "response":     reply,
        "session_ended":False,
    }

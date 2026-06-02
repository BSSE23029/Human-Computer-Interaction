"""
pipeline/turn.py -- the single multimodal step that ties everything together.

process_turn() takes ANY combination of text / audio / frame, resolves it to
text, runs the full text engine (sentiment -> tier -> classify -> transition ->
alert), generates a response (rule-based or llama3-with-memory), logs everything
to the SessionState, and returns a result dict.

    r = process_turn(state, text="I failed my exam")
    r = process_turn(state, audio=mic_array)            # voice turn
    r = process_turn(state, text="...", frame=cam_bgr)  # text + vision context
"""
from text.scale import assess, check_and_alert
from text.classify import classify
from text.trajectory import detect_transition


def process_turn(state, text: str = None, audio=None, frame=None,
                 use_llm_response: bool = False) -> dict:
    """Run one full multimodal turn. Returns a dict with every sub-result."""
    # 1) Resolve the spoken/typed text + its source.
    source = "text"
    if text is None and audio is not None:
        from voice.stt import transcribe
        text = transcribe(audio).get("text", "")
        source = "voice"
    text = text or ""
    turn = state.add_turn(text, source=source)

    # 2) Optional vision context.
    vlabel = None
    if frame is not None:
        from vision.bridge import frame_to_label
        vlabel = frame_to_label(frame)
        state.log_vision(vlabel)

    # 3) Text engine.
    wb = assess(text)
    state.log_wellbeing(wb)
    sup = classify(text)
    state.log_support(sup)
    tr = detect_transition(state.support_log, turn["turn"])
    state.log_transition(tr)
    check_and_alert(wb, turn["turn"])

    # 4) Response: llama3-with-memory (optionally vision-aware) OR rule combo.
    if use_llm_response:
        from core.llm import chat_messages, persona_system
        msgs = state.messages_for_llm(system=persona_system())
        if vlabel and msgs and msgs[-1]["role"] == "user":
            from vision.bridge import vision_context_string
            msgs[-1]["content"] = f"[camera: {vision_context_string(vlabel)}]\n{msgs[-1]['content']}"
        reply = chat_messages(msgs)
    else:
        from text.respond import respond
        reply = respond(text, sup["primary"], wb["tier"])
    state.add_response(reply)

    return {"turn": turn["turn"], "text": text, "source": source,
            "wellbeing": wb, "support": sup, "transition": tr,
            "vision": vlabel, "response": reply}

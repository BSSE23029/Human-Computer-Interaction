"""
pipeline/turn.py  --  the public API for processing one turn.

Two interfaces:

1. NEW (stage-based, tab-aware):
       from pipeline.turn import run_turn
       ctx = run_turn(tab="chat", text="I failed my exam", turn=1)

   Delegates to core/executor.py → pipeline/stages.py.
   Reads/writes LIVE provider — no state passed as parameters.

2. LEGACY (backward-compatible, kept for exam_adapter + replay):
       from pipeline.turn import process_turn
       result = process_turn(state, text="...", audio=..., frame=...)

   Bridges the old signature into run_turn so existing code
   (exam_adapter, replay, tests) works without change.
"""

from core.conf import get
from core.provider import LIVE


# ── NEW API ───────────────────────────────────────────────────────────────────

def run_turn(tab: str = "chat", **initial_ctx) -> dict:
    """
    Run the ordered stage pipeline for `tab`.

    Any kwargs are injected into the initial TurnContext.
    Most useful kwargs:
        text      str    pre-typed text (placed into text_raw so text_input stage picks it up)
        audio     array  raw float32 mic data (placed into LIVE.audio.buffer)
        turn      int    explicit turn number (defaults to session.n_turns + 1)

    Returns the completed TurnContext dict.
    Session state is updated by the `log` stage automatically.
    Never raises (errors collected in ctx["errors"]).
    """
    from core.executor import run_tab_pipeline

    state = LIVE.session
    turn_num = initial_ctx.pop("turn", (state.n_turns + 1) if state else 1)

    # Pre-populate LIVE.audio.buffer if caller passed raw audio
    audio = initial_ctx.pop("audio", None)
    if audio is not None:
        LIVE.audio.buffer = audio
        LIVE.audio.transcript = None   # force re-transcription

    # text goes into ctx["text_raw"] for the text_input stage to read
    text = initial_ctx.pop("text", None)
    if text is not None:
        initial_ctx["text_raw"] = str(text)

    ctx = {"turn": turn_num, **initial_ctx}
    return run_tab_pipeline(tab, initial_ctx=ctx)


# ── LEGACY API (backward-compatible) ─────────────────────────────────────────

def process_turn(
    state,
    text: str = None,
    audio=None,
    frame=None,
    use_llm_response: bool = None,
    tab: str = "chat",
) -> dict:
    """
    Legacy wrapper — keeps exam_adapter, replay, and existing tests working.

    Maps old (state, text, audio, frame) signature to run_turn().
    Returns a dict with the same keys as the old version for compatibility:
        turn, text, source, wellbeing, support, transition, vision, response, session_ended
    """
    # session guard — max_turns check
    max_turns = int(get("session.max_turns", 0))
    if max_turns > 0 and state.n_turns >= max_turns:
        return {
            "session_ended": True, "turn": state.n_turns,
            "text": "", "source": "none", "wellbeing": {}, "support": {},
            "transition": None, "vision": None, "response": "",
        }

    # temporarily point LIVE.session at the passed state
    # (handles replay where a fresh SessionState is created per run)
    _prev_session = LIVE.session
    LIVE.session = state

    # resolve tab from input type when not specified
    if tab == "chat":
        if audio is not None:
            tab = "voice"
        elif frame is not None:
            tab = "vision"

    ctx = run_turn(tab=tab, text=text, audio=audio, turn=state.n_turns + 1)

    LIVE.session = _prev_session

    # build backward-compatible result dict
    return {
        "session_ended": False,
        "turn":          ctx.get("log", {}).get("turn", state.n_turns),
        "text":          ctx.get("text", text or ""),
        "source":        ctx.get("source", "text"),
        "wellbeing":     ctx.get("scale", {}),
        "support":       ctx.get("classify", {}),
        "transition":    (ctx.get("trajectory") or {}).get("transition"),
        "vision":        ctx.get("vision_input"),
        "response":      ctx.get("response", ""),
        "errors":        ctx.get("errors", []),
    }

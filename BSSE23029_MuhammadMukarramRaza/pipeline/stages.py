"""
pipeline/stages.py  --  one implementation function per stage type.

Every function has the same signature:
    fn(cfg: dict, ctx: dict) -> dict

`cfg`  = the fully resolved stage config (method, fallback, config sub-dict, ...)
`ctx`  = the TurnContext dict — read upstream results, write yours here

INTERDEPENDENCY CONTRACT (matches executor.py):
    Before reading a dependency's output, always check:
        ctx["dependency_name"].get("_skipped")
    If True → the dependency was disabled or failed.
    Handle gracefully — use LLM fallback or safe default.

STAGE REGISTRY at bottom — maps type strings to functions.
"""

import re
import time

from core.conf import get as _get
from core.provider import LIVE


# ── INPUT STAGES ─────────────────────────────────────────────────────────────

def run_voice_input(cfg: dict, ctx: dict) -> dict:
    """STT: mic buffer → transcript dict.
    Reads from LIVE.audio.transcript if already populated (gradio mic flow),
    or falls back to transcribing LIVE.audio.buffer (programmatic flow).
    """
    transcript = LIVE.latest_transcript()
    if transcript and transcript.get("text"):
        return transcript

    buf = LIVE.audio.buffer
    if buf is None or (hasattr(buf, "__len__") and len(buf) == 0):
        return {"text": "", "language": "?", "confidence": "low"}

    try:
        from voice.stt import transcribe
        sr = int(_get("audio.sample_rate", 16000))
        result = transcribe(buf, sr)
        LIVE.audio.set_transcript(result)
        return result
    except Exception as e:
        return {"text": "", "language": "?", "confidence": "low", "_error": str(e)}


def run_text_input(cfg: dict, ctx: dict) -> dict:
    """Pass-through: the text was already placed in ctx["text_raw"] by the
    gradio handler before calling the pipeline."""
    text = ctx.get("text_raw") or ""
    return {"text": str(text), "_skipped": False}


def run_vision_input(cfg: dict, ctx: dict) -> dict:
    """Capture vision label from LIVE (camera thread writes this continuously).
    Returns the snapshot at this exact moment — the 'what the camera sees right now'.
    """
    label = LIVE.vision_snapshot()
    if label.get("_stale"):
        # camera stopped or frame is old — return null but not _skipped
        return {**label, "_stale": True}
    return label


# ── FUSER ────────────────────────────────────────────────────────────────────

def run_fuser(cfg: dict, ctx: dict) -> dict:
    """
    Resolve multiple input stages into ONE text payload.

    Priority list (from config or default):
        voice → text → vision

    Walk the list; first source with actual content wins.
    If nothing has content → no_input_behavior triggers.
    """
    method   = cfg.get("method", "auto_priority")
    sub_cfg  = cfg.get("config") or {}
    priority = sub_cfg.get("priority", ["voice", "text", "vision"])

    def _from_stt():
        r = ctx.get("stt_result") or {}
        if r.get("_skipped"):
            return None
        t = r.get("text", "").strip()
        return ("voice", t) if t else None

    def _from_text():
        r = ctx.get("text_input") or {}
        if r.get("_skipped"):
            return None
        t = r.get("text", ctx.get("text_raw", "")).strip()
        return ("text", t) if t else None

    def _from_vision():
        label = ctx.get("vision_label") or ctx.get("vision_input") or {}
        if label.get("_skipped") or label.get("_stale"):
            return None
        try:
            from vision.bridge import vision_context_string
            text = vision_context_string(label)
            return ("vision", text) if text and text != "No person is visible on camera." else None
        except Exception:
            return None

    _SOURCES = {"voice": _from_stt, "text": _from_text, "vision": _from_vision}

    chosen_source, chosen_text = "none", ""

    if method in ("voice_only", "text_only", "vision_only"):
        key = method.replace("_only", "")
        result = _SOURCES.get(key, lambda: None)()
        if result:
            chosen_source, chosen_text = result

    else:   # auto_priority
        for src_name in priority:
            result = _SOURCES.get(src_name, lambda: None)()
            if result:
                chosen_source, chosen_text = result
                break

    if not chosen_text:
        no_input = sub_cfg.get("no_input_behavior", "warn")
        if no_input == "error":
            raise ValueError("Fuser: no input source produced content. "
                             "Check that at least one input stage is enabled.")
        # warn or hide_tab: return empty and let downstream handle it
        return {"text": "", "source": "none", "word_count": 0, "_no_input": True}

    return {
        "text":       chosen_text,
        "source":     chosen_source,
        "word_count": len(chosen_text.split()),
    }


# ── CONTEXT INJECTOR ────────────────────────────────────────────────────────

def run_context_injector(cfg: dict, ctx: dict) -> dict:
    """
    Attach vision context to the turn based on vision_fusion method.

    method: "context"  → append vision_context_string to LLM prompt context
            "score"    → store score_delta for the sentiment stage to use
            "primary"  → vision IS the text (override ctx["text"])
            "annotate" → log only, no effect on processing
    """
    method = cfg.get("method", "context")
    label  = LIVE.vision_snapshot()

    if label.get("_stale") or not label.get("present"):
        return {"injected": False, "method": method}

    try:
        from vision.bridge import vision_context_string
        ctx_str = vision_context_string(label)
    except Exception:
        return {"injected": False}

    if method == "primary":
        # Vision replaces text entirely
        ctx["text"]   = ctx_str
        ctx["source"] = "vision"
        return {"injected": True, "method": "primary", "visual_text": ctx_str}

    if method == "context":
        # Store for LLM stages to prepend to their prompts
        ctx["vision_context_str"] = ctx_str
        return {"injected": True, "method": "context", "visual_text": ctx_str}

    if method == "score":
        # Compute a score delta from visual signals
        weights = (cfg.get("config") or {}).get("score_weights", {})
        delta = 0.0
        mood = label.get("mood", "")
        if mood in weights:
            delta += float(weights[mood])
        if label.get("is_drowsy") and "drowsy" in weights:
            delta += float(weights["drowsy"])
        gesture = label.get("gesture", "")
        if gesture in weights:
            delta += float(weights[gesture])
        ctx["vision_score_delta"] = delta
        return {"injected": True, "method": "score", "delta": delta}

    # annotate — log only
    ctx["vision_label"] = label
    return {"injected": True, "method": "annotate"}


# ── ANALYSIS STAGES ───────────────────────────────────────────────────────────

def run_sentiment(cfg: dict, ctx: dict) -> dict:
    """
    text → compound score → {compound, pos, neg, neu}

    method: "lexicon" → VADER-free lexicon scorer
            "llm"     → llama3 scores the text
            "hybrid"  → average of both

    Legacy flag: sentiment.enabled=false → forces method to "llm"
    Vision score delta is added if context_inject ran with method="score".
    """
    text   = ctx.get("text", "")
    # Legacy flag switches method (does not disable the stage)
    if not _get("sentiment.enabled", True):
        method = "llm"
    else:
        method = cfg.get("method", "lexicon")

    # vision score delta (may be 0.0 if not injected)
    delta = float(ctx.get("vision_score_delta", 0.0))

    def _lexicon():
        from text.sentiment import score_text, analyze
        result = analyze(text)
        result["compound"] = max(-1.0, min(1.0, result["compound"] + delta))
        return result

    def _llm():
        from core.llm import score as llm_score, is_alive
        if not is_alive():
            raise RuntimeError("LLM offline")
        s = llm_score(text, -1.0, 1.0, "emotional sentiment (-1=distressed, +1=positive)")
        s = max(-1.0, min(1.0, s + delta))
        return {"compound": round(s, 4), "pos": 0.0, "neg": 0.0, "neu": 0.0}

    if method == "hybrid":
        try:
            lex = _lexicon()
            llm = _llm()
            compound = (lex["compound"] + llm["compound"]) / 2.0
            return {**lex, "compound": round(compound, 4)}
        except Exception:
            return _lexicon()   # hybrid fails → lexicon is safe fallback

    if method == "llm":
        return _llm()

    return _lexicon()   # default: lexicon


def run_scale(cfg: dict, ctx: dict) -> dict:
    """
    compound score → tier dict {tier, score, emoji, is_at_risk}

    INTERDEPENDENCY: reads ctx["sentiment"].
      _skipped=True  → sentiment stage was HARD-DISABLED (not in pipeline).
                        Scale asks LLM to score the raw text.
      _skipped absent → sentiment ran (via lexicon OR llm method).
                        Scale reads the compound score normally.

    Legacy flag: scale.enabled=false → forces method to "llm" (scores text directly).
    """
    from text.scale import to_tier

    sent = ctx.get("sentiment", {})
    text = ctx.get("text", "")

    # Case 1: sentiment stage was hard-disabled (_skipped marker)
    if sent.get("_skipped"):
        try:
            from core.llm import score as llm_score, is_alive
            if not is_alive():
                raise RuntimeError("LLM offline")
            compound = llm_score(
                text, -1.0, 1.0,
                "emotional sentiment (-1=severely distressed, +1=very positive)"
            )
        except Exception:
            compound = 0.0
        return to_tier(compound)

    # Case 2: legacy flag scale.enabled=false → LLM scores text directly
    if not _get("scale.enabled", True):
        try:
            from core.llm import score as llm_score, is_alive
            if not is_alive():
                raise RuntimeError("LLM offline")
            compound = llm_score(
                text, -1.0, 1.0,
                "emotional sentiment (-1=severely distressed, +1=very positive)"
            )
        except Exception:
            compound = float(sent.get("compound", 0.0))
        return to_tier(compound)

    # Case 3: normal path — read sentiment compound score
    compound = float(sent.get("compound", 0.0))
    return to_tier(compound)


def run_classify(cfg: dict, ctx: dict) -> dict:
    """
    text → {primary, all_detected, scores}

    method: "keyword"           → fast deterministic matching
            "llm"               → llama3 classifies
            "keyword_then_llm"  → keyword first; if no hits, llm decides

    Legacy flag: categories.enabled=false → forces method to "llm"
    """
    text   = ctx.get("text", "")
    # Legacy flag switches method (does not skip the stage)
    if not _get("categories.enabled", True):
        method = "llm"
    else:
        method = cfg.get("method", "keyword")

    def _keyword():
        from text.classify import classify
        return classify(text)

    def _llm():
        from text.classify import classify_llm, classify_llm_multi
        from core.llm import is_alive
        if not is_alive():
            raise RuntimeError("LLM offline")
        primary = classify_llm(text)
        all_det = classify_llm_multi(text)
        if primary not in all_det:
            all_det.insert(0, primary)
        return {"primary": primary, "all_detected": all_det, "scores": {}}

    if method == "llm":
        return _llm()

    if method == "keyword_then_llm":
        result = _keyword()
        if result["primary"] == "GENERAL" and not result["all_detected"]:
            try:
                return _llm()
            except Exception:
                return result
        return result

    return _keyword()   # default: keyword


def run_trajectory(cfg: dict, ctx: dict) -> dict:
    """
    session history → {trend, lowest_tier, at_risk_turns, transitions}

    INTERDEPENDENCY: uses state.wellbeing_log + state.support_log.
    If classify was _skipped → support_log may be empty; skip transition detection.
    """
    from text.trajectory import compute_trajectory, detect_transition

    state = LIVE.session
    if state is None:
        return {"trend": "unknown", "lowest_tier": None, "at_risk_turns": [],
                "_skipped": True}

    traj = compute_trajectory(state.wellbeing_log)

    # Transition detection — only if classify stage ran AND produced real data.
    # _skipped=True means classify was HARD-DISABLED (not in pipeline_order).
    # When categories.enabled=false, classify runs via LLM → still produces data.
    classify_result = ctx.get("classify", {})
    transition = None
    if not classify_result.get("_skipped") and len(state.support_log) >= 1:
        transition = detect_transition(state.support_log, ctx.get("turn", state.n_turns))

    return {**traj, "transition": transition}


# ── RESPONDER ─────────────────────────────────────────────────────────────────

def run_respond(cfg: dict, ctx: dict) -> dict:
    """
    tier + category + text → response string

    INTERDEPENDENCY: reads ctx["scale"] and ctx["classify"].
    If either was _skipped → degrade gracefully (use UNKNOWN/GENERAL).

    Vision context string is prepended to LLM prompt if context_inject ran.
    """
    text     = ctx.get("text", "")
    method   = cfg.get("method", "rule_then_llm")

    # Read tier — degrade gracefully if scale was skipped
    scale_r  = ctx.get("scale", {})
    tier     = "UNKNOWN" if scale_r.get("_skipped") else scale_r.get("tier", "NEUTRAL")

    # Read category — degrade gracefully if classify was skipped
    cls_r    = ctx.get("classify", {})
    category = "GENERAL" if cls_r.get("_skipped") else cls_r.get("primary", "GENERAL")

    # Legacy flag: responses.enabled=false → skip rule table, go direct to LLM
    if not _get("responses.enabled", True):
        method = "llm_only"

    def _rule():
        from text.respond import respond
        return respond(text, category, tier)

    def _llm():
        from core.llm import chat, build_prompt, persona_system, is_alive
        if not is_alive():
            raise RuntimeError("LLM offline")
        vis_ctx = ctx.get("vision_context_str", "")
        full_text = f"{vis_ctx}\n{text}".strip() if vis_ctx else text
        prompt = build_prompt(
            "respond",
            persona=persona_system(),
            tier=tier,
            category=category,
            text=full_text,
        )
        return chat(prompt)

    def _rule_then_llm():
        result = _rule()
        default = _get("responses.default", "")
        if result == default:
            try:
                return _llm()
            except Exception:
                return result
        return result

    _METHODS = {
        "rule_only":     _rule,
        "llm_only":      _llm,
        "rule_then_llm": _rule_then_llm,
    }

    response = _METHODS.get(method, _rule_then_llm)()
    return {"response": response}


# ── OUTPUT STAGES ─────────────────────────────────────────────────────────────

def run_display(cfg: dict, ctx: dict) -> dict:
    """
    No-op in the pipeline — gradio display is handled by the event handler.
    This stage's presence in the pipeline confirms 'display is enabled for this tab'.
    The event handler reads ctx["response"] and appends to the chatbot.
    """
    return {"displayed": True}


def run_tts(cfg: dict, ctx: dict) -> dict:
    """
    Speak the response aloud using system TTS.

    method: "system" → macOS: `say`, Linux: `espeak`
            "silent" → log only, no audio

    Runs in a background thread (non-blocking) unless cfg.config.blocking=True.
    """
    import subprocess
    import threading
    import platform

    response = ctx.get("response", "")
    if not response:
        return {"spoken": False}

    method   = cfg.get("method", "system")
    sub_cfg  = cfg.get("config") or {}
    rate     = int(sub_cfg.get("rate", 150))
    blocking = bool(sub_cfg.get("blocking", False))

    if method == "silent":
        return {"spoken": False, "reason": "silent mode"}

    # Build platform command
    system = platform.system()
    if system == "Darwin":
        cmd = ["say", "-r", str(rate), response]
    elif system == "Linux":
        cmd = ["espeak", "-s", str(rate), response]
    else:
        # Windows: no reliable offline TTS without pyttsx3
        return {"spoken": False, "reason": f"TTS not supported on {system}"}

    def _speak():
        try:
            subprocess.run(cmd, timeout=60, check=False,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    if blocking:
        _speak()
    else:
        threading.Thread(target=_speak, daemon=True).start()

    return {"spoken": True, "method": method, "system": system}


# ── LOGGER ────────────────────────────────────────────────────────────────────

def run_log(cfg: dict, ctx: dict) -> dict:
    """
    Write the completed turn to SessionState.
    This stage always runs (never disable — session history integrity).

    Writes:
        state.turns          via add_turn()
        state.wellbeing_log  via log_wellbeing()
        state.support_log    via log_support()
        state.transition_log via log_transition()
        state.vision_log     via log_vision()
        state.responses      via add_response()
    """
    state = LIVE.session
    if state is None:
        return {"logged": False, "_error": "LIVE.session not initialised"}

    text       = ctx.get("text", "")
    source     = ctx.get("source", "text")
    word_count = ctx.get("word_count", len(text.split()))
    turn_num   = ctx.get("turn", state.n_turns + 1)

    state.add_turn(text, source=source, word_count=word_count)

    # wellbeing — write whatever scale produced (or its null)
    scale_r = ctx.get("scale") or {"tier": "NEUTRAL", "score": 0.0,
                                    "emoji": "😐", "is_at_risk": False}
    state.log_wellbeing(scale_r)

    # support — write whatever classify produced (or its null)
    cls_r = ctx.get("classify") or {"primary": "GENERAL", "all_detected": [], "scores": {}}
    state.log_support(cls_r)

    # transition — from trajectory stage
    traj_r = ctx.get("trajectory") or {}
    transition = traj_r.get("transition")
    if transition:
        state.log_transition(transition)

    # vision — write snapshot if available
    vision_label = LIVE.vision_snapshot()
    if vision_label and not vision_label.get("_stale"):
        state.log_vision(vision_label)

    # response
    state.add_response(ctx.get("response", ""))

    # fire any at-risk alerts
    from text.scale import check_and_alert
    check_and_alert(scale_r, turn_num)

    return {"logged": True, "turn": turn_num}


# ── STAGE REGISTRY ────────────────────────────────────────────────────────────
# Maps stage type strings (from config) to implementation functions.

STAGE_REGISTRY = {
    "voice_input":    run_voice_input,
    "text_input":     run_text_input,
    "vision_input":   run_vision_input,
    "fuser":          run_fuser,
    "context_injector": run_context_injector,
    "analyzer":       run_sentiment,
    "tier_mapper":    run_scale,
    "classifier":     run_classify,
    "tracker":        run_trajectory,
    "responder":      run_respond,
    "output_display": run_display,
    "output_tts":     run_tts,
    "logger":         run_log,
}

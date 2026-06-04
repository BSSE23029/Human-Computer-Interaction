"""
text/report.py -- risk scoring + the structured session report (NEXUS Stage 5).

scoring.enabled=True  → formula-based risk score
scoring.enabled=False → llama3 estimates risk from full history

report.use_llm_narrative=True → llama3 writes the clinical narrative section

    risk = compute_risk(state)         -> {'score':72, 'action':'URGENT_REFERRAL', ...}
    generate_report(state)             -> prints + returns the full banner string
    report_dict(state)                 -> machine-readable dict (for gradio Report tab)
"""

from core.conf import get, enabled
from text.trajectory import compute_trajectory, frequency_rank_log


# ── risk formula ──────────────────────────────────────────────────────────────
def _action_for(score: float) -> str:
    """Pick the action whose threshold the score clears (checked high → low)."""
    thresholds = get("scoring.thresholds") or {"NO_ACTION": 0}
    for name, minv in sorted(thresholds.items(), key=lambda kv: float(kv[1]), reverse=True):
        if score >= float(minv):
            return name
    return "NO_ACTION"


def compute_risk(state) -> dict:
    """Config-driven risk formula clamped to [0, 100].

    Formula (when scoring.enabled=True):
        risk = base_risk
             + abs(avg_sentiment) * wellbeing_weight
             + sum(tier_weights[tier] for each turn)
             + escalation_count * escalation_weight
             + verbose_bonus (if avg_words > verbose_threshold)

    When scoring.enabled=False → asks llama3 to estimate a 0-100 risk score.

    (NEXUS Risk Score formula equivalent.)
    """
    if not enabled("scoring"):
        return _risk_fallback_llm(state)

    sc         = get("scoring") or {}
    tier_w     = sc.get("tier_weights") or {}
    wl         = state.wellbeing_log
    scores     = [w.get("score", 0.0) for w in wl]
    avg        = sum(scores) / len(scores) if scores else 0.0
    escalations = state.escalation_count

    risk = float(sc.get("base_risk", 20))
    risk += abs(avg) * float(sc.get("wellbeing_weight", 40))
    risk += sum(float(tier_w.get(w.get("tier", ""), 0)) for w in wl)
    risk += escalations * float(sc.get("escalation_weight", 10))
    if state.avg_words > float(sc.get("verbose_threshold", 20)):
        risk += float(sc.get("verbose_bonus", 5))

    risk = max(0, min(100, round(risk)))
    return {
        "score":          risk,
        "action":         _action_for(risk),
        "avg_wellbeing":  round(avg, 3),
        "at_risk_count":  state.at_risk_count,
        "escalations":    escalations,
    }


def _risk_fallback_llm(state) -> dict:
    """When scoring.enabled=False — ask llama3 for a 0-100 risk estimate."""
    try:
        from core.llm import score as llm_score, is_alive
        if is_alive():
            convo = state.conversation_text()
            risk_float = llm_score(
                convo, lo=0.0, hi=100.0,
                criterion="mental health risk for a university student (0=fine, 100=crisis)"
            )
            risk = max(0, min(100, round(risk_float)))
            return {"score": risk, "action": _action_for(risk),
                    "avg_wellbeing": state.avg_sentiment,
                    "at_risk_count": state.at_risk_count,
                    "escalations": state.escalation_count}
    except Exception:
        pass
    return {"score": 0, "action": "NO_ACTION",
            "avg_wellbeing": 0.0, "at_risk_count": 0, "escalations": 0}


# ── report dict (machine-readable) ───────────────────────────────────────────
def report_dict(state) -> dict:
    """All report fields as a plain dict — used by gradio Report tab."""
    traj = compute_trajectory(state.wellbeing_log)
    risk = compute_risk(state)
    cats = frequency_rank_log(state.support_log, "primary")
    escalations = [e for e in state.transition_log if e.get("is_escalation")]
    return {
        "turns_total":       state.n_turns,
        "voice_turns":       state.voice_turns,
        "text_turns":        state.text_turns,
        "trajectory":        traj["trend"],
        "lowest_tier":       traj["lowest_tier"],
        "at_risk_alerts":    state.at_risk_count,
        "at_risk_turns":     [i + 1 for i in state.at_risk_turns],
        "categories_ranked": cats,
        "escalation_events": escalations,
        "risk_score":        risk["score"],
        "recommended_action":risk["action"],
        "avg_wellbeing":     risk["avg_wellbeing"],
    }


# ── LLM narrative ─────────────────────────────────────────────────────────────
def generate_narrative(state, risk_data: dict) -> str:
    """Ask llama3 to write the clinical narrative section.
    Falls back to a plain text summary if Ollama is unavailable.
    """
    try:
        from core.llm import chat, is_alive
        if not is_alive():
            raise RuntimeError("Ollama offline")

        prompt_template = get("report.llm_narrative_prompt", "")
        if not prompt_template:
            raise ValueError("no narrative prompt in config")

        prompt = prompt_template.format(
            audience     = get("report.narrative_audience", "counselling team"),
            risk_score   = risk_data.get("score", "?"),
            action       = risk_data.get("action", "?"),
            tier_summary = state.tier_summary_text(),
            conversation = state.conversation_text(),
        )
        reply = chat(prompt, system="You are writing a clinical session report. Be precise and professional.")
        if reply and "[LLM" not in reply:
            return reply
    except Exception:
        pass

    # deterministic fallback narrative
    return (
        f"Session of {state.n_turns} turns. "
        f"Lowest tier reached: {state.lowest_tier}. "
        f"At-risk alerts: {state.at_risk_count}. "
        f"Dominant support need: {state.dominant_category}. "
        f"Risk score: {risk_data.get('score', '?')}/100 → "
        f"{risk_data.get('action', 'NO_ACTION')}."
    )


# ── full banner report ────────────────────────────────────────────────────────
def generate_report(state, do_print: bool = True) -> str:
    """Build and optionally print the full session report.
    Returns the report as a string too (for gradio display).
    (NEXUS `generate_intelligence_report()` equivalent.)
    """
    d    = report_dict(state)
    risk = compute_risk(state)
    secs = get("report.sections") or {}
    w    = int(get("report.banner_width", 55))
    bar  = "=" * w
    title    = get("report.banner_title", "SESSION REPORT")
    subtitle = get("report.banner_subtitle", "")

    lines = [bar, f"  {title}"]
    if subtitle:
        lines.append(f"  {subtitle}")
    # inject exam meta into the banner if present
    sid  = get("exam_meta.student_id", "")
    sname = get("exam_meta.student_name", "")
    course = get("exam_meta.course", "")
    edate  = get("exam_meta.exam_date", "")
    if sid or sname:
        lines.append(f"  {sid}  {sname}".strip())
    if course:
        lines.append(f"  {course}")
    if edate:
        lines.append(f"  {edate}")
    lines.append(bar)

    if secs.get("session_summary", True):
        lines.append(f"Turns total      : {d['turns_total']}  "
                     f"(voice: {d['voice_turns']}, text: {d['text_turns']})")

    if secs.get("wellbeing_trajectory", True):
        lines.append(f"Wellbeing trend  : {str(d['trajectory']).upper()}")
        lines.append(f"Lowest tier      : {d['lowest_tier']}")

    if secs.get("support_categories", True):
        cats = ", ".join(f"{c}({n})" for c, n in d["categories_ranked"]) or "none"
        lines.append(f"Categories       : {cats}")

    if secs.get("escalations", True):
        esc_str = ", ".join(
            f"{e['prev']}->{e['curr']}@t{e['turn']}"
            for e in d["escalation_events"]
        ) or "none"
        lines.append(f"Escalations      : {len(d['escalation_events'])}   {esc_str}")

    if secs.get("risk_score", True):
        lines.append(f"At-risk alerts   : {d['at_risk_alerts']}   "
                     f"turns={d['at_risk_turns']}")
        lines.append(f"Risk score       : {d['risk_score']}/100")
        lines.append(f"Recommended      : {d['recommended_action']}")

    if secs.get("include_raw_turns", False):
        lines.append("")
        lines.append("-- Turn Log --")
        for i, turn in enumerate(state.turns):
            wb = state.wellbeing_log[i] if i < len(state.wellbeing_log) else {}
            sp = state.support_log[i]   if i < len(state.support_log)   else {}
            lines.append(
                f"  T{turn['turn']:02d} [{turn['source']}] "
                f"{wb.get('emoji','')} {wb.get('tier','?'):12} "
                f"{sp.get('primary','?'):12} | {turn['text'][:50]}"
            )

    if secs.get("llm_narrative", True) and get("report.use_llm_narrative", True):
        lines.append("")
        lines.append("-- Clinical Narrative --")
        narrative = generate_narrative(state, risk)
        # word-wrap at ~w chars
        words = narrative.split()
        line, wrapped = [], []
        for word in words:
            if len(" ".join(line + [word])) > w - 2:
                wrapped.append("  " + " ".join(line))
                line = [word]
            else:
                line.append(word)
        if line:
            wrapped.append("  " + " ".join(line))
        lines.extend(wrapped)

    lines.append(bar)
    text = "\n".join(lines)
    if do_print:
        print(text)
    return text


if __name__ == "__main__":
    # Quick self-test with synthetic data
    from core.state import SessionState
    s = SessionState()
    messages = [
        ("Hi I need help", "GENERAL", "NEUTRAL", 0.0, False),
        ("I have a deadline tomorrow", "ACADEMIC", "STRESSED", -0.25, False),
        ("I feel completely hopeless", "WELLBEING", "CRISIS", -0.75, True),
        ("My fees are overdue too", "FINANCIAL", "DISTRESSED", -0.5, False),
        ("A friend texted, feeling a bit better", "SOCIAL", "NEUTRAL", 0.1, False),
    ]
    for text, cat, tier, score, at_risk in messages:
        s.add_turn(text, source="text")
        s.log_wellbeing({"tier": tier, "score": score, "emoji": "❓", "is_at_risk": at_risk})
        s.log_support({"primary": cat, "all_detected": [cat], "scores": {}})
        s.add_response("I hear you.")

    generate_report(s)

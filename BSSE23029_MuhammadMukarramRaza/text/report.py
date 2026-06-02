"""
text/report.py -- risk scoring + the structured session report (NEXUS Stage 5).

    risk = compute_risk(state)          -> {'score':72,'action':'URGENT_REFERRAL',...}
    print(generate_report(state))       -> the full banner report (also returns the string)
    report_dict(state)                  -> machine-readable version (for gradio/JSON)
"""
from core.conf import get
from text.trajectory import compute_trajectory, frequency_rank_log


def _action_for(score: float) -> str:
    """Pick the action whose threshold the score clears (checked high -> low)."""
    thresholds = get("scoring.thresholds") or {"NO_ACTION": 0}
    for name, minv in sorted(thresholds.items(), key=lambda kv: kv[1], reverse=True):
        if score >= minv:
            return name
    return "NO_ACTION"


def compute_risk(state) -> dict:
    """Config-driven risk formula, clamped to [0,100], mapped to an action.
    (NEXUS Risk Score formula.)"""
    sc = get("scoring") or {}
    wl = state.wellbeing_log
    scores = [w.get("score", 0.0) for w in wl]
    avg = sum(scores) / len(scores) if scores else 0.0
    at_risk = sum(1 for w in wl if w.get("is_at_risk"))
    escalations = sum(1 for e in state.transition_log if e.get("is_escalation"))

    risk = sc.get("base_risk", 20)
    risk += abs(avg) * sc.get("wellbeing_weight", 40)
    risk += at_risk * sc.get("at_risk_weight", 15)
    risk += escalations * sc.get("escalation_weight", 10)
    if state.avg_words > sc.get("verbose_words_per_turn", 20):
        risk += sc.get("verbose_bonus", 5)
    risk = max(0, min(100, round(risk)))

    return {"score": risk, "action": _action_for(risk),
            "avg_wellbeing": round(avg, 3),
            "at_risk_count": at_risk, "escalations": escalations}


def report_dict(state) -> dict:
    """All report fields as a dict (handy for the gradio Report tab)."""
    traj = compute_trajectory(state.wellbeing_log)
    risk = compute_risk(state)
    cats = frequency_rank_log(state.support_log, "primary")
    at_risk_turns = [i + 1 for i, w in enumerate(state.wellbeing_log) if w.get("is_at_risk")]
    escalations = [e for e in state.transition_log if e.get("is_escalation")]
    return {
        "turns_total": state.n_turns,
        "voice_turns": state.voice_turns,
        "text_turns": state.text_turns,
        "trajectory": traj["trend"],
        "lowest_tier": traj["lowest_tier"],
        "at_risk_alerts": len(at_risk_turns),
        "at_risk_turns": at_risk_turns,
        "categories_ranked": cats,
        "escalation_events": escalations,
        "risk_score": risk["score"],
        "recommended_action": risk["action"],
    }


def generate_report(state, do_print: bool = True) -> str:
    """Build the banner report. Prints to console AND returns the string.
    (NEXUS `generate_intelligence_report()` equivalent.)"""
    d = report_dict(state)
    bar = "=" * 55
    banner = get("ui.report_banner", "SESSION REPORT")
    code = get("ui.report_code", "")

    cats = ", ".join(f"{c}({n})" for c, n in d["categories_ranked"]) or "none"
    esc = ", ".join(f"{e['prev']}->{e['curr']}@t{e['turn']}" for e in d["escalation_events"]) or "none"

    lines = [
        bar,
        f"  {banner}",
    ]
    if code:
        lines.append(f"  {code}")
    lines += [
        bar,
        f"Turns total      : {d['turns_total']}  (voice: {d['voice_turns']}, text: {d['text_turns']})",
        f"Wellbeing trend  : {str(d['trajectory']).upper()}",
        f"Lowest tier      : {d['lowest_tier']}",
        f"At-risk alerts   : {d['at_risk_alerts']}   turns={d['at_risk_turns']}",
        f"Categories       : {cats}",
        f"Escalations      : {len(d['escalation_events'])}   {esc}",
        f"Risk score       : {d['risk_score']}/100",
        f"Recommended      : {d['recommended_action']}",
        bar,
    ]
    text = "\n".join(lines)
    if do_print:
        print(text)
    return text


def narrative(state, max_words: int = 60) -> str:
    """An LLM-written prose summary of the session (optional, needs Ollama)."""
    from core.llm import summarize
    convo = "\n".join(f"{t['source']}: {t['text']}" for t in state.turns)
    return summarize(convo, max_words=max_words, audience="a university counsellor")

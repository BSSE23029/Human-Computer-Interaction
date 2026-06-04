"""
text/trajectory.py -- session-level analysis: trend, transitions, escalations.

    compute_trajectory(wellbeing_log)
        -> {'trend':'declining', 'lowest_tier':'CRISIS', 'at_risk_turns':[4,5], ...}

    detect_transition(support_log, turn_number)
        -> {'prev':'ACADEMIC','curr':'WELLBEING','turn':4,'is_escalation':True} or None

    log_transition(transition_log, prev, curr, turn)
        -> appends + returns the event dict, or None if no change

    frequency_rank(['A','B','A'])  -> [('A',2),('B',1)]
"""

from collections import Counter
from core.conf import get, active_categories


def compute_trajectory(wellbeing_log: list) -> dict:
    """Compare first-half vs second-half mean score → improving/declining/fluctuating.

    Thresholds come from config (trajectory.improving_threshold / declining_threshold).
    Method 'halves' (default) matches the NEXUS exam algorithm exactly.
    Method 'slope' uses linear regression sign as an alternative.

    Returns:
        trend         : 'improving' | 'declining' | 'fluctuating'
        lowest_tier   : tier name of the worst turn
        at_risk_turns : list of 0-based indices where is_at_risk was True
        first_avg     : mean score of first half
        second_avg    : mean score of second half
    """
    scores = [w.get("score", 0.0) for w in (wellbeing_log or [])]
    n = len(scores)

    if n == 0:
        return {"trend": "unknown", "lowest_tier": None,
                "at_risk_turns": [], "first_avg": 0.0, "second_avg": 0.0}

    method   = get("trajectory.method", "halves")
    imp_thr  = float(get("trajectory.improving_threshold", 0.10))
    dec_thr  = float(get("trajectory.declining_threshold", 0.10))

    if method == "slope":
        trend = _slope_trend(scores, imp_thr, dec_thr)
        first_avg = sum(scores[:n//2 or 1]) / max(n//2, 1)
        second_avg = sum(scores[n//2:]) / max(n - n//2, 1)
    else:
        # "halves" — NEXUS default
        half = n // 2
        first_half  = scores[:half] if half > 0 else scores[:1]
        second_half = scores[half:] if half > 0 else scores[-1:]
        first_avg  = sum(first_half)  / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        if second_avg > first_avg + imp_thr:
            trend = "improving"
        elif second_avg < first_avg - dec_thr:
            trend = "declining"
        else:
            trend = "fluctuating"

    # lowest tier = turn with worst (most negative) score
    worst = min(wellbeing_log, key=lambda w: w.get("score", 0.0))
    at_risk = [i for i, w in enumerate(wellbeing_log) if w.get("is_at_risk")]

    return {
        "trend":        trend,
        "lowest_tier":  worst.get("tier", "UNKNOWN"),
        "at_risk_turns": at_risk,
        "first_avg":    round(first_avg, 3),
        "second_avg":   round(second_avg, 3),
    }


def _slope_trend(scores: list, imp_thr: float, dec_thr: float) -> str:
    """Linear regression on scores list → sign of slope → trend."""
    n = len(scores)
    if n < 2:
        return "fluctuating"
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(scores) / n
    numerator   = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, scores))
    denominator = sum((x - x_mean) ** 2 for x in xs) or 1e-9
    slope = numerator / denominator

    if slope > imp_thr / n:
        return "improving"
    if slope < -dec_thr / n:
        return "declining"
    return "fluctuating"


def detect_transition(support_log: list, turn_number: int):
    """Compare last two entries in support_log. Returns a transition event dict
    if the primary category changed, else None.

    is_escalation: True if the new primary category has escalation_target=True in config.
    """
    if not support_log or len(support_log) < 2:
        return None
    prev = support_log[-2].get("primary")
    curr = support_log[-1].get("primary")
    if prev == curr or curr is None:
        return None
    return _make_event(prev, curr, turn_number)


def log_transition(transition_log: list,
                   prev_primary: str,
                   new_primary: str,
                   turn_number: int):
    """Explicit-args version (NEXUS `log_support_transition` style).
    Appends to transition_log and returns the event, or None if no change.
    """
    if not new_primary or prev_primary == new_primary:
        return None
    event = _make_event(prev_primary or "NONE", new_primary, turn_number)
    transition_log.append(event)
    return event


def _make_event(prev: str, curr: str, turn: int) -> dict:
    """Build a transition event dict. escalation_target is read from config."""
    cats = active_categories()
    curr_cfg = cats.get(curr, {})
    is_esc = bool(curr_cfg.get("escalation_target", False))
    return {"prev": prev, "curr": curr, "turn": turn, "is_escalation": is_esc}


def frequency_rank(values) -> list:
    """[('ACADEMIC',3),('WELLBEING',2),...] sorted by count desc."""
    return Counter(v for v in values if v).most_common()


def frequency_rank_log(log: list, key: str) -> list:
    """frequency_rank on one field across a list of dicts (e.g. key='primary')."""
    return frequency_rank([d.get(key) for d in (log or [])])


if __name__ == "__main__":
    # simulate NEXUS student log scores
    scores = [0.0, -0.15, -0.35, -0.55, -0.65, -0.5, -0.35, -0.3, 0.05, 0.2]
    wl = [{"score": s, "tier": "X", "is_at_risk": s < -0.6} for s in scores]
    t = compute_trajectory(wl)
    print("trajectory:", t)

    sl = [{"primary": "ACADEMIC"}, {"primary": "WELLBEING"}, {"primary": "WELLBEING"}]
    tr = detect_transition(sl, turn_number=2)
    print("transition:", tr)

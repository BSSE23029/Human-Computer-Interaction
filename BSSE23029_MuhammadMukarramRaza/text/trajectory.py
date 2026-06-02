"""
text/trajectory.py -- session-level analysis: trend, transitions, frequency.

    compute_trajectory(wellbeing_log)
        -> {'trend':'declining', 'lowest_tier':'CRISIS', 'at_risk_turns':[4,5]}
    detect_transition(support_log, turn_number)   # compares last two turns
        -> {'prev':'ACADEMIC','curr':'WELLBEING','turn':4,'is_escalation':True} or None
    frequency_rank(['A','B','A'])                  -> [('A',2),('B',1)]
"""
from collections import Counter

from core.conf import get


def compute_trajectory(wellbeing_log: list) -> dict:
    """Compare first-half vs second-half mean score -> improving/declining/fluctuating.
    Also report the lowest tier reached and the 0-based at-risk turn indices.
    (NEXUS Q2.2 equivalent.)"""
    scores = [w.get("score", 0.0) for w in (wellbeing_log or [])]
    n = len(scores)
    if n == 0:
        return {"trend": "unknown", "lowest_tier": None, "at_risk_turns": []}

    half = n // 2
    first = scores[:half] if half else scores[:1]
    second = scores[half:] if half else scores[-1:]
    first_avg = sum(first) / len(first)
    second_avg = sum(second) / len(second)

    if second_avg > first_avg + 0.1:
        trend = "improving"
    elif second_avg < first_avg - 0.1:
        trend = "declining"
    else:
        trend = "fluctuating"

    lowest = min(wellbeing_log, key=lambda w: w.get("score", 0.0))
    at_risk_turns = [i for i, w in enumerate(wellbeing_log) if w.get("is_at_risk")]
    return {"trend": trend, "lowest_tier": lowest.get("tier"),
            "at_risk_turns": at_risk_turns,
            "first_avg": round(first_avg, 3), "second_avg": round(second_avg, 3)}


def detect_transition(support_log: list, turn_number: int) -> dict:
    """Compare the last two per-turn classify dicts in `support_log`.
    Returns a transition event if the primary category changed, else None.
    is_escalation=True if the new category is in transitions.escalation_into."""
    if not support_log or len(support_log) < 2:
        return None
    prev = support_log[-2].get("primary")
    curr = support_log[-1].get("primary")
    if prev == curr:
        return None
    escalate_into = set(get("transitions.escalation_into") or [])
    return {"prev": prev, "curr": curr, "turn": turn_number,
            "is_escalation": curr in escalate_into}


def log_transition(transition_log: list, prev_primary: str, new_primary: str,
                   turn_number: int) -> dict:
    """Explicit-args variant (NEXUS `log_support_transition` style): append an event
    to `transition_log` if the category changed. Returns the event or None."""
    if prev_primary == new_primary or new_primary is None:
        return None
    escalate_into = set(get("transitions.escalation_into") or [])
    event = {"prev": prev_primary, "curr": new_primary, "turn": turn_number,
             "is_escalation": new_primary in escalate_into}
    transition_log.append(event)
    return event


def frequency_rank(values) -> list:
    """[('ACADEMIC',3),('WELLBEING',2),...] sorted by count desc."""
    return Counter(v for v in values if v).most_common()


def frequency_rank_log(log: list, key: str) -> list:
    """Frequency-rank one field across a list of dicts (e.g. key='primary')."""
    return frequency_rank([d.get(key) for d in (log or [])])

"""
text/scale.py -- map a sentiment score onto the configured N-tier scale.

enabled: true  → use the tier table from config
enabled: false → ask llama3 to name the emotional state directly

    to_tier(-0.7)        -> {'tier':'CRISIS','score':-0.7,'emoji':'🆘','is_at_risk':True}
    assess("I'm great")  -> {'tier':'CONTENT', ...}    # = NEXUS assess_wellbeing()
    check_and_alert(result, turn=3)                    # prints alert if at-risk
"""

from core.conf import get, enabled, tier_behavior
from text.sentiment import score_text, score_llm


def to_tier(score: float) -> dict:
    """Map a compound score to a tier dict.

    Reads tier rows [name, low, high, emoji] from config.
    is_at_risk is read from scale.behavior.<name>.is_at_risk (not a global list).

    Fallback chain:
      - if no tier matches (should never happen with -100/+100 bounds) → UNKNOWN
    """
    if not enabled("scale"):
        # scale section disabled — return a minimal dict with neutral defaults
        return {"tier": "UNKNOWN", "score": round(float(score), 4),
                "emoji": "❓", "is_at_risk": False}

    tiers = get("scale.tiers") or []
    score = float(score)

    for row in tiers:
        try:
            name, low, high, emoji = row[0], float(row[1]), float(row[2]), row[3]
        except (IndexError, ValueError, TypeError):
            continue
        if low <= score < high:
            beh = tier_behavior(name)
            return {
                "tier":       name,
                "score":      round(score, 4),
                "emoji":      emoji,
                "is_at_risk": bool(beh.get("is_at_risk", False)),
            }

    # nothing matched — use last tier as worst-case fallback
    if tiers:
        try:
            name, _, _, emoji = tiers[-1]
            beh = tier_behavior(name)
            return {"tier": name, "score": round(score, 4), "emoji": emoji,
                    "is_at_risk": bool(beh.get("is_at_risk", False))}
        except Exception:
            pass

    return {"tier": "UNKNOWN", "score": round(score, 4), "emoji": "❓", "is_at_risk": False}


def assess(text: str, use_llm: bool = False) -> dict:
    """Full pipeline: text → sentiment score → tier dict.
    This is the NEXUS `assess_wellbeing()` equivalent.

    use_llm=True  → skip lexicon, ask llama3 for the compound score
    use_llm=False → use the lexicon (default, fast, offline)

    If scale.enabled=False the LLM is asked to name the tier directly
    (via llm.classify against the tier names).
    """
    if not enabled("scale"):
        # scale fully disabled — ask llama3 to name the tier directly
        try:
            from core.llm import classify as llm_classify
            tier_names = [row[0] for row in (get("scale.tiers") or [])]
            if not tier_names:
                tier_names = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
            tier_name = llm_classify(text, tier_names)
            return {"tier": tier_name, "score": 0.0, "emoji": "❓", "is_at_risk": False}
        except Exception:
            return {"tier": "NEUTRAL", "score": 0.0, "emoji": "😐", "is_at_risk": False}

    # sentiment.enabled=False  →  use LLM scorer instead of lexicon
    use_llm_score = use_llm or not enabled("sentiment")
    score = score_llm(text) if use_llm_score else score_text(text)
    return to_tier(score)


def check_and_alert(wellbeing_result: dict, turn_number: int) -> bool:
    """Print a console alert if the tier's behavior says show=True.
    Always fires for at-risk tiers regardless of the show flag.
    Returns the is_at_risk boolean.
    (NEXUS Q2.3 `check_and_alert()` equivalent.)
    """
    tier_name = wellbeing_result.get("tier", "")
    is_at_risk = bool(wellbeing_result.get("is_at_risk", False))
    beh = tier_behavior(tier_name)
    alert_cfg = beh.get("alert", {})

    should_show = bool(alert_cfg.get("show", False)) or is_at_risk

    if should_show:
        msg_template = alert_cfg.get("message", "Alert — Turn {turn}: {tier}.")
        msg = msg_template.format(turn=turn_number, tier=tier_name)
        color = alert_cfg.get("color", "red")

        # simple ANSI colour — works in most terminals
        _ANSI = {"red": "\033[91m", "orange": "\033[93m",
                 "yellow": "\033[93m", "green": "\033[92m",
                 "blue": "\033[94m", "reset": "\033[0m"}
        c = _ANSI.get(color, "")
        r = _ANSI["reset"]
        print(f"\n{c}{'!' * 55}{r}")
        print(f"{c}  {msg}{r}")
        print(f"{c}{'!' * 55}{r}\n")

    return is_at_risk


if __name__ == "__main__":
    tests = [
        ("I feel completely hopeless and alone", False),
        ("Things are getting better, I feel calm", False),
        ("I am so stressed about my exam", False),
        ("Hello, I need some help", False),
    ]
    for text, use_llm in tests:
        r = assess(text, use_llm)
        flag = check_and_alert(r, turn_number=1)
        print(f"{r['emoji']} {r['tier']:12} score={r['score']:+.3f}  at_risk={flag}  | {text}")

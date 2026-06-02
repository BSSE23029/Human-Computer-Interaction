"""
text/scale.py -- map a sentiment score onto the configured N-tier scale.

The tier list lives in config (scale.tiers), so you can switch a 6-tier
wellbeing scale to a 3-tier satisfaction scale by editing YAML only.

    to_tier(-0.7)        -> {'tier':'CRISIS','score':-0.7,'emoji':'🆘','is_at_risk':True}
    assess("I'm great")  -> {'tier':'CONTENT', ...}        # = NEXUS assess_wellbeing()
    assess(text, use_llm=True)                              # score via llama3 instead
"""
from core.conf import get
from text.sentiment import score_text, score_llm


def to_tier(score: float) -> dict:
    """Find the tier whose [low, high) range contains `score`."""
    tiers = get("scale.tiers") or []
    at_risk = set(get("scale.at_risk_tiers") or [])
    score = float(score)
    for row in tiers:
        name, low, high, emoji = row[0], float(row[1]), float(row[2]), row[3]
        if low <= score < high:
            return {"tier": name, "score": round(score, 4), "emoji": emoji,
                    "is_at_risk": name in at_risk}
    # Fallback: worst (last) tier if nothing matched.
    if tiers:
        name, _, _, emoji = tiers[-1]
        return {"tier": name, "score": round(score, 4), "emoji": emoji,
                "is_at_risk": name in at_risk}
    return {"tier": "UNKNOWN", "score": round(score, 4), "emoji": "", "is_at_risk": False}


def assess(text: str, use_llm: bool = False) -> dict:
    """Sentiment -> tier. This is the NEXUS `assess_wellbeing()` equivalent."""
    score = score_llm(text) if use_llm else score_text(text)
    return to_tier(score)


def check_and_alert(wellbeing_result: dict, turn_number: int) -> bool:
    """Print a prominent alert if the result is at-risk. Returns the flag.
    (NEXUS Q2.3 equivalent.)"""
    flag = bool(wellbeing_result.get("is_at_risk"))
    if flag:
        print("\n" + "!" * 55)
        print(f"  AT-RISK ALERT  (turn {turn_number})")
        print(f"  Tier: {wellbeing_result.get('tier')}  {wellbeing_result.get('emoji','')}")
        print("  Please contact the university counselling line immediately.")
        print("!" * 55 + "\n")
    return flag

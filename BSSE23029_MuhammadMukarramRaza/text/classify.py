"""
text/classify.py -- keyword classifiers (multi-label categories + single-label intent).

categories.enabled: true  → keyword matching
categories.enabled: false → llama3 classifies instead

    classify("I failed my exam and can't pay rent")
        -> {'primary':'ACADEMIC', 'all_detected':['ACADEMIC','FINANCIAL'],
            'scores':{'ACADEMIC':1,'FINANCIAL':1,...}}

    classify_intent("hi there")   -> 'greeting'
    classify_llm("...", labels)   -> one label via llama3
    classify_llm_multi("...", labels) -> all matching labels via llama3
"""

from core.conf import get, enabled, active_categories
from text.preprocess import tokenize, count_keywords


def _score_categories(text: str) -> dict:
    """Count keyword hits per ENABLED category. Returns {cat_name: hit_count}.
    Uses the scoring_method from config (default 'count').
    """
    cats = active_categories()           # only enabled categories, no 'enabled' key
    method = get("categories.scoring_method", "count")
    scores = {}

    for cat_name, cfg in cats.items():
        keywords = cfg.get("keywords") or []
        hits = count_keywords(text, keywords)

        if method == "presence":
            scores[cat_name] = 1 if hits > 0 else 0
        else:
            scores[cat_name] = hits   # default: raw count

    return scores


def _priority(cat_name: str) -> int:
    """Response priority for a category (higher = more dominant)."""
    return get(f"categories.{cat_name}.response_priority", 1)


def classify(text: str) -> dict:
    """Multi-label category classifier.
    Returns:
        primary      : the dominant category (highest score, tie broken by priority)
        all_detected : all categories with score > 0, sorted by score desc
        scores       : raw score dict for all categories

    If categories.enabled=False → delegates to llama3.
    (NEXUS `classify_support_need()` equivalent.)
    """
    if not enabled("categories"):
        return _classify_fallback_llm(text)

    scores = _score_categories(text)
    detected = [c for c, v in scores.items() if v > 0]

    # sort by score desc, then by priority desc as tie-breaker
    detected.sort(key=lambda c: (scores[c], _priority(c)), reverse=True)

    primary = detected[0] if detected else "GENERAL"
    return {"primary": primary, "all_detected": detected, "scores": scores}


def _classify_fallback_llm(text: str) -> dict:
    """When categories.enabled=False — ask llama3 to classify."""
    try:
        from core.llm import classify as llm_classify, multi_classify
        cats = list(active_categories().keys())
        if not cats:
            cats = ["GENERAL"]
        primary = llm_classify(text, cats)
        all_det = multi_classify(text, cats)
        if primary not in all_det:
            all_det.insert(0, primary)
        return {"primary": primary, "all_detected": all_det,
                "scores": {c: (1 if c in all_det else 0) for c in cats}}
    except Exception:
        return {"primary": "GENERAL", "all_detected": [], "scores": {}}


def classify_intent(text: str) -> str:
    """Single-label intent classifier using the intents keyword map.
    Returns 'unknown' if nothing matches.

    The intents section is a separate, simpler map:
        greeting: [hi, hello, hey]
        question: [how, what, why, ...]
    """
    mapping = get("intents") or {}
    if not mapping:
        return "unknown"

    best_cat, best_count = "unknown", 0
    for intent, keywords in mapping.items():
        hits = count_keywords(text, keywords)
        if hits > best_count:
            best_count, best_cat = hits, intent

    return best_cat


def classify_llm(text: str, labels: list = None) -> str:
    """Single label chosen by llama3. Use when keywords are too brittle."""
    from core.llm import classify as _c
    labels = labels or list(active_categories().keys()) or ["GENERAL"]
    return _c(text, labels)


def classify_llm_multi(text: str, labels: list = None) -> list:
    """All applicable labels chosen by llama3."""
    from core.llm import multi_classify
    labels = labels or list(active_categories().keys()) or ["GENERAL"]
    return multi_classify(text, labels)


if __name__ == "__main__":
    tests = [
        "I have a major assignment due tomorrow",
        "I cannot afford my rent and my fees are overdue",
        "I feel completely hopeless and I cannot sleep",
        "The student portal won't let me log in",
        "I feel very alone and I don't have any friends here",
        "Hi I need some help please",
    ]
    for t in tests:
        r = classify(t)
        print(f"{r['primary']:12}  all={r['all_detected']}  | {t[:55]}")

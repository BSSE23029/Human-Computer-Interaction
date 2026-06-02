"""
text/classify.py -- keyword classifiers (multi-label categories + single-label intent),
plus an LLM-backed alternative.

    classify("I failed my exam and can't pay rent")
        -> {'primary':'ACADEMIC', 'all_detected':['ACADEMIC','FINANCIAL'],
            'scores':{'ACADEMIC':1,'FINANCIAL':1,...}}
    classify_intent("hi there")        -> 'greeting'
    classify_llm("...", labels=[...])  -> one label, decided by llama3
"""
from core.conf import get
from text.preprocess import tokenize


def _score_map(text: str, mapping: dict) -> dict:
    """Count keyword hits per category. Single alpha words match whole tokens;
    multi-word / punctuation keywords (e.g. 'good morning', '?') match substrings."""
    toks = set(tokenize(text))
    raw = (text or "").lower()
    scores = {}
    for cat, kws in (mapping or {}).items():
        c = 0
        for kw in kws:
            k = str(kw).lower()
            if k.isalpha():
                c += 1 if k in toks else 0
            else:
                c += 1 if k in raw else 0
        scores[cat] = c
    return scores


def classify(text: str) -> dict:
    """Multi-label category classifier. Returns primary + all_detected + scores.
    (NEXUS `classify_support_need()` equivalent.)"""
    mapping = get("categories") or {}
    scores = _score_map(text, mapping)
    detected = [c for c, v in scores.items() if v > 0]
    detected.sort(key=lambda c: scores[c], reverse=True)
    primary = detected[0] if detected else "GENERAL"
    return {"primary": primary, "all_detected": detected, "scores": scores}


def classify_intent(text: str) -> str:
    """Single-label intent from the `intents` config map."""
    mapping = get("intents") or {}
    scores = _score_map(text, mapping)
    if not scores:
        return "unknown"
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def classify_llm(text: str, labels: list = None) -> str:
    """Single label chosen by llama3 (use when keywords are too brittle)."""
    from core.llm import classify as _c
    labels = labels or list((get("categories") or {}).keys())
    return _c(text, labels)


def classify_llm_multi(text: str, labels: list = None) -> list:
    """All applicable labels chosen by llama3."""
    from core.llm import multi_classify
    labels = labels or list((get("categories") or {}).keys())
    return multi_classify(text, labels)

"""
text/sentiment.py -- VADER-free sentiment (nltk is NOT in the install list).

Produces a 'compound'-style score in [-1, 1] from the config lexicon, with
booster and negation handling, then a VADER-shaped dict so any task that
expects sia.polarity_scores()-style keys still works.

    score_text("I feel hopeless")      -> -0.62
    analyze("I feel hopeless")          -> {'compound': -0.62, 'pos':0.0, 'neg':0.33, 'neu':0.67}
    score_llm("I feel hopeless")        -> uses llama3 instead of the lexicon
"""
import math

from core.conf import get
from text.preprocess import tokenize


def score_text(text: str) -> float:
    """Lexicon sentiment -> compound score in [-1, 1].
    Looks back up to 3 tokens for boosters (scale) and negations (flip)."""
    pos = {str(k).lower(): float(v) for k, v in (get("sentiment.positive_words") or {}).items()}
    neg = {str(k).lower(): float(v) for k, v in (get("sentiment.negative_words") or {}).items()}
    boosters = {str(k).lower(): float(v) for k, v in (get("sentiment.boosters") or {}).items()}
    negations = set(str(w).lower() for w in (get("sentiment.negations") or []))

    toks = tokenize(text)
    total = 0.0
    for i, tok in enumerate(toks):
        val = pos.get(tok, 0.0) - neg.get(tok, 0.0)
        if val == 0.0:
            continue
        window = toks[max(0, i - 3):i]
        for w in window:
            if w in boosters:
                val *= boosters[w]
        if any(w in negations for w in window):
            val = -val * 0.74   # negation flips and slightly dampens (VADER-like)
        total += val

    if total == 0.0:
        return 0.0
    compound = total / math.sqrt(total * total + 15.0)   # squash to (-1, 1)
    return round(max(-1.0, min(1.0, compound)), 4)


def analyze(text: str) -> dict:
    """VADER-shaped output: {'compound', 'pos', 'neg', 'neu'} (props sum ~1)."""
    pos = set(k.lower() for k in (get("sentiment.positive_words") or {}))
    neg = set(k.lower() for k in (get("sentiment.negative_words") or {}))
    toks = tokenize(text)
    n = max(len(toks), 1)
    p = sum(1 for t in toks if t in pos)
    ng = sum(1 for t in toks if t in neg)
    neu = max(0, n - p - ng)
    return {
        "compound": score_text(text),
        "pos": round(p / n, 3),
        "neg": round(ng / n, 3),
        "neu": round(neu / n, 3),
    }


def score_llm(text: str) -> float:
    """Sentiment via llama3 (use when the lexicon is too shallow for the domain)."""
    from core.llm import score
    return score(text, lo=-1.0, hi=1.0,
                 criterion="emotional sentiment where -1 is severe distress and +1 is very positive")

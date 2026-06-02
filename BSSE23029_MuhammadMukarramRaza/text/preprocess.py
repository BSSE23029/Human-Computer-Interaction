"""
text/preprocess.py -- small text helpers questions tend to assume you have.
Pure standard library (no nltk -- it is not in the install list).
"""
import re


def tokenize(text: str) -> list:
    """Lowercase word tokens, keeping apostrophes (so "can't" stays one token)."""
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def normalize(text: str) -> str:
    """Lowercase + trim."""
    return (text or "").lower().strip()


def word_count(text: str) -> int:
    """Whitespace word count."""
    return len((text or "").split())


def sentences(text: str) -> list:
    """Split into sentences on . ! ? boundaries."""
    return [s.strip() for s in re.split(r"[.!?]+", text or "") if s.strip()]


def count_keywords(text: str, keywords) -> int:
    """How many of `keywords` appear. Single alpha words match whole tokens;
    multi-word / punctuation keywords match as substrings."""
    toks = set(tokenize(text))
    raw = (text or "").lower()
    n = 0
    for kw in keywords:
        k = str(kw).lower()
        if k.isalpha():
            n += 1 if k in toks else 0
        else:
            n += 1 if k in raw else 0
    return n

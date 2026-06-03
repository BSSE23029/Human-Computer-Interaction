"""
vowel_classifier.py — Fuzzy-membership vowel classifier with temporal voting.

Architecture
------------
1. Gaussian membership function per (vowel, feature) pair.
2. Weighted product aggregation → confidence score per vowel.
3. Ring buffer of last N predictions → majority vote for stability.
4. Optional calibration: records neutral position and adjusts corner_lift offset.
"""

import math
import numpy as np
from collections import deque
from config import (
    VOWEL_PROFILES, FEATURE_WEIGHTS,
    VOWEL_HISTORY_LEN, VOWEL_CONFIDENCE_MIN,
)


def _gaussian(x: float, mu: float, sigma: float) -> float:
    """Normalised Gaussian membership ∈ [0, 1]."""
    return math.exp(-0.5 * ((x - mu) / (sigma + 1e-9)) ** 2)


class VowelClassifier:

    VOWELS = ["A", "E", "I", "O", "U"]

    def __init__(self):
        self._history: deque = deque(maxlen=VOWEL_HISTORY_LEN)
        self._conf_history: dict = {v: deque(maxlen=VOWEL_HISTORY_LEN)
                                    for v in self.VOWELS}
        # Calibration offset for corner_lift (set during neutral recording)
        self._corner_lift_offset = 0.0
        self._calibrating        = False
        self._calib_samples: list = []

    # ─── Public API ──────────────────────────────────────────────────────────
    def classify(self, features: dict) -> tuple[str, float, dict]:
        """
        Parameters
        ----------
        features : dict from FeatureExtractor.extract()

        Returns
        -------
        (vowel, confidence, all_scores)
            vowel      : "A"|"E"|"I"|"O"|"U"|"NONE"
            confidence : float [0, 1]
            all_scores : dict {vowel: score}
        """
        # Apply calibration
        feat = dict(features)
        feat["corner_lift"] -= self._corner_lift_offset

        scores = self._score_all(feat)
        best_v, best_s = max(scores.items(), key=lambda kv: kv[1])

        # Update history
        voted_label = best_v if best_s >= VOWEL_CONFIDENCE_MIN else "NONE"
        self._history.append(voted_label)
        for v in self.VOWELS:
            self._conf_history[v].append(scores[v])

        # Majority-vote label
        if len(self._history) == 0:
            return "NONE", 0.0, scores

        counts = {v: self._history.count(v) for v in self.VOWELS + ["NONE"]}
        stable_label = max(counts, key=counts.get)

        # Smoothed confidence = mean of last N for stable label
        if stable_label in self.VOWELS:
            smooth_conf = float(np.mean(self._conf_history[stable_label]))
        else:
            smooth_conf = 0.0

        return stable_label, smooth_conf, scores

    def start_calibration(self):
        """Call to begin recording neutral-face corner_lift samples."""
        self._calibrating   = True
        self._calib_samples = []

    def finish_calibration(self):
        """Stop recording and set offset. Returns measured offset."""
        self._calibrating = False
        if self._calib_samples:
            self._corner_lift_offset = float(np.median(self._calib_samples))
        return self._corner_lift_offset

    def feed_calibration(self, features: dict):
        if self._calibrating:
            self._calib_samples.append(features["corner_lift"])

    @property
    def is_calibrating(self):
        return self._calibrating

    # ─── Internal ─────────────────────────────────────────────────────────────
    def _score_all(self, feat: dict) -> dict:
        scores = {}
        for vowel, profile in VOWEL_PROFILES.items():
            log_score = 0.0
            for fname, (mu, sigma) in profile.items():
                m = _gaussian(feat[fname], mu, sigma)
                w = FEATURE_WEIGHTS.get(fname, 1.0)
                # Log-space to avoid underflow with many features
                log_score += w * math.log(m + 1e-12)
            scores[vowel] = log_score

        # Softmax normalisation → [0, 1]
        vals    = np.array([scores[v] for v in self.VOWELS])
        vals   -= vals.max()          # numerical stability
        exps    = np.exp(vals)
        softmax = exps / exps.sum()
        return {v: float(softmax[i]) for i, v in enumerate(self.VOWELS)}

"""
core/state.py -- the session container.

One SessionState lives for the duration of a conversation.
Every module writes into it; report.py reads from it at the end.

    s = SessionState()

    # each turn:
    s.add_turn("I failed my exam", source="text")
    s.log_wellbeing({"tier": "STRESSED", "score": -0.3, "emoji": "😟", "is_at_risk": False})
    s.log_support({"primary": "ACADEMIC", "all_detected": ["ACADEMIC"], "scores": {}})
    s.log_transition({"prev": "NONE", "curr": "ACADEMIC", "turn": 1, "is_escalation": False})
    s.log_vision({"face_present": True, "mood": "neutral", "gesture": "None", "head_zone": "Forward"})
    s.add_response("Exam pressure is real. Have you spoken to your tutor?")

    # end of session:
    from text.report import generate_report
    generate_report(s)
"""

from core.conf import get


class SessionState:
    """Accumulates all turn data across one conversation session."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear everything — called at __init__ and to start a new session."""
        self.turns          = []   # [{text, source, turn, word_count, ...}]
        self.wellbeing_log  = []   # [{tier, score, emoji, is_at_risk}]
        self.support_log    = []   # [{primary, all_detected, scores}]
        self.transition_log = []   # [{prev, curr, turn, is_escalation}]
        self.vision_log     = []   # [{face_present, mood, gesture, head_zone, ...}]
        self.responses      = []   # assistant reply strings, one per turn

    # ── adding data ───────────────────────────────────────────────────────────
    def add_turn(
        self,
        text: str,
        source: str = "text",
        word_count: int = None,
        **extra,
    ) -> dict:
        """Record one user turn. `source` is 'text' or 'voice'.
        Any extra kwargs are stored in the turn dict (e.g. confidence, timestamp).
        Returns the completed turn dict.
        """
        turn = {
            "text":       text,
            "source":     source,
            "turn":       len(self.turns) + 1,
            "word_count": word_count if word_count is not None else len((text or "").split()),
        }
        turn.update(extra)
        self.turns.append(turn)
        return turn

    def log_wellbeing(self, result: dict) -> None:
        """Store one tier assessment result (output of text/scale.assess)."""
        self.wellbeing_log.append(result or {})

    def log_support(self, result: dict) -> None:
        """Store one category classification result (output of text/classify.classify)."""
        self.support_log.append(result or {})

    def log_transition(self, event: dict) -> None:
        """Store a category transition event (output of text/trajectory.log_transition).
        Only appended when the primary category actually changed."""
        if event:
            self.transition_log.append(event)

    def log_vision(self, labels: dict) -> None:
        """Store the vision label dict for this turn (from vision/bridge.py)."""
        self.vision_log.append(labels or {})

    def add_response(self, text: str) -> None:
        """Store the assistant's reply for this turn."""
        self.responses.append(text or "")

    # ── convenience properties ─────────────────────────────────────────────────
    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def voice_turns(self) -> int:
        return sum(1 for t in self.turns if t.get("source") == "voice")

    @property
    def text_turns(self) -> int:
        return sum(1 for t in self.turns if t.get("source") == "text")

    @property
    def avg_words(self) -> float:
        if not self.turns:
            return 0.0
        return sum(t.get("word_count", 0) for t in self.turns) / len(self.turns)

    @property
    def at_risk_count(self) -> int:
        return sum(1 for w in self.wellbeing_log if w.get("is_at_risk"))

    @property
    def at_risk_turns(self) -> list:
        """0-based indices of turns where is_at_risk was True."""
        return [i for i, w in enumerate(self.wellbeing_log) if w.get("is_at_risk")]

    @property
    def escalation_count(self) -> int:
        return sum(1 for e in self.transition_log if e.get("is_escalation"))

    @property
    def escalation_events(self) -> list:
        return [e for e in self.transition_log if e.get("is_escalation")]

    @property
    def sentiment_scores(self) -> list:
        """All compound scores from wellbeing_log (floats)."""
        return [w.get("score", 0.0) for w in self.wellbeing_log if "score" in w]

    @property
    def avg_sentiment(self) -> float:
        scores = self.sentiment_scores
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def tier_counts(self) -> dict:
        """How many turns each tier appeared. e.g. {"STRESSED": 3, "CRISIS": 1}"""
        counts = {}
        for w in self.wellbeing_log:
            t = w.get("tier", "UNKNOWN")
            counts[t] = counts.get(t, 0) + 1
        return counts

    @property
    def lowest_tier(self) -> str:
        """The worst tier reached this session (based on lowest sentiment score)."""
        if not self.wellbeing_log:
            return "UNKNOWN"
        worst = min(self.wellbeing_log, key=lambda w: w.get("score", 0.0))
        return worst.get("tier", "UNKNOWN")

    @property
    def category_counts(self) -> dict:
        """How many turns each category appeared as primary.
        e.g. {"ACADEMIC": 4, "WELLBEING": 3}"""
        counts = {}
        for s in self.support_log:
            cat = s.get("primary")
            if cat:
                counts[cat] = counts.get(cat, 0) + 1
        return counts

    @property
    def dominant_category(self) -> str:
        """Category that appeared most as primary. 'UNKNOWN' if no turns yet."""
        counts = self.category_counts
        if not counts:
            return "UNKNOWN"
        return max(counts, key=counts.get)

    # ── LLM context builder ───────────────────────────────────────────────────
    def messages_for_llm(self, system: str = None) -> list:
        """Build an openai-style messages list from the full conversation history.
        Pairs each user turn with the assistant reply at the same index.

        Pass this to llm.chat_messages() to give llama3 memory of the whole session.
        context_window from config is applied inside llm.chat_messages — no need here.
        """
        from core.conf import get as _get
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        for i, turn in enumerate(self.turns):
            msgs.append({"role": "user", "content": turn["text"]})
            if i < len(self.responses):
                msgs.append({"role": "assistant", "content": self.responses[i]})
        return msgs

    def conversation_text(self) -> str:
        """Flat text version of the conversation (for report narrative prompt)."""
        lines = []
        for i, turn in enumerate(self.turns):
            lines.append(f"[Turn {turn['turn']} | {turn['source']}] User: {turn['text']}")
            if i < len(self.responses):
                lines.append(f"  Assistant: {self.responses[i]}")
        return "\n".join(lines)

    def tier_summary_text(self) -> str:
        """One-line tier summary for the report prompt.
        e.g. 'STRESSED x3, CRISIS x1, NEUTRAL x2'
        """
        counts = self.tier_counts
        if not counts:
            return "no data"
        return ", ".join(f"{t} x{n}" for t, n in sorted(counts.items(), key=lambda x: -x[1]))

    # ── snapshot for debugging ────────────────────────────────────────────────
    def summary(self) -> dict:
        """Compact dict snapshot — useful for quick display or debugging."""
        return {
            "turns":          self.n_turns,
            "voice_turns":    self.voice_turns,
            "text_turns":     self.text_turns,
            "avg_words":      round(self.avg_words, 1),
            "avg_sentiment":  round(self.avg_sentiment, 3),
            "lowest_tier":    self.lowest_tier,
            "at_risk_count":  self.at_risk_count,
            "at_risk_turns":  self.at_risk_turns,
            "escalations":    self.escalation_count,
            "tier_counts":    self.tier_counts,
            "category_counts":self.category_counts,
        }

    def __repr__(self):
        return (f"<SessionState turns={self.n_turns} "
                f"at_risk={self.at_risk_count} "
                f"escalations={self.escalation_count}>")

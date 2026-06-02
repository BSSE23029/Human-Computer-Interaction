"""
core/state.py -- the session container.

Holds everything that accumulates across a conversation so the report stage
(and trajectory/transition logic) has data to work with.

    s = SessionState()
    s.add_turn("I failed my exam", source="text", word_count=4)
    s.log_wellbeing({"tier": "STRESSED", "score": -0.3, "is_at_risk": False})
    s.log_support({"primary": "ACADEMIC", "all_detected": ["ACADEMIC"]})
    ...
    report = generate_report(s)     # see text/report.py
"""


class SessionState:
    """Accumulates turns + per-turn analysis logs for one session."""

    def __init__(self):
        self.turns = []            # [{text, source, turn, word_count, ...}]
        self.wellbeing_log = []    # [{tier, score, emoji, is_at_risk}]  (from text/scale.assess)
        self.support_log = []      # [{primary, all_detected, scores}]   (from text/classify.classify)
        self.transition_log = []   # [{prev, curr, turn, is_escalation}] (from text/trajectory)
        self.vision_log = []       # [{present, mood, gesture, motion, head_zone}]
        self.responses = []        # assistant replies, in order

    # --- adding data -------------------------------------------------------
    def add_turn(self, text: str, source: str = "text", **meta) -> dict:
        """Record one user turn. `source` is 'text' or 'voice'. Returns the turn dict."""
        turn = {
            "text": text,
            "source": source,
            "turn": len(self.turns) + 1,
            "word_count": meta.pop("word_count", len((text or "").split())),
        }
        turn.update(meta)
        self.turns.append(turn)
        return turn

    def log_wellbeing(self, result: dict):
        self.wellbeing_log.append(result)

    def log_support(self, result: dict):
        self.support_log.append(result)

    def log_transition(self, event: dict):
        if event:
            self.transition_log.append(event)

    def log_vision(self, label: dict):
        self.vision_log.append(label)

    def add_response(self, text: str):
        self.responses.append(text)

    # --- convenience views -------------------------------------------------
    @property
    def voice_turns(self) -> int:
        return sum(1 for t in self.turns if t.get("source") == "voice")

    @property
    def text_turns(self) -> int:
        return sum(1 for t in self.turns if t.get("source") == "text")

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def avg_words(self) -> float:
        if not self.turns:
            return 0.0
        return sum(t.get("word_count", 0) for t in self.turns) / len(self.turns)

    def messages_for_llm(self, system: str = None) -> list:
        """Build an openai-style messages list from the conversation, so the LLM
        can 'remember' the whole session. Pairs each user turn with its reply."""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        for i, t in enumerate(self.turns):
            msgs.append({"role": "user", "content": t["text"]})
            if i < len(self.responses):
                msgs.append({"role": "assistant", "content": self.responses[i]})
        return msgs

    def summary(self) -> dict:
        """A compact snapshot (useful for debugging / quick display)."""
        return {
            "turns": self.n_turns,
            "voice_turns": self.voice_turns,
            "text_turns": self.text_turns,
            "avg_words": round(self.avg_words, 1),
            "at_risk_count": sum(1 for w in self.wellbeing_log if w.get("is_at_risk")),
            "escalations": sum(1 for e in self.transition_log if e.get("is_escalation")),
        }

    def reset(self):
        self.__init__()

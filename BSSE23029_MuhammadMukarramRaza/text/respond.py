"""
text/respond.py -- response generation.

Rule lookup order (for a given category + tier):
  1. "CATEGORY|TIER"  exact match
  2. "CATEGORY|*"     wildcard tier
  3. "CATEGORY"       category-only (no tier needed)
  4. "*"              catch-all (last resort before LLM)
  5. LLM fallback     (if responses.use_llm_fallback=True and Ollama is alive)
  6. responses.default string

responses.enabled=False → skip rule table entirely; everything goes to LLM.
If LLM is also unavailable → responses.default string.

    respond("I can't sleep", "WELLBEING", "DISTRESSED")
    respond_llm("...", context="...")          single-shot LLM reply
    respond_with_memory(state, text)           LLM WITH full conversation history
"""

from core.conf import get, enabled


def _rule_lookup(category: str, tier: str):
    """Try all key patterns in priority order. Returns matched string or None."""
    rules = get("responses.rules") or {}
    for key in (
        f"{category}|{tier}",   # exact match
        f"{category}|*",        # wildcard tier
        f"{category}",          # category only
        "*",                    # catch-all
    ):
        if key in rules:
            return rules[key]
    return None


def respond(text: str, category: str, tier: str) -> str:
    """Main response function.

    1. Rule table (if responses.enabled=True)
    2. LLM fallback (if Ollama is alive and use_llm_fallback=True)
    3. Default string (always available)

    (NEXUS `nexus_respond()` equivalent.)
    """
    # rule table
    if enabled("responses"):
        hit = _rule_lookup(category or "GENERAL", tier or "NEUTRAL")
        if hit:
            return hit

    # LLM fallback
    if get("responses.use_llm_fallback", True):
        try:
            from core.llm import is_alive, chat, build_prompt, persona_system
            if is_alive():
                prompt = build_prompt(
                    "respond",
                    persona=persona_system(),
                    tier=tier or "NEUTRAL",
                    category=category or "GENERAL",
                    text=text,
                )
                reply = chat(prompt)
                if reply and "[LLM" not in reply:
                    return reply
        except Exception:
            pass   # LLM crashed — fall through to default

    return get("responses.default", "Thank you for sharing that. Can you tell me more?")


def respond_llm(text: str, context: str = None, tier: str = None, category: str = None) -> str:
    """Single-shot llama3 reply, optionally steered by context/tier/category.
    Used when the rule table is disabled or no rule matched.
    """
    from core.llm import chat, build_prompt, persona_system, is_alive
    if not is_alive():
        return get("responses.default", "Thank you for sharing that.")

    if tier or category:
        prompt = build_prompt(
            "respond",
            persona=persona_system(),
            tier=tier or "NEUTRAL",
            category=category or "GENERAL",
            text=text,
        )
    elif context:
        prompt = f"{context}\nUser said: \"{text}\"\nReply helpfully in 1-3 sentences."
    else:
        prompt = text

    reply = chat(prompt)
    if reply and "[LLM" not in reply:
        return reply
    return get("responses.default", "Thank you for sharing that.")


def respond_with_memory(state, text: str) -> str:
    """llama3 reply with full conversation history.
    This gives the model memory of all prior turns — best quality but slowest.
    Falls back to respond_llm (single-turn) if multi-turn fails.
    """
    from core.llm import chat_messages, persona_system, is_alive
    if not is_alive():
        return get("responses.default", "Thank you for sharing that.")
    try:
        msgs = state.messages_for_llm(system=persona_system())
        msgs.append({"role": "user", "content": text})
        reply = chat_messages(msgs)
        if reply and "[LLM" not in reply:
            return reply
    except Exception:
        pass
    return respond_llm(text)


if __name__ == "__main__":
    combos = [
        ("I feel completely hopeless",     "WELLBEING", "CRISIS"),
        ("I have an exam tomorrow",        "ACADEMIC",  "STRESSED"),
        ("I cannot afford my rent",        "FINANCIAL", "DISTRESSED"),
        ("The portal won't let me log in", "TECHNICAL", "NEUTRAL"),
        ("I feel very alone",              "SOCIAL",    "DISTRESSED"),
        ("No specific category match",     "GENERAL",   "NEUTRAL"),
    ]
    for text, cat, tier in combos:
        r = respond(text, cat, tier)
        print(f"[{cat}|{tier}] -> {r[:70]}")

"""
text/respond.py -- response generation: rule-combo table first, llama3 fallback.

    respond("I can't sleep", "WELLBEING", "DISTRESSED")  -> the matching template
    respond_llm("...", context="user is stressed about exams")  -> free llama3 reply
    respond_with_memory(state, "and now I'm broke too")  -> llama3 WITH conversation history
"""
from core.conf import get


def _rule_lookup(category: str, tier: str):
    """Look up 'CATEGORY|TIER', then 'CATEGORY|*' wildcard."""
    rules = get("responses.rules") or {}
    for key in (f"{category}|{tier}", f"{category}|*"):
        if key in rules:
            return rules[key]
    return None


def respond(text: str, category: str, tier: str) -> str:
    """Combo response. Rule table -> (optional) llama3 fallback -> default string.
    (NEXUS `nexus_respond()` equivalent.)"""
    hit = _rule_lookup(category, tier)
    if hit:
        return hit
    if get("responses.use_llm_fallback", True):
        from core.llm import is_alive
        if is_alive():
            return respond_llm(text, context=f"The user seems {tier} and needs {category} support.")
    return get("responses.default", "Thank you for sharing that.")


def respond_llm(text: str, context: str = None) -> str:
    """Single-shot llama3 reply, optionally steered by `context`."""
    from core.llm import chat
    if context:
        prompt = f"{context}\nUser said: \"{text}\"\nReply helpfully in 1-3 sentences."
    else:
        prompt = text
    return chat(prompt)


def respond_with_memory(state, text: str) -> str:
    """llama3 reply that REMEMBERS the whole conversation (passes message history)."""
    from core.llm import chat_messages, persona_system
    msgs = state.messages_for_llm(system=persona_system())
    msgs.append({"role": "user", "content": text})
    return chat_messages(msgs)

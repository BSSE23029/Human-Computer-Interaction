"""
core/llm.py -- the LLM brain: local llama3 via Ollama, using the openai client.

Ollama exposes an OpenAI-compatible API at http://localhost:11434/v1.
We use the `openai` package pointed at localhost — NO internet, NO real API key.

START OLLAMA FIRST (separate terminal, leave open):   ollama serve

Fallback chain for every function:
  1. Call llama3 via Ollama
  2. If Ollama is unreachable AND offline_fallback=true → return a safe default
  3. If offline_fallback=false → raise so the user knows setup is broken

Public API
----------
    is_alive()                              -> bool
    warm_up()                               -> None
    persona_system()                        -> str
    build_prompt(name, **vars)              -> str
    chat(prompt, system, ...)               -> str
    chat_messages(messages, ...)            -> str
    chat_stream(prompt_or_msgs, ...)        -> Iterator[str]
    complete_json(prompt, ...)              -> dict
    classify(text, labels)                  -> str   (validated single label)
    multi_classify(text, labels)            -> list  (all matching labels)
    score(text, lo, hi, criterion)          -> float (clamped)
    extract(text, fields)                   -> dict
    summarize(text, max_words, audience)    -> str
"""

import re
import json
import requests

from core.conf import get

# ── lazy singleton openai client ─────────────────────────────────────────────
_client = None


def _get_client():
    """Return a cached OpenAI client aimed at Ollama. Lazy so import never crashes."""
    global _client
    if _client is None:
        try:
            from openai import OpenAI  # type: ignore
            _client = OpenAI(
                base_url=get("llm.base_url", "http://localhost:11434/v1"),
                api_key=get("llm.api_key", "ollama"),
                timeout=float(get("llm.timeout_seconds", 60)),
            )
        except ImportError:
            return None
    return _client


def _offline_ok() -> bool:
    """True if we should silently fall back instead of raising."""
    return bool(get("llm.offline_fallback", True))


# ── health / warmup ───────────────────────────────────────────────────────────
def is_alive(timeout: float = 2.0) -> bool:
    """Ping Ollama. Returns True if the server is up and responding."""
    base = get("llm.base_url", "http://localhost:11434/v1").replace("/v1", "")
    try:
        r = requests.get(f"{base}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def warm_up() -> None:
    """Fire a tiny 1-token request so llama3 loads into RAM before the demo.
    First call can take 5-10 seconds — this hides that lag at startup."""
    print("[llm] warming up llama3...")
    try:
        result = chat("hi", system="Reply: ok", max_tokens=3)
        if "[LLM" not in result:
            print("[llm] warm-up done ✓")
        else:
            print(f"[llm] warm-up: {result}")
    except Exception as e:
        print(f"[llm] warm-up failed: {e}")


# ── persona / system prompt ───────────────────────────────────────────────────
def persona_system() -> str:
    """Build the system prompt string from the active persona in config."""
    name = get("llm.active_persona", "default")
    persona = get(f"llm.personas.{name}")
    if persona:
        # personas are stored as full sentences starting with "You are..."
        return persona if persona.startswith("You are") else f"You are {persona}"
    # absolute fallback
    return get("llm.system_prompt", "You are a concise, helpful assistant.")


def build_prompt(name: str, **vars) -> str:
    """Fill a named template from config (llm.prompts.<name>) with {slot} values.

        build_prompt("classify", labels="A, B, C", text="I failed my exam")
        -> "Classify into ONE of: A, B, C.\nReply with ONLY the label.\n..."

    If the template or a slot is missing, returns a plain version that still works.
    """
    template = get(f"llm.prompts.{name}", "")
    if not template:
        # construct a minimal working prompt as fallback
        return f"{name}: {vars.get('text', str(vars))}"
    try:
        return template.format(**vars)
    except KeyError as e:
        # some slot is missing — return the template with unfilled slots rather than crash
        print(f"[llm] build_prompt('{name}') missing slot {e} — using partial template")
        return template


# ── core chat ─────────────────────────────────────────────────────────────────
def chat(
    prompt: str,
    system: str = None,
    temperature: float = None,
    max_tokens: int = None,
    model: str = None,
) -> str:
    """Single-turn completion. Returns the reply string.

    On failure returns a descriptive '[LLM ...]' string — never raises —
    so downstream code can display it or detect it with `"[LLM" in result`.
    """
    messages = [
        {"role": "system",  "content": system or persona_system()},
        {"role": "user",    "content": prompt},
    ]
    return chat_messages(messages, temperature=temperature,
                         max_tokens=max_tokens, model=model)


def chat_messages(
    messages: list,
    temperature: float = None,
    max_tokens: int = None,
    model: str = None,
) -> str:
    """Multi-turn completion. `messages` = list of {role, content} dicts.
    Pass the full conversation history here to give llama3 memory of prior turns.

    Context window:  if llm.context_window > 0, only the last N user+assistant
    pairs are sent (plus the system message) to keep llama3 focused.
    """
    c = _get_client()
    if c is None:
        msg = "[LLM unavailable] `openai` package not installed. Run: pip install openai"
        if not _offline_ok():
            raise RuntimeError(msg)
        return msg

    # apply context window trimming
    window = int(get("llm.context_window", 0))
    if window > 0 and len(messages) > 1:
        system_msgs = [m for m in messages if m["role"] == "system"]
        conv_msgs   = [m for m in messages if m["role"] != "system"]
        # keep last window*2 entries (each exchange = user + assistant = 2)
        conv_msgs = conv_msgs[-(window * 2):]
        messages = system_msgs + conv_msgs

    try:
        resp = c.chat.completions.create(
            model=model or get("llm.model", "llama3"),
            messages=messages,
            temperature=temperature if temperature is not None else get("llm.temperature", 0.4),
            max_tokens=max_tokens or get("llm.max_tokens", 512),
        )
        return (resp.choices[0].message.content or "").strip()

    except Exception as e:
        # give a human-readable error, not a traceback
        if not is_alive():
            msg = ("[LLM offline] Ollama is not responding.\n"
                   "Fix: open a terminal → run `ollama serve` → leave it open.")
        else:
            msg = f"[LLM error] {type(e).__name__}: {e}"

        if not _offline_ok():
            raise RuntimeError(msg) from e
        return msg


def chat_stream(prompt_or_messages, system: str = None, temperature: float = None):
    """Yield reply text chunks as they arrive (for gradio live-typing effect).

    `prompt_or_messages` can be:
        str   -> wrapped into a single-turn messages list automatically
        list  -> used as-is (full messages list with history)

    Usage in gradio:
        for chunk in llm.chat_stream("hello"):
            yield chunk
    """
    if isinstance(prompt_or_messages, str):
        messages = [
            {"role": "system", "content": system or persona_system()},
            {"role": "user",   "content": prompt_or_messages},
        ]
    else:
        messages = prompt_or_messages

    c = _get_client()
    if c is None:
        yield "[LLM unavailable] openai package not installed."
        return

    try:
        stream = c.chat.completions.create(
            model=get("llm.model", "llama3"),
            messages=messages,
            temperature=temperature if temperature is not None else get("llm.temperature", 0.4),
            max_tokens=get("llm.max_tokens", 512),
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"[LLM stream error] {e}"


# ── JSON output (robust) ──────────────────────────────────────────────────────
def _extract_json(s: str):
    """Pull a JSON object/array out of messy model output.
    Handles markdown fences, leading prose, trailing explanation."""
    if not s:
        return None
    s = s.strip()
    # strip ```json ... ``` fences
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s*```$", "", s).strip()
    # try direct parse first
    try:
        return json.loads(s)
    except Exception:
        pass
    # extract first balanced {...} or [...]
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = s.find(open_c)
        end   = s.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except Exception:
                continue
    return None


def complete_json(prompt: str, system: str = None, retries: int = 2) -> dict:
    """Ask llama3 for JSON output. Retries up to `retries` times.
    Returns {} if all attempts fail — never raises."""
    sys_msg = system or "Reply with ONLY valid JSON. No explanation, no prose, no markdown."
    use_fmt = bool(get("llm.use_json_format", False))

    c = _get_client()
    if c is None:
        return {}

    for attempt in range(retries + 1):
        try:
            kwargs = dict(
                model=get("llm.model", "llama3"),
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,   # deterministic for structured output
                max_tokens=get("llm.max_tokens", 512),
            )
            if use_fmt:
                kwargs["response_format"] = {"type": "json_object"}

            resp = c.chat.completions.create(**kwargs)
            parsed = _extract_json(resp.choices[0].message.content)
            if isinstance(parsed, (dict, list)):
                return parsed

        except Exception as e:
            if not is_alive():
                break  # no point retrying if Ollama is down
            if attempt < retries:
                continue

    return {}


# ── classification / scoring / extraction ────────────────────────────────────
def classify(text: str, labels: list, system: str = None) -> str:
    """Pick exactly ONE label from `labels`. Validates the response.
    Falls back to labels[0] if the model goes off-script.

    Why validation: llama3 sometimes says 'The label is ACADEMIC because...'
    We scan for any label name in the output to recover from that.
    """
    if not labels:
        return ""
    prompt = build_prompt("classify", labels=", ".join(labels), text=text)
    out = chat(
        prompt,
        system=system or "Reply with ONLY a single label from the list. Nothing else.",
        temperature=0.0,
    )
    if "[LLM" in out:
        return labels[0]
    # scan output for any matching label (case-insensitive)
    out_upper = out.upper()
    for lab in labels:
        if lab.upper() in out_upper:
            return lab
    return labels[0]


def multi_classify(text: str, labels: list) -> list:
    """Return ALL applicable labels (subset of `labels`). Returns [] if none match."""
    if not labels:
        return []
    prompt = build_prompt("multilabel", labels=", ".join(labels), text=text)
    out = chat(
        prompt,
        system="Reply with ONLY matching labels as a comma-separated list. Nothing else.",
        temperature=0.0,
    )
    if "[LLM" in out:
        return []
    out_upper = out.upper()
    return [lab for lab in labels if lab.upper() in out_upper]


def score(
    text: str,
    lo: float = -1.0,
    hi: float = 1.0,
    criterion: str = "sentiment",
) -> float:
    """Get a numeric rating in [lo, hi]. Falls back to midpoint on parse failure."""
    prompt = build_prompt("score", criterion=criterion, lo=lo, hi=hi, text=text)
    out = chat(
        prompt,
        system="Reply with ONLY a decimal number. Nothing else.",
        temperature=0.0,
    )
    if "[LLM" in out:
        return (lo + hi) / 2.0
    m = re.search(r"-?\d+(?:\.\d+)?", out)
    if not m:
        return (lo + hi) / 2.0
    return max(lo, min(hi, float(m.group())))


def extract(text: str, fields: list) -> dict:
    """Extract named fields from text into a dict. Returns {field: None} on failure."""
    if not fields:
        return {}
    prompt = build_prompt("extract", fields=", ".join(fields), text=text)
    data = complete_json(prompt)
    if isinstance(data, dict) and data:
        # fill in any missing fields with None
        return {f: data.get(f) for f in fields}
    return {f: None for f in fields}


def summarize(
    text: str,
    max_words: int = None,
    audience: str = None,
) -> str:
    """Summarize a conversation / text for a given audience."""
    max_words = max_words or 60
    audience  = audience  or get("report.narrative_audience", "a counsellor")
    prompt = build_prompt(
        "summarize",
        conversation=text,
        max_words=max_words,
        audience=audience,
    )
    return chat(prompt, system="Write a concise, accurate summary.")


if __name__ == "__main__":
    print("Ollama alive:", is_alive())
    if is_alive():
        warm_up()
        print("chat:         ", chat("Reply with exactly: hello world"))
        print("classify:     ", classify("I failed my exam", ["ACADEMIC","FINANCIAL","SOCIAL"]))
        print("multi_classify:", multi_classify("I am stressed about fees", ["ACADEMIC","FINANCIAL","WELLBEING"]))
        print("score:        ", score("I feel completely hopeless and alone"))
        print("extract:      ", extract("My name is Ali and I am 20 years old", ["name","age"]))
    else:
        print("Ollama not running. Start with: ollama serve")

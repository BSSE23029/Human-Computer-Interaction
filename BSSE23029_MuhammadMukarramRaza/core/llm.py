"""
core/llm.py -- the LLM brain: local llama3 served by Ollama, via the openai client.

Ollama exposes an OpenAI-compatible API at http://localhost:11434/v1, so we use
the standard `openai` package but point it at localhost. NO internet, NO real
OpenAI key.

START OLLAMA FIRST (separate terminal, leave open):   ollama serve

Public functions
----------------
    is_alive()                      -> bool        ping Ollama
    warm_up()                       -> None        pre-load model (avoid first-call lag)
    chat(prompt, system=None)       -> str         single-turn reply
    chat_messages(messages)         -> str         multi-turn (you pass history)
    chat_stream(prompt_or_msgs)     -> iterator    yields text chunks (for gradio)
    complete_json(prompt)           -> dict        robust JSON output (retries + parse)
    classify(text, labels)          -> str         pick ONE label (validated)
    multi_classify(text, labels)    -> list        pick ALL applicable labels
    score(text, lo, hi, criterion)  -> float       numeric rating, clamped
    extract(text, fields)           -> dict         pull named fields into a dict
    summarize(text, max_words)      -> str          summary
    build_prompt(name, **vars)      -> str          fill a config prompt template
    persona_system()                -> str          active persona as a system prompt
"""
import re
import json
import requests

from core.conf import get

# --- lazy singleton client -------------------------------------------------
_client = None


def client():
    """Return a cached openai client pointed at Ollama. Imported lazily so the
    rest of the package works even if `openai` is not installed."""
    global _client
    if _client is None:
        from openai import OpenAI  # lazy import
        _client = OpenAI(
            base_url=get("llm.base_url", "http://localhost:11434/v1"),
            api_key=get("llm.api_key", "ollama"),
            timeout=get("llm.timeout_seconds", 60),
        )
    return _client


# --- health / warmup -------------------------------------------------------
def is_alive(timeout: float = 2.0) -> bool:
    """True if the Ollama server answers. Use this to show a friendly error."""
    base = get("llm.base_url", "http://localhost:11434/v1").replace("/v1", "")
    try:
        r = requests.get(f"{base}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def warm_up() -> None:
    """Fire a tiny request so the model loads into RAM before the demo."""
    try:
        chat("hello", system="Reply with one word.", max_tokens=5)
    except Exception:
        pass


# --- core chat -------------------------------------------------------------
def persona_system() -> str:
    """Build the system prompt from the active persona (falls back to system_prompt)."""
    name = get("llm.active_persona", "default")
    persona = get(f"llm.personas.{name}")
    if persona:
        return f"You are {persona}"
    return get("llm.system_prompt", "You are a helpful assistant.")


def chat(prompt: str, system: str = None, temperature: float = None,
         max_tokens: int = None, model: str = None) -> str:
    """Single-turn completion -> reply string. Returns a clear error string on failure."""
    messages = []
    messages.append({"role": "system", "content": system or persona_system()})
    messages.append({"role": "user", "content": prompt})
    return chat_messages(messages, temperature=temperature,
                         max_tokens=max_tokens, model=model)


def chat_messages(messages: list, temperature: float = None,
                  max_tokens: int = None, model: str = None) -> str:
    """Multi-turn completion. `messages` is a list of {role, content} dicts.
    role is one of: system / user / assistant. This is how you give the
    assistant MEMORY of the conversation."""
    try:
        resp = client().chat.completions.create(
            model=model or get("llm.model", "llama3"),
            messages=messages,
            temperature=get("llm.temperature", 0.4) if temperature is None else temperature,
            max_tokens=max_tokens or get("llm.max_tokens", 512),
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        if not is_alive():
            return ("[LLM offline] Ollama is not responding. Open a terminal and run "
                    "`ollama serve`, then make sure `ollama pull llama3` was done.")
        return f"[LLM error] {e}"


def chat_stream(prompt_or_messages, system: str = None, temperature: float = None):
    """Yield reply chunks as they arrive. `prompt_or_messages` may be a string or
    a messages list. Use in gradio with `yield`-based handlers for live typing."""
    if isinstance(prompt_or_messages, str):
        messages = [{"role": "system", "content": system or persona_system()},
                    {"role": "user", "content": prompt_or_messages}]
    else:
        messages = prompt_or_messages
    try:
        stream = client().chat.completions.create(
            model=get("llm.model", "llama3"),
            messages=messages,
            temperature=get("llm.temperature", 0.4) if temperature is None else temperature,
            max_tokens=get("llm.max_tokens", 512),
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"[LLM error] {e}"


# --- prompt templates ------------------------------------------------------
def build_prompt(name: str, **vars) -> str:
    """Fill a named template from config (llm.prompts.<name>) with {slot} values."""
    template = get(f"llm.prompts.{name}", "")
    try:
        return template.format(**vars)
    except Exception:
        return template


# --- JSON / structured output ---------------------------------------------
def _extract_json(s: str):
    """Best-effort: pull a JSON object/array out of model text (handles ```fences```)."""
    if not s:
        return None
    s = s.strip()
    # strip markdown code fences
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    # direct parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # grab first balanced {...} or [...]
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = s.find(open_c)
        end = s.rfind(close_c)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end + 1])
            except Exception:
                continue
    return None


def complete_json(prompt: str, system: str = None, retries: int = 2) -> dict:
    """Ask the model for JSON and parse it robustly. Returns {} if all attempts fail.
    Tries Ollama's json format mode if enabled in config."""
    sys = system or "You are a precise assistant that replies with ONLY valid JSON, no prose."
    use_format = get("llm.use_json_format", False)
    for attempt in range(retries + 1):
        try:
            kwargs = dict(
                model=get("llm.model", "llama3"),
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=get("llm.max_tokens", 512),
            )
            if use_format:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client().chat.completions.create(**kwargs)
            parsed = _extract_json(resp.choices[0].message.content)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            if not is_alive():
                break
    return {}


# --- classification / scoring / extraction (LLM versions) ------------------
def classify(text: str, labels: list, system: str = None) -> str:
    """Pick exactly ONE label. Validates the model's answer against `labels`;
    returns labels[0] if the model goes off-script."""
    prompt = build_prompt("classify", labels=", ".join(labels), text=text) or \
        f"Classify into one of {labels}. Reply with only the label.\n{text}"
    out = chat(prompt, system=system or "Reply with only a single label.", temperature=0.0)
    out_clean = out.strip().strip(".").upper()
    for lab in labels:
        if lab.upper() in out_clean:
            return lab
    return labels[0] if labels else out.strip()


def multi_classify(text: str, labels: list) -> list:
    """Return ALL applicable labels (subset of `labels`)."""
    prompt = build_prompt("multilabel", labels=", ".join(labels), text=text) or \
        f"List ALL applicable from {labels}, comma separated.\n{text}"
    out = chat(prompt, system="Reply with only labels, comma-separated.", temperature=0.0)
    found = [lab for lab in labels if lab.upper() in out.upper()]
    return found


def score(text: str, lo: float = -1.0, hi: float = 1.0, criterion: str = "sentiment") -> float:
    """Return a numeric rating in [lo, hi]. Falls back to midpoint on parse failure."""
    prompt = build_prompt("score", criterion=criterion, lo=lo, hi=hi, text=text) or \
        f"Rate the {criterion} from {lo} to {hi}. Reply with only a number.\n{text}"
    out = chat(prompt, system="Reply with only a number.", temperature=0.0)
    m = re.search(r"-?\d+(?:\.\d+)?", out)
    if not m:
        return (lo + hi) / 2.0
    val = float(m.group())
    return max(lo, min(hi, val))


def extract(text: str, fields: list) -> dict:
    """Extract named fields from text into a dict (entity extraction)."""
    prompt = build_prompt("extract", fields=", ".join(fields), text=text) or \
        f"Extract {fields} as JSON.\n{text}"
    data = complete_json(prompt)
    if isinstance(data, dict):
        return data
    return {f: None for f in fields}


def summarize(text: str, max_words: int = 60, audience: str = "a counsellor") -> str:
    """Summarize text (e.g., a conversation) for a given audience."""
    prompt = build_prompt("summarize", conversation=text, max_words=max_words, audience=audience) or \
        f"Summarize in {max_words} words for {audience}.\n{text}"
    return chat(prompt, system="You write concise, accurate summaries.")


if __name__ == "__main__":
    print("Ollama alive:", is_alive())
    if is_alive():
        print("Chat:", chat("Reply with exactly: hello"))
        print("Classify:", classify("I failed my exam", ["ACADEMIC", "FINANCIAL", "SOCIAL"]))
        print("Score:", score("I feel completely hopeless"))

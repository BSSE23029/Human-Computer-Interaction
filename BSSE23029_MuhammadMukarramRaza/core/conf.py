"""
core/conf.py -- configuration loader (single source of truth).

Loads config.yaml (preferred) or falls back to config.json if PyYAML is
unavailable. Exposes:

    CFG            -- the whole config as a nested dict
    get(path, default)  -- dotted lookup, e.g. get("llm.model")
    reload()       -- re-read the files from disk (handy while tuning)
    sync_json()    -- regenerate config.json from config.yaml (run at home)

Usage:
    from core.conf import CFG, get
    model = get("llm.model")              # "llama3"
    tiers = get("scale.tiers")            # list of [name, low, high, emoji]
"""
import os
import json

# Project root = the folder this file's package lives in (one level up from core/)
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
YAML_PATH = os.path.join(ROOT, "config.yaml")
JSON_PATH = os.path.join(ROOT, "config.json")


def _load() -> dict:
    """Load config from YAML if possible, else JSON. Never crash the import."""
    # Prefer YAML (gradio ships PyYAML, so this normally succeeds).
    try:
        import yaml  # type: ignore
        if os.path.exists(YAML_PATH):
            with open(YAML_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception as e:  # PyYAML missing or YAML malformed
        print(f"[conf] YAML unavailable ({e!r}); trying config.json")
    # Fallback: JSON mirror.
    try:
        if os.path.exists(JSON_PATH):
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[conf] could not load config.json either: {e!r}")
    print("[conf] WARNING: running with EMPTY config.")
    return {}


CFG: dict = _load()


def get(path: str, default=None):
    """Dotted-path lookup into CFG. Returns `default` if any key is missing.

    Example: get("scoring.thresholds.FOLLOW_UP") -> 40
    """
    node = CFG
    for key in path.split("."):
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    return node


def reload() -> dict:
    """Re-read config from disk into CFG (in place, so existing imports see it)."""
    global CFG
    fresh = _load()
    CFG.clear()
    CFG.update(fresh)
    return CFG


def sync_json() -> None:
    """Regenerate config.json from config.yaml. Requires PyYAML (available at home)."""
    import yaml  # type: ignore
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[conf] synced config.json from config.yaml")


if __name__ == "__main__":
    # Quick self-test: print a few values.
    print("Loaded config from:", "YAML" if os.path.exists(YAML_PATH) else "JSON")
    print("llm.model        =", get("llm.model"))
    print("whisper.model_size =", get("whisper.model_size"))
    print("scale tiers      =", len(get("scale.tiers", [])))
    print("categories       =", list((get("categories") or {}).keys()))

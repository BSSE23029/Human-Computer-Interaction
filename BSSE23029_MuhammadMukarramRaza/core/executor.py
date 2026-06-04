"""
core/executor.py  --  the ordered stage executor.

Reads the pipeline definition from config (tabs.<tab>.pipeline_order + overrides),
resolves global stage defaults, then runs each stage in order.

KEY CONTRACT — interdependency safety:
    Every disabled or failed stage writes its fallback to TurnContext WITH
    _skipped=True.  Dependent stages MUST check ctx[dep].get("_skipped") before
    reading the value.  This makes config → code correspondence explicit:
    if config says disabled, code provably branches on _skipped.

    Example:
        scale stage checks  ctx["sentiment"].get("_skipped")
        → True  → uses LLM to score text directly (sentiment was off)
        → False → reads ctx["sentiment"]["compound"] (normal path)

        trajectory checks   ctx["classify"].get("_skipped")
        → True  → skips transition detection, returns wellbeing trend only
        → False → full transition + trend analysis

TurnContext dict keys (written by stages in order):
    stt_result      voice_input stage output
    text_raw        text_input stage output
    vision_label    vision_input stage output
    text            fuser output — the resolved semantic text
    source          fuser output — which input won ("text"|"voice"|"vision")
    sentiment       analyzer output
    scale           tier_mapper output
    classify        classifier output
    trajectory      tracker output
    respond         responder output
    errors          list of (stage_name, exception) — accumulated across the run

Usage:
    from core.executor import run_tab_pipeline
    ctx = run_tab_pipeline("voice", initial_ctx={"turn": 3})
"""

import copy
import traceback
from typing import Any

from core.conf import get as _get


# ── config helpers ────────────────────────────────────────────────────────────

def _global_stage(name: str) -> dict:
    """Return the global stage definition for `name`, or empty dict."""
    stages = _get("stages") or []
    for s in stages:
        if isinstance(s, dict) and s.get("name") == name:
            return s
    return {}


def _tab_cfg(tab: str) -> dict:
    """Return the tab context config, or empty dict if tab not found."""
    return (_get(f"tabs.{tab}") or {})


def _resolve_stage(tab: str, stage_name: str) -> dict:
    """
    Merge: global stage definition  ←  tab override.
    Tab override wins on any key conflict.
    Returns the fully resolved stage config dict.
    """
    base     = copy.deepcopy(_global_stage(stage_name))
    overrides = (_tab_cfg(tab).get("overrides") or {}).get(stage_name, {})
    _deep_merge(base, overrides)
    return base


def _deep_merge(base: dict, override: dict) -> None:
    """Merge override INTO base in-place."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _stage_enabled(tab: str, stage_name: str) -> bool:
    """
    A stage is enabled unless it is EXPLICITLY disabled.
    Two ways to disable a stage:
        1. Tab override:           overrides.{stage}.enabled: false
        2. Global stage definition: stages[name].enabled: false

    Legacy section flags (sentiment.enabled, categories.enabled, etc.) do NOT
    disable stages — they switch the method used within the stage (lexicon→LLM,
    keyword→LLM, rule→LLM).  This is intentional: a stage should always produce
    output; only the backend changes.  See stages.py for method-switching logic.

    Priority:
        tab override.enabled    (most specific — wins immediately if present)
        global stage.enabled    (the config default)
        True                    (ultimate default — enabled if not specified)
    """
    # 1. Tab override (explicit wins)
    tab_overrides = (_tab_cfg(tab).get("overrides") or {}).get(stage_name, {})
    if "enabled" in tab_overrides:
        return bool(tab_overrides["enabled"])

    # 2. Global stage definition default
    global_stage = _global_stage(stage_name)
    if "enabled" in global_stage:
        return bool(global_stage["enabled"])

    return True


# ── null / fallback values per stage type ─────────────────────────────────────
# These are injected when a stage is disabled or fails with no fallback method.

_NULL: dict[str, Any] = {
    "stt_result":   {"text": "", "language": "?", "confidence": "low", "_skipped": True},
    "text_raw":     {"text": "", "_skipped": True},
    "vision_label": {"present": False, "faces": 0, "mood": "no_face",
                     "head_zone": None, "gesture": "none", "fingers": 0, "_skipped": True},
    "fuser":        {"text": "", "source": "none", "word_count": 0, "_skipped": True},
    "context_inject":{"injected": False, "_skipped": True},
    "sentiment":    {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0, "_skipped": True},
    "scale":        {"tier": "NEUTRAL", "score": 0.0, "emoji": "😐",
                     "is_at_risk": False, "_skipped": True},
    "classify":     {"primary": "GENERAL", "all_detected": [], "scores": {}, "_skipped": True},
    "trajectory":   {"trend": "unknown", "lowest_tier": None, "at_risk_turns": [],
                     "first_avg": 0.0, "second_avg": 0.0, "_skipped": True},
    "respond":      {"response": "", "_skipped": True},
    "tts":          {"spoken": False, "_skipped": True},
    "display":      {"displayed": True},
    "log":          {"logged": True},
}


def _null(stage_name: str) -> dict:
    return copy.deepcopy(_NULL.get(stage_name, {"_skipped": True}))


# ── executor ──────────────────────────────────────────────────────────────────

def run_tab_pipeline(tab: str, initial_ctx: dict = None) -> dict:
    """
    Run the ordered pipeline for `tab`.

    Returns the completed TurnContext dict containing every stage's output.
    Never raises — errors are collected in ctx["errors"].
    """
    from pipeline.stages import STAGE_REGISTRY  # imported here to avoid circular

    ctx = {
        "tab":    tab,
        "errors": [],
        **(initial_ctx or {}),
    }

    tab_config = _tab_cfg(tab)
    if not tab_config:
        ctx["errors"].append(("_executor", f"No tab config found for '{tab}'"))
        return ctx

    pipeline_order = tab_config.get("pipeline_order") or []

    for stage_name in pipeline_order:
        stage_cfg = _resolve_stage(tab, stage_name)
        ctx = _run_stage(stage_name, stage_cfg, ctx, tab)

    return ctx


def _run_stage(name: str, cfg: dict, ctx: dict, tab: str) -> dict:
    """Execute one stage. Writes result (or null with _skipped) into ctx."""
    from pipeline.stages import STAGE_REGISTRY

    # ── disabled ──────────────────────────────────────────────────────
    if not _stage_enabled(tab, name):
        null = _null(name)
        # Merge into ctx under the stage's canonical key
        ctx[name] = null
        # also write to convenience aliases
        _write_aliases(name, null, ctx)
        return ctx

    # ── execute ───────────────────────────────────────────────────────
    executor_fn = STAGE_REGISTRY.get(cfg.get("type", ""))
    if executor_fn is None:
        ctx["errors"].append((name, f"unknown stage type '{cfg.get('type')}'"))
        ctx[name] = _null(name)
        return ctx

    try:
        result = executor_fn(cfg, ctx)
        ctx[name] = result
        _write_aliases(name, result, ctx)

    except Exception as exc:
        # ── fallback ──────────────────────────────────────────────────
        ctx["errors"].append((name, str(exc)))
        fb = cfg.get("fallback") or {}
        fb_method = fb.get("method")
        recovered = False

        if fb_method and fb_method != cfg.get("method"):
            try:
                fb_cfg = {**cfg, "method": fb_method}
                result = executor_fn(fb_cfg, ctx)
                ctx[name] = result
                _write_aliases(name, result, ctx)
                recovered = True
            except Exception as fb_exc:
                ctx["errors"].append((f"{name}_fallback", str(fb_exc)))

        if not recovered:
            # last resort: literal fallback value or null
            fb_value = fb.get("value")
            if fb_value is not None:
                val = copy.deepcopy(fb_value) if isinstance(fb_value, dict) else fb_value
                ctx[name] = val
                _write_aliases(name, val, ctx)
            else:
                ctx[name] = _null(name)

        if not _get("pipeline.strict", False):
            pass   # silent fallback (exam default)
        else:
            raise  # strict mode: re-raise

    return ctx


def _write_aliases(stage_name: str, result: Any, ctx: dict) -> None:
    """Write stage output to convenient top-level ctx keys for downstream stages."""
    _ALIASES = {
        "stt_result":    lambda r: ctx.update({"stt_text": r.get("text", "") if isinstance(r, dict) else ""}),
        "fuser":         lambda r: ctx.update({
                            "text":       r.get("text", "")       if isinstance(r, dict) else "",
                            "source":     r.get("source", "none") if isinstance(r, dict) else "none",
                            "word_count": r.get("word_count", 0)  if isinstance(r, dict) else 0,
                        }),
        "scale":         lambda r: ctx.update({
                            "tier":       r.get("tier", "NEUTRAL")   if isinstance(r, dict) else "NEUTRAL",
                            "is_at_risk": r.get("is_at_risk", False) if isinstance(r, dict) else False,
                        }),
        "classify":      lambda r: ctx.update({
                            "category": r.get("primary", "GENERAL") if isinstance(r, dict) else "GENERAL",
                        }),
        "respond":       lambda r: ctx.update({
                            "response": r.get("response", "") if isinstance(r, dict) else str(r or ""),
                        }),
    }
    fn = _ALIASES.get(stage_name)
    if fn:
        try:
            fn(result)
        except Exception:
            pass   # alias write failure never breaks the pipeline


# ── tab validation ────────────────────────────────────────────────────────────

def validate_tabs() -> list:
    """
    Called at startup. Returns a list of warning strings for any tab where
    no input stage is enabled (the tab would render with no way to submit).
    Warnings are printed, not raised — never crash at startup.
    """
    warnings = []
    tabs_cfg = _get("tabs") or {}
    input_stage_types = {"voice_input", "text_input", "vision_input"}

    for tab_name, tab_cfg in tabs_cfg.items():
        if not isinstance(tab_cfg, dict) or not tab_cfg.get("enabled", True):
            continue
        order = tab_cfg.get("pipeline_order") or []
        has_input = False
        for stage_name in order:
            resolved = _resolve_stage(tab_name, stage_name)
            if resolved.get("type") in input_stage_types:
                if _stage_enabled(tab_name, stage_name):
                    has_input = True
                    break
        if not has_input and order:
            no_input_beh = (
                (_tab_cfg(tab_name).get("overrides") or {})
                .get("fuser", {})
                .get("config", {})
                .get("no_input_behavior", "warn")
            )
            msg = (f"Tab '{tab_name}': no input stage is enabled. "
                   f"Behaviour: {no_input_beh}")
            warnings.append(msg)

    return warnings

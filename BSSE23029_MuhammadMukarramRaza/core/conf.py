"""
core/conf.py -- configuration loader (single source of truth).

Load priority:
  1. config.yaml   (preferred — human-readable, commented)
  2. config.json   (fallback if PyYAML not installed)
  3. DEFAULTS      (hardcoded last-resort — app ALWAYS starts, even cold)

Usage:
    from core.conf import CFG, get, reload

    get("llm.model")               -> "llama3"
    get("scale.tiers")             -> list
    get("missing.key", "fallback") -> "fallback"
    CFG["llm"]["model"]            -> "llama3"   (direct dict access)
    reload()                       -> re-read files without restarting
    sync_json()                    -> write config.json mirror from YAML
"""

import json
import os
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(_HERE)
YAML_PATH = os.path.join(ROOT, "config.yaml")
JSON_PATH = os.path.join(ROOT, "config.json")

# ── hardcoded fallback — always works, never empty ───────────────────────────
DEFAULTS: dict = {
    "llm": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "llama3",
        "temperature": 0.4,
        "max_tokens": 512,
        "timeout_seconds": 60,
        "context_window": 0,
        "offline_fallback": True,
        "use_json_format": False,
        "active_persona": "default",
        "personas": {
            "default": "You are a concise, empathetic assistant. Reply in 1-3 sentences.",
            "tutor":   "You are a patient academic tutor. Explain clearly and encourage.",
            "support": "You are a calm, professional customer-service agent.",
        },
        "prompts": {
            "classify":   "Classify into ONE of: {labels}.\nReply with ONLY the label.\nMessage: {text}\nLabel:",
            "multilabel": "List ALL applicable from: {labels}.\nComma-separated only.\nMessage: {text}\nLabels:",
            "score":      "Rate the {criterion} from {lo} to {hi}. ONLY a number.\nMessage: {text}\nNumber:",
            "extract":    "Extract {fields} as JSON. Reply with ONLY the JSON object.\nMessage: {text}\nJSON:",
            "respond":    "You are {persona}.\nTier: {tier}, Category: {category}.\nRespond to: {text}",
            "summarize":  "Summarize in {max_words} words for {audience}.\n{conversation}\nSummary:",
        },
    },
    "whisper":  {"model_size": "base", "language": None, "fp16": False},
    "audio":    {"sample_rate": 16000, "seconds": 7, "channels": 1},
    "voice":    {
        "recognizer": "whisper",
        "countdown": True,
        "silence_threshold": 0.01,
        "silence_max_seconds": 2.0,
        "confidence_min_chars": 10,
    },
    "sentiment": {
        "enabled": True,
        "positive_words": {
            "happy": 2.0, "good": 1.5, "great": 2.0, "better": 1.5,
            "calm": 1.5, "hopeful": 1.8, "relieved": 1.6, "thanks": 1.2,
            "love": 2.0, "excited": 1.8, "confident": 1.6,
        },
        "negative_words": {
            "sad": 2.0, "stressed": 2.0, "anxious": 2.2, "hopeless": 3.0,
            "depressed": 3.0, "overwhelmed": 2.5, "alone": 2.0, "lonely": 2.0,
            "afraid": 2.2, "tired": 1.5, "fail": 1.8, "failed": 1.8,
            "panic": 2.4, "cry": 2.0, "worried": 1.8,
        },
        "boosters":  {"very": 1.5, "really": 1.4, "completely": 1.7, "so": 1.3, "extremely": 1.8},
        "negations": ["not", "no", "never", "cannot", "can't", "don't", "didn't"],
    },
    "scale": {
        "enabled": True,
        "tiers": [
            ["THRIVING",     0.60,  100.0, "🌟"],
            ["CONTENT",      0.20,    0.60, "😊"],
            ["NEUTRAL",     -0.19,    0.20, "😐"],
            ["STRESSED",    -0.40,   -0.19, "😟"],
            ["DISTRESSED",  -0.60,   -0.40, "😢"],
            ["CRISIS",    -100.0,    -0.60, "🆘"],
        ],
        "behavior": {
            "STRESSED": {
                "alert": {"show": False, "message": "Turn {turn}: stressed.", "color": "orange"},
                "is_at_risk": False,
            },
            "DISTRESSED": {
                "alert": {"show": True, "message": "⚠ Turn {turn}: DISTRESSED.", "color": "orange"},
                "is_at_risk": False,
            },
            "CRISIS": {
                "alert": {"show": True, "message": "🚨 CRISIS — Turn {turn}: Immediate help needed.", "color": "red"},
                "is_at_risk": True,
            },
        },
    },
    "categories": {
        "enabled": True,
        "ACADEMIC":  {"escalation_target": False, "response_priority": 2,
                      "keywords": ["assignment","deadline","exam","grade","fail","study","professor","submit","midterm"]},
        "WELLBEING": {"escalation_target": True,  "response_priority": 5,
                      "keywords": ["stress","anxious","depressed","lonely","overwhelmed","panic","cry","hopeless","afraid","sleep","sad"]},
        "FINANCIAL": {"escalation_target": False, "response_priority": 3,
                      "keywords": ["fees","scholarship","loan","afford","money","rent","bursary","payment","debt","overdue"]},
        "TECHNICAL": {"escalation_target": False, "response_priority": 1,
                      "keywords": ["portal","login","password","system","error","access","email","vpn","reset","laptop","broke","files"]},
        "SOCIAL":    {"escalation_target": False, "response_priority": 2,
                      "keywords": ["friends","roommate","belong","isolated","group","relationship","community","alone"]},
        "ADMIN":     {"escalation_target": False, "response_priority": 1,
                      "keywords": ["enrolment","certificate","transcript","registration","form","office","register"]},
    },
    "responses": {
        "enabled": True,
        "use_llm_fallback": True,
        "default": "Thank you for sharing that. Can you tell me a little more?",
        "rules": {
            "WELLBEING|CRISIS":     "I am very concerned. Please contact counselling RIGHT NOW: 0800-XXX-XXXX.",
            "WELLBEING|DISTRESSED": "That sounds really difficult. Have you spoken to anyone about how you feel?",
            "WELLBEING|STRESSED":   "Things seem to be piling up. What is weighing on you most right now?",
            "ACADEMIC|STRESSED":    "Exam pressure is real. Have you spoken to your tutor about support?",
            "FINANCIAL|*":          "Financial difficulty is common. The bursary office can help.",
            "TECHNICAL|*":          "Let me help. Which system are you trying to access?",
            "SOCIAL|DISTRESSED":    "Feeling isolated is hard. The student union runs weekly social events.",
            "WELLBEING":            "It sounds like a tough time. Would you like to talk more about how you feel?",
            "ACADEMIC":             "Let us look at what academic support is available.",
            "SOCIAL":               "Social connections matter. Have you tried any student society events?",
        },
    },
    "scoring": {
        "enabled": True,
        "base_risk": 20,
        "wellbeing_weight": 40,
        "escalation_weight": 10,
        "verbose_threshold": 20,
        "verbose_bonus": 5,
        "tier_weights": {"STRESSED": 5, "DISTRESSED": 10, "CRISIS": 15},
        "thresholds": {"URGENT_REFERRAL": 70, "FOLLOW_UP": 40, "NO_ACTION": 0},
    },
    "trajectory": {
        "method": "halves",
        "improving_threshold": 0.10,
        "declining_threshold": 0.10,
    },
    "session": {
        "max_turns": 10,
        "auto_report": True,
        "input_modes": {"text": True, "voice": True, "vision": False},
    },
    "replay": {
        "enabled": True,
        "print_per_turn": True,
        "print_report": True,
        "log": [
            "Hi, I need some help please.",
            "I have a major assignment due tomorrow and I have not started.",
            "My laptop also broke yesterday so I cannot access my files.",
            "To be honest I have been struggling a lot lately, not just academically.",
            "I have not been sleeping, I feel completely hopeless about everything.",
            "I think I might need to talk to someone but I do not know who.",
            "Also I got an email saying my fees are overdue and I cannot register.",
            "Sorry for dumping all this. I just feel very alone right now.",
            "Actually, my friend just texted. I feel a tiny bit better now.",
            "Thank you for listening. I will try to contact the counsellor.",
        ],
    },
    "report": {
        "use_llm_narrative": True,
        "sections": {
            "session_summary": True,
            "wellbeing_trajectory": True,
            "support_categories": True,
            "escalations": True,
            "risk_score": True,
            "llm_narrative": True,
            "vision_summary": True,
            "include_raw_turns": False,
        },
        "llm_narrative_prompt": (
            "You are writing a {audience} report.\n"
            "Risk: {risk_score}/100 → Action: {action}\n"
            "Summary: {tier_summary}\n\nConversation:\n{conversation}\n\n"
            "Write a 3-5 sentence clinical narrative covering: "
            "emotional arc, support needs, critical moments, recommended next step. "
            "Do NOT repeat the raw numbers — interpret them."
        ),
        "narrative_audience": "university counselling team",
        "banner_title": "SESSION INTELLIGENCE REPORT",
        "banner_subtitle": "Counsellor Eyes Only",
        "banner_width": 55,
    },
    "vision": {
        "webcam_index": 0, "flip_webcam": True, "wait_key_ms": 1,
        "haar": {
            "face":  "haarcascade_frontalface_default.xml",
            "eye":   "haarcascade_eye.xml",
            "smile": "haarcascade_smile.xml",
            "face_scale": 1.1, "face_neighbors": 5,
            "eye_neighbors": 8, "smile_neighbors": 20,
        },
        "face_detect": {"enabled": True},
        "blink":       {"enabled": True, "eyes_closed_frames": 2, "display_count": True},
        "drowsy":      {"enabled": True, "closed_frames_alert": 20, "alert_message": "DROWSY!"},
        "head_pose":   {"enabled": True, "deadzone_ratio": 0.15, "display_zone": True},
        "smile_mood":  {"enabled": True, "display_mood": True},
        "gesture": {
            "enabled": True,
            "skin_ycrcb": {"lower": [0,133,77], "upper": [255,173,127]},
            "min_contour_area": 5000,
            "display_gesture": True,
            "gesture_map": {
                "0": ["Fist","✊"], "1": ["One","☝️"], "2": ["Peace","✌️"],
                "3": ["Three","3️⃣"], "4": ["Four","4️⃣"], "5": ["Open Hand","🖐"],
            },
        },
        "color_track": {
            "enabled": False, "active_color": "red", "display_centroid": True,
            "presets": {
                "red":   {"lower": [0,120,70],   "upper": [10,255,255]},
                "green": {"lower": [40,70,70],   "upper": [80,255,255]},
                "blue":  {"lower": [100,150,0],  "upper": [140,255,255]},
            },
        },
        "motion": {"enabled": False, "method": "diff", "threshold": 25, "min_area": 500, "display_motion": True},
        "hud": {
            "font_scale": 0.7, "thickness": 2,
            "color_default": [0,255,0], "color_warning": [0,0,255], "color_info": [255,255,0],
            "show_fps": True, "show_tier": True, "show_score": True, "show_turn": True,
            "layout": {
                "top_left":    ["blink","drowsy","head_zone"],
                "top_right":   ["fps","tier","score"],
                "bottom_left": ["gesture"],
                "bottom_right":["mood","turn"],
            },
        },
        "display": {"width": 640, "height": 480, "stream_fps": 15, "show_window": False},
        "stability": {"gesture_hold_frames": 5, "blink_debounce_frames": 3, "mood_history_frames": 15},
        "bridge": {
            "include_in_llm_context": True,
            "context_template": "[Vision] Face: {face_present}, Mood: {mood}, Gesture: {gesture}, Head: {head_zone}",
        },
    },
    "ui": {
        "app_title": "Multimodal HCI Assistant",
        "app_subtitle": "Text + Voice + Vision | llama3 + Whisper + OpenCV",
        "server_port": 7860,
        "share": False,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override INTO base recursively. Override always wins on conflicts.
    This means DEFAULTS fill in any keys missing from config.yaml."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load() -> tuple:
    """Return (merged_config_dict, source_string). Never raises."""
    # 1 — try YAML (gradio ships PyYAML, so this normally works)
    if os.path.exists(YAML_PATH):
        try:
            import yaml  # type: ignore
            with open(YAML_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            merged = _deep_merge(DEFAULTS, data)
            return merged, "config.yaml"
        except ImportError:
            print("[conf] PyYAML not installed — trying config.json")
        except Exception as e:
            print(f"[conf] config.yaml parse error ({e}) — trying config.json")

    # 2 — try JSON mirror
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            merged = _deep_merge(DEFAULTS, data)
            return merged, "config.json"
        except Exception as e:
            print(f"[conf] config.json error ({e}) — using hardcoded defaults")

    # 3 — hardcoded defaults (always works)
    print("[conf] WARNING: no config file loaded — using hardcoded defaults")
    return DEFAULTS.copy(), "hardcoded defaults"


# ── module singleton ──────────────────────────────────────────────────────────
CFG: dict
_source: str
CFG, _source = _load()
print(f"[conf] loaded from {_source}")


def get(dotted_key: str, default: Any = None) -> Any:
    """
    Dot-notation read from CFG. Never raises — returns `default` if key missing.

        get("llm.model")                   -> "llama3"
        get("scoring.tier_weights.CRISIS") -> 15
        get("nope.missing", "fallback")    -> "fallback"
    """
    node = CFG
    for key in dotted_key.split("."):
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    return node


def reload() -> None:
    """Re-read config files into CFG in-place. All existing `from core.conf import CFG`
    references automatically see the new values (same dict object, updated)."""
    global CFG, _source
    fresh, _source = _load()
    CFG.clear()
    CFG.update(fresh)
    print(f"[conf] reloaded from {_source}")


def sync_json() -> None:
    """Write a config.json mirror from config.yaml.
    Run once at home so the exam machine never needs PyYAML:
        python -c "from core.conf import sync_json; sync_json()"
    """
    try:
        import yaml  # type: ignore
        with open(YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[conf] synced → {JSON_PATH}")
    except Exception as e:
        print(f"[conf] sync_json failed: {e}")


# ── helpers used by other modules ─────────────────────────────────────────────
def enabled(section: str) -> bool:
    """True if config section has enabled: true (or key missing → default True)."""
    return bool(get(f"{section}.enabled", True))


def tier_behavior(tier_name: str) -> dict:
    """Return the behavior dict for a tier name. Safe — never raises."""
    return get(f"scale.behavior.{tier_name}", {
        "alert": {"show": False, "message": "", "color": "blue"},
        "is_at_risk": False,
    })


def category_cfg(cat_name: str) -> dict:
    """Return config for a single category. Safe fallback if missing."""
    return get(f"categories.{cat_name}", {
        "escalation_target": False,
        "response_priority": 1,
        "keywords": [],
    })


def active_categories() -> dict:
    """Return only the enabled categories as {name: cfg_dict}, excluding the
    top-level 'enabled' key itself."""
    cats = get("categories", {})
    return {
        k: v for k, v in cats.items()
        if k != "enabled" and isinstance(v, dict)
    }


if __name__ == "__main__":
    print("source       :", _source)
    print("llm.model    :", get("llm.model"))
    print("whisper size :", get("whisper.model_size"))
    print("scale tiers  :", len(get("scale.tiers", [])))
    print("categories   :", list(active_categories().keys()))
    print("session turns:", get("session.max_turns"))
    print("replay lines :", len(get("replay.log", [])))

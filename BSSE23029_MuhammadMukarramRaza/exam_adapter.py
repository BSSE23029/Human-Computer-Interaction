"""
exam_adapter.py  ===  YOUR EXAM-DAY WORKBENCH  ===

Two kinds of things live here:

A) EXACT-NAME WRAPPERS  (bottom of file)
   The exam mandates exact function names ("any other names score zero").
   You cannot predict them, but the logic lives in the packages.
   On exam day: define the mandated name as a 2-line wrapper that calls
   the library.  Rename / add as the paper demands.

B) TASK SWITCHERS  (top of file)
   One-call functions that flip active_persona + disable the right pipeline
   sections simultaneously.  Saves 5 minutes of manual config editing.

   use_ner()           → NER persona, sentiment/scale/categories/scoring OFF
   use_translator()    → Translation persona, same sections OFF
   use_qa()            → QA persona, sentiment/scale/categories OFF
   use_classifier()    → Text classification persona, categories OFF
   use_generator()     → Pure generation, ALL deterministic stages OFF
   restore_default()   → Wellbeing assistant, ALL sections ON

Usage:
    from exam_adapter import *

    # switch to NER task:
    use_ner()
    result = run_ner("Ali met Sara in Lahore on Monday at 3pm")
    print(result)   # {'person': ['Ali','Sara'], 'location': ['Lahore'], 'date':[], 'time':['3pm']}

    # back to the default wellbeing assistant:
    restore_default()
"""

# ═══════════════════════════════════════════════════════════════════════════════
# B) TASK SWITCHERS
# ═══════════════════════════════════════════════════════════════════════════════

def use_persona(name: str) -> None:
    """
    Switch active_persona AND disable the appropriate pipeline sections in one call.

    The disable map is read from config (llm.persona_disable_map.<name>).
    Any section listed as `true` is disabled; others are re-enabled.

    After calling this, all subsequent run_turn() calls use the new persona.
    Call restore_default() to go back.
    """
    from core.conf import CFG, get
    from core.llm import persona_system   # pre-warm the cache

    # 1. set persona
    CFG["llm"]["active_persona"] = name

    # 2. reset all managed sections to enabled first
    _managed = ["sentiment", "scale", "categories", "responses", "scoring"]
    for sec in _managed:
        if sec in CFG:
            CFG[sec]["enabled"] = True

    # 3. apply the disable map for this persona
    disable_map = get(f"llm.persona_disable_map.{name}") or {}
    disabled = []
    for section, should_disable in disable_map.items():
        if should_disable and section in CFG:
            CFG[section]["enabled"] = False
            disabled.append(section)

    print(f"[exam] persona → '{name}'")
    if disabled:
        print(f"       disabled : {', '.join(disabled)}")
    else:
        print(f"       all pipeline sections: ON")


def restore_default() -> None:
    """Reset to the default wellbeing assistant with all pipeline sections ON."""
    use_persona("default")


# ── Task-specific one-liners ──────────────────────────────────────────────────

def use_ner() -> None:
    """Named Entity Recognition mode. Disable: sentiment, scale, categories, scoring."""
    use_persona("ner_extractor")

def use_translator() -> None:
    """Translation mode. Disable: sentiment, scale, categories, scoring."""
    use_persona("translator")

def use_qa() -> None:
    """Question Answering mode. Disable: sentiment, scale, categories."""
    use_persona("qa_engine")

def use_classifier() -> None:
    """Text Classification mode. Disable: categories (LLM classifies instead)."""
    use_persona("text_classifier")

def use_intent_detector() -> None:
    """Intent Detection mode. Disable: categories."""
    use_persona("intent_detector")

def use_generator() -> None:
    """Pure text generation / NLG mode. Disable ALL deterministic stages."""
    use_persona("generator")


# ── Task helper functions ─────────────────────────────────────────────────────

def run_ner(text: str) -> dict:
    """
    Extract named entities from text.
    Automatically switches to ner_extractor persona.
    Returns {'person': [...], 'location': [...], 'date': [...], 'time': [...]}.
    """
    use_ner()
    from core.llm import complete_json, persona_system
    prompt = f"Extract named entities from:\n{text}"
    result = complete_json(prompt, system=persona_system())
    restore_default()
    # guarantee all keys present even if LLM misses some
    for key in ("person", "location", "date", "time"):
        if key not in result:
            result[key] = []
    return result


def run_translation(text: str, source_lang: str = "English",
                    target_lang: str = "Spanish") -> str:
    """Translate text. Switches to translator persona automatically."""
    use_translator()
    from core.llm import chat, persona_system, build_prompt
    prompt = build_prompt("translate",
                          source_lang=source_lang,
                          target_lang=target_lang,
                          text=text)
    result = chat(prompt, system=persona_system())
    restore_default()
    return result


def run_qa(question: str, context: str) -> str:
    """Answer a question from a context paragraph. Switches to qa_engine persona."""
    use_qa()
    from core.llm import chat, persona_system, build_prompt
    prompt = build_prompt("qa", context=context, text=question)
    result = chat(prompt, system=persona_system())
    restore_default()
    return result


def run_classify_llm(text: str, labels: list) -> str:
    """Classify text using LLM. Switches to text_classifier persona."""
    use_classifier()
    from core.llm import classify, persona_system
    result = classify(text, labels, system=persona_system())
    restore_default()
    return result


def run_generate(text: str, style_hint: str = "") -> str:
    """Generate / complete text. Switches to generator persona."""
    use_generator()
    from core.llm import chat, persona_system, build_prompt
    prompt = build_prompt("generate", text=text, style_hint=style_hint,
                          persona=persona_system())
    result = chat(prompt, system=persona_system())
    restore_default()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# A) EXACT-NAME WRAPPERS  ──  NEXUS (Text+Voice paper) — baseline wired already
# Rename / add as the exam demands.  See CHEATSHEET.md for the full map.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Input ────────────────────────────────────────────────────────────
from voice.capture import capture_audio as nexus_capture_audio
# nexus_capture_audio(seconds=7, sample_rate=16000) → float32 array

from voice.stt import transcribe as nexus_transcribe
# nexus_transcribe(audio_array, sample_rate) → {text, language, confidence}


def nexus_get_input(turn_number: int) -> dict:
    """Unified text-or-voice input. Returns {text, source, turn, word_count}."""
    choice = input(f"[Turn {turn_number}] (t)ype or (v)oice? ").strip().lower()
    if choice.startswith("v"):
        audio  = nexus_capture_audio()
        text   = nexus_transcribe(audio, 16000)["text"]
        source = "voice"
    else:
        text   = input("You: ")
        source = "text"
    return {"text": text, "source": source, "turn": turn_number,
            "word_count": len(text.split())}


# ── Stage 2: Wellbeing engine ─────────────────────────────────────────────────
from text.scale      import assess        as assess_wellbeing
from text.scale      import check_and_alert
from text.trajectory import compute_trajectory


# ── Stage 3: Support classifier + responses ───────────────────────────────────
from text.classify import classify as classify_support_need
from text.respond  import respond  as nexus_respond


def log_support_transition(support_log: list, new_primary: str,
                            turn_number: int) -> dict:
    """NEXUS Q3.2: append a transition event when primary category changes."""
    from text.trajectory import log_transition
    prev = support_log[-1].get("primary") if support_log else None
    return log_transition(support_log, prev, new_primary, turn_number)


# ── Stage 5: Report ───────────────────────────────────────────────────────────
def generate_intelligence_report(session_log, wellbeing_log,
                                  support_log, transition_log):
    """NEXUS Q5.1: build the counsellor report from the four logs."""
    from core.state import SessionState
    from text.report import generate_report
    s                 = SessionState()
    s.turns           = session_log
    s.wellbeing_log   = wellbeing_log
    s.support_log     = support_log
    s.transition_log  = transition_log
    return generate_report(s)


# ── Vision (past paper) ───────────────────────────────────────────────────────
from vision.faces import run_blink            # CHALLENGE A — blink detection
from vision.hands import run_gesture          # CHALLENGE C — gesture recognition
from vision.faces import run_all_face         # combined face analysis
from vision.lips  import run_lips             # vowel / lip reading (MediaPipe)


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-CHECK  (python exam_adapter.py)
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== NEXUS wrappers ===")
    wb   = assess_wellbeing("I feel completely hopeless and alone")
    need = classify_support_need("I cannot pay my fees and I cannot sleep")
    resp = nexus_respond("help", need["primary"], wb["tier"])
    print("assess_wellbeing    :", wb)
    print("classify_support    :", need)
    print("nexus_respond       :", resp[:60])

    print()
    print("=== Persona switchers ===")
    use_ner()
    print("active after use_ner()      :", __import__('core.conf', fromlist=['get']).get('llm.active_persona'))
    print("categories.enabled          :", __import__('core.conf', fromlist=['get']).get('categories.enabled'))
    restore_default()
    print("active after restore_default:", __import__('core.conf', fromlist=['get']).get('llm.active_persona'))
    print("categories.enabled          :", __import__('core.conf', fromlist=['get']).get('categories.enabled'))

    print()
    print("=== Task helpers (Ollama required) ===")
    from core.llm import is_alive
    if is_alive():
        ner_result = run_ner("Ali met Sara in Lahore on Monday at 3pm.")
        print("run_ner:", ner_result)
        print("run_qa:", run_qa("What is HCI?",
              "Human-Computer Interaction (HCI) is the study of how people interact with computers."))
    else:
        print("Ollama offline — skipping LLM task helpers")

"""
exam_adapter.py  ===  YOUR EXAM-DAY WORKBENCH  ===

The exam mandates EXACT function names ("any other names score zero") and exact
dict keys. You CANNOT predict them, but the working logic already lives in the
packages. So on exam day you do ONE thing here: define the mandated name as a
2-line wrapper that calls the library.

Below are the NEXUS (Text+Voice) names already wired, as a worked example +
ready-to-submit baseline. Rename / add as the paper demands.

    from exam_adapter import *      # or copy the wrappers you need into app.py
"""

# ---- Stage 1: Input layer -------------------------------------------------
from voice.capture import capture_audio as nexus_capture_audio        # (seconds=7, sample_rate=16000)
from voice.stt import transcribe as nexus_transcribe                  # (audio_array, sample_rate) -> {text,language,confidence}


def nexus_get_input(turn_number: int) -> dict:
    """Unified text-or-voice input with source tracking + word count.
    Returns {'text','source','turn','word_count'}."""
    choice = input(f"[Turn {turn_number}] (t)ype or (v)oice? ").strip().lower()
    if choice.startswith("v"):
        audio = nexus_capture_audio()
        text = nexus_transcribe(audio, 16000)["text"]
        source = "voice"
    else:
        text = input("You: ")
        source = "text"
    return {"text": text, "source": source, "turn": turn_number,
            "word_count": len(text.split())}


# ---- Stage 2: Wellbeing engine -------------------------------------------
from text.scale import assess as assess_wellbeing                     # (text) -> {tier,score,emoji,is_at_risk}
from text.scale import check_and_alert                                # (wellbeing_result, turn_number) -> bool
from text.trajectory import compute_trajectory                        # (wellbeing_log) -> {trend,lowest_tier,at_risk_turns}


# ---- Stage 3: Support classifier + responses -----------------------------
from text.classify import classify as classify_support_need          # (text) -> {primary,all_detected,scores}
from text.respond import respond as nexus_respond                     # (text, support_need, wellbeing) -> str


def log_support_transition(support_log: list, new_primary: str, turn_number: int) -> dict:
    """NEXUS Q3.2 style: append a transition event to support_log when the primary
    category changes from the previous turn. Returns the event (or None)."""
    from text.trajectory import log_transition
    prev = support_log[-1]["curr"] if support_log else None
    return log_transition(support_log, prev, new_primary, turn_number)


# ---- Stage 5: Report ------------------------------------------------------
def generate_intelligence_report(session_log, wellbeing_log, support_log, transition_log):
    """NEXUS Q5.1: build the counsellor report. Accepts the four logs, packs them
    into a SessionState, and prints the banner report."""
    from core.state import SessionState
    from text.report import generate_report
    s = SessionState()
    s.turns = session_log
    s.wellbeing_log = wellbeing_log
    s.support_log = support_log
    s.transition_log = transition_log
    return generate_report(s)


# ---- Vision (past paper) names -------------------------------------------
from vision.faces import run_blink            # CHALLENGE A — blink detection
from vision.hands import run_gesture          # CHALLENGE C — gesture recognition


if __name__ == "__main__":
    # Tiny self-check of the adapter wiring (deterministic, no Ollama/mic needed).
    wb = assess_wellbeing("I feel completely hopeless")
    print("assess_wellbeing:", wb)
    print("classify_support_need:", classify_support_need("I can't pay my rent or my fees"))
    print("nexus_respond:", nexus_respond("help", "FINANCIAL", wb["tier"]))

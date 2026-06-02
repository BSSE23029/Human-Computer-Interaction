"""
pipeline/replay.py -- the "surprise stage": process a PROVIDED log through the
full pipeline with no microphone / camera. Mirrors NEXUS Stage 4.

    from pipeline.replay import run_replay
    run_replay(STUDENT_LOG)                       # deterministic responses
    run_replay(STUDENT_LOG, use_llm_response=True) # llama3 responses
"""
from core.state import SessionState
from pipeline.turn import process_turn
from text.report import generate_report


def run_replay(log_lines, use_llm_response: bool = False, do_report: bool = True) -> SessionState:
    """Feed each line through process_turn() as a text turn; print per-turn analysis
    and (optionally) the final intelligence report. Returns the SessionState."""
    state = SessionState()
    for i, line in enumerate(log_lines):
        r = process_turn(state, text=line, use_llm_response=use_llm_response)
        wb, sup = r["wellbeing"], r["support"]
        flag = "  ⚠ AT-RISK" if wb["is_at_risk"] else ""
        print(f"Turn {i + 1}: {wb['emoji']} {wb['tier']} | {sup['primary']} "
              f"| all: {sup['all_detected']}{flag}")
        print(f"  NEXUS: {r['response']}\n")
    if do_report:
        generate_report(state)
    return state


# A sample log so you can demo replay immediately (replace with the exam's log).
SAMPLE_LOG = [
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
]


if __name__ == "__main__":
    run_replay(SAMPLE_LOG)

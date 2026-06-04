"""
pipeline/replay.py -- offline log replay (the exam "surprise stage").

Two ways to run:

  1. From config (exam day — paste log into config.yaml, run one line):
       from pipeline.replay import run_replay_from_config
       run_replay_from_config()

  2. Passing a list directly (code style, matches NEXUS Stage 4):
       from pipeline.replay import run_replay
       run_replay(STUDENT_LOG)

Both print per-turn analysis and the final intelligence report.
"""

from core.conf import get
from core.state import SessionState
from pipeline.turn import process_turn
from text.report import generate_report


def run_replay(
    log_lines,
    use_llm_response: bool = False,   # False = rule/default (safe for offline)
    do_report: bool = True,
    print_per_turn: bool = True,
) -> SessionState:
    """Feed each line through process_turn() as a text turn.

    Prints per-turn analysis and (optionally) the final report.
    Returns the completed SessionState for further inspection.

    use_llm_response=True  → llama3 writes every response (better quality)
    use_llm_response=False → rule table / default (fast, fully offline)
    """
    state = SessionState()

    for i, line in enumerate(log_lines):
        r = process_turn(state, text=line, use_llm_response=use_llm_response)

        if r.get("session_ended"):
            print(f"[replay] session.max_turns reached at turn {i}. Stopping.")
            break

        wb  = r["wellbeing"]
        sup = r["support"]
        tr  = r["transition"]

        if print_per_turn:
            at_risk_flag = "  ⚠ AT-RISK" if wb.get("is_at_risk") else ""
            esc_flag = "  ↑ ESCALATION" if tr and tr.get("is_escalation") else ""
            print(f"Turn {i+1:02d}: {wb.get('emoji','')} {wb.get('tier','?'):12} | "
                  f"{sup.get('primary','?'):12} | all: {sup.get('all_detected', [])}"
                  f"{at_risk_flag}{esc_flag}")
            print(f"  → {r['response']}\n")

    if do_report:
        generate_report(state)

    return state


def run_replay_from_config(use_llm_response: bool = False) -> SessionState:
    """Run replay using the log defined in config.yaml (replay.log).
    Reads print_per_turn and print_report from config too.

    Exam-day usage:
        python -c "from pipeline.replay import run_replay_from_config; run_replay_from_config()"
    Or click the button in the gradio Report tab.
    """
    log   = get("replay.log") or []
    ppt   = bool(get("replay.print_per_turn", True))
    # report.print is controlled here, NOT inside process_turn (avoids double-print)
    rpt   = bool(get("replay.print_report", True))

    if not log:
        print("[replay] config.replay.log is empty — paste the exam's STUDENT_LOG there first.")
        return SessionState()

    print(f"[replay] running {len(log)} lines from config.yaml ...\n")
    return run_replay(log, use_llm_response=use_llm_response,
                      do_report=rpt, print_per_turn=ppt)


if __name__ == "__main__":
    run_replay_from_config()

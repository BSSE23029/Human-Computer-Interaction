"""
NEXUS — AI-Powered Student Wellbeing Advisor
HCI Assessment | SE305T (Spring-26) | BSSE23B
Author: [Your Name Here]
Date  : 20-May-2026
"""

# ─────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────
import numpy as np
import sounddevice as sd
import whisper
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon if not already present
nltk.download('vader_lexicon', quiet=True)

# ─────────────────────────────────────────────────────────────
#  STAGE 1 — Input Layer: Voice + Text Capture & Transcription
# ─────────────────────────────────────────────────────────────

def nexus_capture_audio(seconds=7, sample_rate=16000):
    """
    Records audio from the default microphone for a given number of seconds.

    Parameters
    ----------
    seconds     : int   – Duration to record (default 7 s).
    sample_rate : int   – Samples per second (default 16 000 Hz, required by Whisper).

    Returns
    -------
    numpy.ndarray (float32)
        Flattened mono audio array ready to be passed directly to nexus_transcribe().
    """
    print(f"[NEXUS] 🎙  Recording for {seconds} seconds — speak now...")
    for remaining in range(seconds, 0, -1):
        print(f"  ⏱  {remaining} second(s) remaining...", end="\r", flush=True)
        import time; time.sleep(1)

    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()          # block until recording is complete
    print("\n[NEXUS] ✅ Recording complete.")
    return audio.flatten()


def nexus_transcribe(audio_array, sample_rate):
    """
    Transcribes a numpy audio array using the OpenAI Whisper 'small' model.

    Parameters
    ----------
    audio_array : numpy.ndarray (float32) – Raw audio samples.
    sample_rate : int                     – Sample rate of the audio.

    Returns
    -------
    dict
        {
            'text'      : str  – Transcribed text.
            'language'  : str  – Detected language code (e.g. 'en').
            'confidence': str  – 'high' if len(text) > 10 chars, else 'low'.
        }
    """
    model = whisper.load_model("small")

    # Whisper expects float32 at 16 kHz; pad/trim to 30 s
    audio_input = whisper.pad_or_trim(audio_array.astype(np.float32))
    mel = whisper.log_mel_spectrogram(audio_input).to(model.device)

    # Detect language
    _, probs = model.detect_language(mel)
    language = max(probs, key=probs.get)

    # Decode
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    text = result.text.strip()

    confidence = 'high' if len(text) > 10 else 'low'

    return {
        'text'      : text,
        'language'  : language,
        'confidence': confidence,
    }


def nexus_get_input(turn_number):
    """
    Unified input handler.  Prompts the student to choose voice or text entry,
    captures the input, and returns a standardised dict with source tracking.

    Parameters
    ----------
    turn_number : int – The current conversation turn (displayed in the prompt).

    Returns
    -------
    dict
        {
            'text'      : str  – The student's message.
            'source'    : str  – 'voice' or 'text'.
            'turn'      : int  – turn_number echoed back.
            'word_count': int  – Number of whitespace-delimited words.
        }
    """
    print(f"\n{'─'*50}")
    print(f"[NEXUS] Turn {turn_number} — How would you like to share what's on your mind?")
    print("  [1] Type your message")
    print("  [2] Speak your message")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '2':
        sample_rate = 16000
        audio = nexus_capture_audio(seconds=7, sample_rate=sample_rate)
        result = nexus_transcribe(audio, sample_rate)
        text   = result['text']
        source = 'voice'
        print(f"[NEXUS] 📝 Transcribed: \"{text}\"")
    else:
        text   = input(f"[Turn {turn_number}] You: ").strip()
        source = 'text'

    word_count = len(text.split()) if text else 0

    return {
        'text'      : text,
        'source'    : source,
        'turn'      : turn_number,
        'word_count': word_count,
    }


# ─────────────────────────────────────────────────────────────
#  STAGE 2 — Wellbeing Engine: 6-Tier Emotional State Tracker
# ─────────────────────────────────────────────────────────────

sia = SentimentIntensityAnalyzer()

WELLBEING_SCALE = [
    ('THRIVING',   0.60,          float('inf'), '🌟'),
    ('CONTENT',    0.20,          0.60,          '😊'),
    ('NEUTRAL',   -0.19,          0.20,          '😐'),
    ('STRESSED',  -0.40,         -0.19,          '😟'),
    ('DISTRESSED',-0.60,         -0.40,          '😢'),
    ('CRISIS',     float('-inf'), -0.60,         '🆘'),
]


def assess_wellbeing(text):
    """
    Analyses the emotional content of *text* using VADER and maps the compound
    score onto the NEXUS 6-tier wellbeing scale.

    Parameters
    ----------
    text : str – Student's message.

    Returns
    -------
    dict  { 'tier': str, 'score': float, 'emoji': str, 'is_at_risk': bool }
    """
    score = sia.polarity_scores(text)['compound']

    tier  = 'NEUTRAL'
    emoji = '😐'
    for name, low, high, em in WELLBEING_SCALE:
        if low <= score < high:
            tier  = name
            emoji = em
            break

    is_at_risk = (tier == 'CRISIS')

    return {
        'tier'      : tier,
        'score'     : score,
        'emoji'     : emoji,
        'is_at_risk': is_at_risk,
    }


def compute_trajectory(wellbeing_log):
    """
    Computes the emotional trajectory across a session.

    Parameters
    ----------
    wellbeing_log : list[dict] – List of dicts returned by assess_wellbeing().

    Returns
    -------
    dict
        {
            'trend'       : str        – 'improving', 'declining', or 'fluctuating'.
            'lowest_tier' : str        – The worst tier reached during the session.
            'at_risk_turns': list[int] – 0-based indices of CRISIS turns.
        }
    """
    if not wellbeing_log:
        return {'trend': 'fluctuating', 'lowest_tier': 'NEUTRAL', 'at_risk_turns': []}

    scores = [entry['score'] for entry in wellbeing_log]
    n      = len(scores)
    mid    = n // 2

    first_half  = scores[:mid]  if mid > 0 else scores
    second_half = scores[mid:]  if mid > 0 else scores

    first_avg  = sum(first_half)  / len(first_half)  if first_half  else 0.0
    second_avg = sum(second_half) / len(second_half) if second_half else 0.0

    if second_avg > first_avg + 0.1:
        trend = 'improving'
    elif second_avg < first_avg - 0.1:
        trend = 'declining'
    else:
        trend = 'fluctuating'

    # Lowest tier = entry with the minimum compound score
    worst_entry = min(wellbeing_log, key=lambda e: e['score'])
    lowest_tier = worst_entry['tier']

    # 0-based indices of CRISIS turns
    at_risk_turns = [i for i, e in enumerate(wellbeing_log) if e['is_at_risk']]

    return {
        'trend'        : trend,
        'lowest_tier'  : lowest_tier,
        'at_risk_turns': at_risk_turns,
    }


def check_and_alert(wellbeing_result, turn_number):
    """
    Prints a prominent crisis alert if the student is flagged at-risk.

    Parameters
    ----------
    wellbeing_result : dict – Output of assess_wellbeing().
    turn_number      : int  – Current turn (displayed in the alert).

    Returns
    -------
    bool – True if the student is at-risk, False otherwise.
    """
    if wellbeing_result.get('is_at_risk'):
        print("\n" + "⚠️ " * 20)
        print(f"  🆘  CRISIS ALERT — Turn {turn_number}")
        print("  A student in crisis has been detected.")
        print("  IMMEDIATE ACTION REQUIRED:")
        print("  Please contact the University Counselling Line: 0800-XXX-XXXX")
        print("  Or visit the Student Support Centre immediately.")
        print("⚠️ " * 20 + "\n")
    return wellbeing_result.get('is_at_risk', False)


# ─────────────────────────────────────────────────────────────
#  STAGE 3 — Support Classifier & At-Risk Flag System
# ─────────────────────────────────────────────────────────────

SUPPORT_KEYWORDS = {
    'ACADEMIC' : ['assignment','deadline','exam','grade','fail','pass',
                  'lecture','study','professor','submit'],
    'WELLBEING': ['stress','anxious','depressed','lonely','overwhelmed',
                  'panic','cry','hopeless','afraid'],
    'FINANCIAL': ['fees','scholarship','loan','afford','money','rent',
                  'bursary','payment','debt'],
    'TECHNICAL': ['portal','login','password','system','error','access',
                  'email','vpn','reset'],
    'SOCIAL'   : ['friends','roommate','belong','isolated','group',
                  'relationship','community'],
    'ADMIN'    : ['enrolment','certificate','transcript','registration',
                  'form','office'],
}


def classify_support_need(text):
    """
    Multi-label classifier that scores every support category against the
    student's message using keyword matching.

    Parameters
    ----------
    text : str – Student's message.

    Returns
    -------
    dict
        {
            'primary'     : str        – Highest-scoring category.
            'all_detected': list[str]  – All categories with score > 0.
            'scores'      : dict       – Raw counts per category.
        }
    """
    lower = text.lower()
    scores = {}

    for category, keywords in SUPPORT_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in lower)
        scores[category] = count

    all_detected = [cat for cat, sc in scores.items() if sc > 0]

    if all_detected:
        primary = max(scores, key=scores.get)
    else:
        primary = 'GENERAL'
        all_detected = ['GENERAL']

    return {
        'primary'     : primary,
        'all_detected': all_detected,
        'scores'      : scores,
    }


def log_support_transition(support_log, new_primary, turn_number):
    """
    Appends a transition event to *support_log* when the primary support
    category changes between turns.

    Parameters
    ----------
    support_log  : list  – Mutable list; transition dicts are appended here.
    new_primary  : str   – Primary category identified for the current turn.
    turn_number  : int   – Current turn number.

    Returns
    -------
    list – The updated support_log (same object, mutated in place).
    """
    if not support_log:
        # Nothing to compare on the first turn
        support_log.append({'primary': new_primary, 'turn': turn_number})
        return support_log

    prev_entry   = support_log[-1]
    prev_primary = prev_entry.get('primary', 'GENERAL')

    if prev_primary != new_primary:
        is_escalation = (new_primary == 'WELLBEING') and (prev_primary != 'WELLBEING')
        event = {
            'prev'         : prev_primary,
            'curr'         : new_primary,
            'turn'         : turn_number,
            'is_escalation': is_escalation,
        }
        support_log.append(event)
        if is_escalation:
            print(f"[NEXUS] ⬆  Escalation detected at Turn {turn_number}: "
                  f"{prev_primary} → WELLBEING")
    else:
        support_log.append({'primary': new_primary, 'turn': turn_number})

    return support_log


def nexus_respond(text, support_need, wellbeing):
    """
    Generates an empathetic response based on the support category and
    wellbeing tier.

    Parameters
    ----------
    text         : str – Student's message (reserved for future NLP expansion).
    support_need : str – Primary support category.
    wellbeing    : str – Wellbeing tier string (e.g. 'CRISIS').

    Returns
    -------
    str – NEXUS response text.
    """
    # ── Priority overrides ────────────────────────────────────────────
    if support_need == 'WELLBEING' and wellbeing == 'CRISIS':
        return ("I am very concerned about you. "
                "Please contact the university counselling line RIGHT NOW: 0800-XXX-XXXX.")

    if support_need == 'WELLBEING' and wellbeing == 'DISTRESSED':
        return ("It sounds like you are going through a really difficult time. "
                "Have you spoken to anyone about how you are feeling?")

    if support_need == 'ACADEMIC' and wellbeing == 'STRESSED':
        return ("Exam pressure is real. "
                "Let us look at what support your faculty offers — "
                "have you spoken to your tutor?")

    if support_need == 'FINANCIAL':
        return ("Financial difficulty is more common than you think. "
                "The university bursary office can help — "
                "shall I give you their contact?")

    if support_need == 'TECHNICAL':
        return ("Let me help you with that technical issue. "
                "Which system are you trying to access?")

    if support_need == 'SOCIAL' and wellbeing == 'DISTRESSED':
        return ("Feeling isolated at university is incredibly hard. "
                "The student union runs weekly social events — would that help?")

    # ── General fallbacks ─────────────────────────────────────────────
    fallbacks = {
        'ACADEMIC' : ("I understand you are facing academic challenges. "
                      "Your university's learning support team is here to help — "
                      "have you checked the student portal for resources?"),
        'WELLBEING': ("Thank you for sharing that with me. "
                      "Your wellbeing matters. "
                      "Would you like me to connect you with student support services?"),
        'FINANCIAL': ("Financial stress can be overwhelming. "
                      "The bursary office offers emergency funds and payment plans — "
                      "shall I provide their details?"),
        'TECHNICAL': ("Let me help you resolve that technical issue. "
                      "Which system or portal are you having trouble with?"),
        'SOCIAL'   : ("Feeling connected is important. "
                      "The student union and campus societies are great places to meet people — "
                      "would you like some suggestions?"),
        'ADMIN'    : ("I can help you navigate the administrative process. "
                      "What specific document or registration step do you need help with?"),
        'GENERAL'  : ("I am here to listen and help. "
                      "Could you tell me a little more about what you are going through?"),
    }
    return fallbacks.get(support_need,
                         "I am here to help. Could you tell me more about what you need?")


# ─────────────────────────────────────────────────────────────
#  STAGE 4 — Offline Session Replay (Surprise Challenge)
# ─────────────────────────────────────────────────────────────

STUDENT_LOG = [
    'Hi, I need some help please.',
    'I have a major assignment due tomorrow and I have not started.',
    'My laptop also broke yesterday so I cannot access my files.',
    'To be honest I have been struggling a lot lately, not just academically.',
    'I have not been sleeping, I feel completely hopeless about everything.',
    'I think I might need to talk to someone but I do not know who.',
    'Also I got an email saying my fees are overdue and I cannot register.',
    'Sorry for dumping all this. I just feel very alone right now.',
    'Actually, my friend just texted. I feel a tiny bit better now.',
    'Thank you for listening. I will try to contact the counsellor.',
]


def run_offline_replay():
    """Processes STUDENT_LOG through the full NEXUS pipeline and prints results."""
    print("\n" + "═"*55)
    print("  NEXUS — OFFLINE SESSION REPLAY")
    print("═"*55)

    wellbeing_log  = []
    session_log    = []
    support_log    = []
    transition_log = []

    for i, msg in enumerate(STUDENT_LOG):
        wb   = assess_wellbeing(msg)
        need = classify_support_need(msg)
        alert = check_and_alert(wb, i + 1)
        resp  = nexus_respond(msg, need['primary'], wb['tier'])

        # Store logs
        wellbeing_log.append(wb)
        session_log.append({
            'text'      : msg,
            'source'    : 'text',
            'turn'      : i + 1,
            'word_count': len(msg.split()),
        })

        # Track support transitions
        if support_log:
            prev_primary = support_log[-1].get('primary', 'GENERAL')
            if prev_primary != need['primary']:
                is_esc = (need['primary'] == 'WELLBEING')
                transition_log.append({
                    'prev'         : prev_primary,
                    'curr'         : need['primary'],
                    'turn'         : i + 1,
                    'is_escalation': is_esc,
                })
        support_log.append({'primary': need['primary'], 'turn': i + 1})

        print(f"\nTurn {i+1}: {wb['emoji']} {wb['tier']} | {need['primary']} "
              f"| all: {need['all_detected']}")
        print(f"NEXUS: {resp}")

    # Final intelligence report
    generate_intelligence_report(session_log, wellbeing_log, support_log, transition_log)


# ─────────────────────────────────────────────────────────────
#  STAGE 5 — Intelligence Report & Submission
# ─────────────────────────────────────────────────────────────

def generate_intelligence_report(session_log, wellbeing_log, support_log, transition_log):
    """
    Generates a structured counsellor intelligence report and prints it to console.

    Parameters
    ----------
    session_log    : list[dict] – Input records from nexus_get_input() or replay.
    wellbeing_log  : list[dict] – Records from assess_wellbeing().
    support_log    : list[dict] – Records from classify_support_need().
    transition_log : list[dict] – Records from log_support_transition().
    """
    print("\n" + "="*55)
    print("  NEXUS STUDENT INTELLIGENCE REPORT")
    print("  Code: NX-2B — Counsellor Eyes Only")
    print("="*55)

    # ── Session stats ─────────────────────────────────────────
    total_turns  = len(session_log)
    voice_turns  = sum(1 for s in session_log if s.get('source') == 'voice')
    text_turns   = total_turns - voice_turns

    print(f"\n📋 SESSION OVERVIEW")
    print(f"  Total turns : {total_turns}")
    print(f"  Voice turns : {voice_turns}")
    print(f"  Text turns  : {text_turns}")

    # ── Wellbeing trajectory ──────────────────────────────────
    trajectory = compute_trajectory(wellbeing_log) if wellbeing_log else \
                 {'trend': 'N/A', 'lowest_tier': 'N/A', 'at_risk_turns': []}

    print(f"\n📈 WELLBEING ANALYSIS")
    print(f"  Trajectory    : {trajectory['trend'].upper()}")
    print(f"  Lowest tier   : {trajectory['lowest_tier']}")

    at_risk_count = len(trajectory['at_risk_turns'])
    at_risk_turns_display = (
        ', '.join(str(t + 1) for t in trajectory['at_risk_turns'])
        if trajectory['at_risk_turns'] else 'None'
    )
    print(f"  At-risk alerts: {at_risk_count}  (turns: {at_risk_turns_display})")

    # ── Average wellbeing score ───────────────────────────────
    avg_score = (sum(e['score'] for e in wellbeing_log) / len(wellbeing_log)
                 if wellbeing_log else 0.0)
    print(f"  Avg wellbeing score: {avg_score:.3f}")

    # ── Support categories ────────────────────────────────────
    from collections import Counter
    cat_counter = Counter(
        e['primary'] for e in support_log if 'primary' in e
    )
    print(f"\n🗂  SUPPORT CATEGORIES DETECTED (ranked by frequency)")
    for cat, freq in cat_counter.most_common():
        print(f"  {cat:<12} : {freq} turn(s)")

    # ── Escalation events ─────────────────────────────────────
    escalations = [t for t in transition_log if t.get('is_escalation')]
    print(f"\n🔺 SUPPORT ESCALATION EVENTS")
    if escalations:
        for ev in escalations:
            print(f"  Turn {ev['turn']}: {ev['prev']} → {ev['curr']}")
    else:
        print("  None recorded.")

    # ── Risk score ────────────────────────────────────────────
    base_risk          = 20
    wellbeing_penalty  = abs(avg_score) * 40
    at_risk_penalty    = at_risk_count  * 15
    escalation_penalty = len(escalations) * 10

    avg_words = (sum(s.get('word_count', 0) for s in session_log) / total_turns
                 if total_turns > 0 else 0)
    word_count_factor = 5 if avg_words > 20 else 0

    risk_score = (base_risk + wellbeing_penalty + at_risk_penalty
                  + escalation_penalty + word_count_factor)
    risk_score = max(0, min(100, round(risk_score)))

    if risk_score >= 70:
        action = 'URGENT_REFERRAL'
    elif risk_score >= 40:
        action = 'FOLLOW_UP'
    else:
        action = 'NO_ACTION'

    print(f"\n⚠️  RISK ASSESSMENT")
    print(f"  Risk score         : {risk_score} / 100")
    print(f"  Avg words / turn   : {avg_words:.1f}")
    print(f"  Recommended action : {action}")

    print("\n" + "="*55)


# ─────────────────────────────────────────────────────────────
#  STAGE 1 Verification
# ─────────────────────────────────────────────────────────────

def verify_stage1():
    result = nexus_get_input(turn_number=1)
    assert 'text'       in result and 'source' in result, "Missing 'text' or 'source'"
    assert 'turn'       in result and 'word_count' in result, "Missing 'turn' or 'word_count'"
    assert result['source'] in ('voice', 'text'), "Invalid source value"
    print('Stage 1 OK:', result)


# ─────────────────────────────────────────────────────────────
#  Interactive Session Loop (optional live demo)
# ─────────────────────────────────────────────────────────────

def run_interactive_session(max_turns=10):
    """
    Runs a live multi-turn NEXUS session using nexus_get_input().
    Generates the intelligence report at the end.
    """
    print("\n" + "═"*55)
    print("  Welcome to NEXUS — Student Wellbeing Advisor")
    print("  Type your message or speak. Type 'exit' to end.")
    print("═"*55)

    session_log    = []
    wellbeing_log  = []
    support_log    = []
    transition_log = []

    for turn in range(1, max_turns + 1):
        user_input = nexus_get_input(turn_number=turn)

        if user_input['text'].lower() in ('exit', 'quit', 'bye'):
            print("[NEXUS] Thank you for talking with me. Take care.")
            break

        wb    = assess_wellbeing(user_input['text'])
        need  = classify_support_need(user_input['text'])
        alert = check_and_alert(wb, turn)
        resp  = nexus_respond(user_input['text'], need['primary'], wb['tier'])

        wellbeing_log.append(wb)
        session_log.append(user_input)

        # Support transition logging
        if len(support_log) > 0:
            prev_primary = support_log[-1].get('primary', 'GENERAL')
            if prev_primary != need['primary']:
                is_esc = (need['primary'] == 'WELLBEING')
                transition_log.append({
                    'prev'         : prev_primary,
                    'curr'         : need['primary'],
                    'turn'         : turn,
                    'is_escalation': is_esc,
                })
        support_log.append({'primary': need['primary'], 'turn': turn})

        print(f"\n  {wb['emoji']} Wellbeing: {wb['tier']}  |  Support: {need['primary']}")
        print(f"  NEXUS: {resp}\n")

    generate_intelligence_report(session_log, wellbeing_log, support_log, transition_log)


# ─────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else 'replay'

    if mode == 'verify':
        verify_stage1()
    elif mode == 'live':
        run_interactive_session()
    else:
        # Default: offline replay (Stage 4 — no microphone needed)
        run_offline_replay()

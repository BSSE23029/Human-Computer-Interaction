# ================================================================
# ARIA — AI-powered Customer Support Agent
# HCI Assessment (CLO-3) — ITU, Spring 2026
# Student : Muhammad Mukarram Raza
# Roll No : 10-29-75
# ================================================================

import sys
import os
import tempfile

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure VADER lexicon is present
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# ================================================================
# Stage 1 — Foundation: Voice + Text Input Pipeline
# ================================================================

def record_voice(duration=6, sr=16000):
    """Record audio from microphone for `duration` seconds at `sr` Hz."""
    print(f"  [Recording for {duration} seconds... speak now]")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()


def transcribe(audio, sr=16000):
    """Save audio to a temp WAV file and transcribe using Whisper 'base'."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp_path = f.name

    try:
        audio_int16 = (audio * 32767).astype(np.int16)
        wav.write(tmp_path, sr, audio_int16)

        model = whisper.load_model('base')
        result = model.transcribe(tmp_path)
        text = result.get('text', '').strip()
        return text if text else ''
    finally:
        os.unlink(tmp_path)


def get_user_input():
    """Prompt user to type a message or press ENTER to speak.

    Returns (text, source) where source is 'voice' or 'text'.
    """
    raw = input("Type a message (or press ENTER to speak): ")

    if raw.strip() == '':
        audio = record_voice()
        text  = transcribe(audio)
        if not text:
            text = '[no speech detected]'
        return (text, 'voice')

    return (raw.strip(), 'text')


# ================================================================
# Stage 2 — Emotion Engine: 5-Level Sentiment Analyser
# ================================================================

sia = SentimentIntensityAnalyzer()

EMOJI_MAP = {
    'ELATED':     '😊',
    'POSITIVE':   '🙂',
    'NEUTRAL':    '😐',
    'NEGATIVE':   '😕',
    'FRUSTRATED': '😤',
}

EMOTION_RANK = {
    'FRUSTRATED': 0,
    'NEGATIVE':   1,
    'NEUTRAL':    2,
    'POSITIVE':   3,
    'ELATED':     4,
}


def classify_emotion(text):
    """Classify text into one of 5 emotion states using VADER compound score.

    Returns {'label': str, 'score': float, 'emoji': str}.
    """
    score = sia.polarity_scores(text)['compound']

    if score >= 0.5:
        label = 'ELATED'
    elif score >= 0.05:
        label = 'POSITIVE'
    elif score >= -0.04:
        label = 'NEUTRAL'
    elif score > -0.5:
        label = 'NEGATIVE'
    else:
        label = 'FRUSTRATED'

    return {'label': label, 'score': score, 'emoji': EMOJI_MAP[label]}


def detect_mood_shift(prev_label, curr_label):
    """Return 'escalating', 'improving', or 'stable' based on emotion rank."""
    prev_rank = EMOTION_RANK.get(prev_label, 2)
    curr_rank = EMOTION_RANK.get(curr_label, 2)

    if curr_rank < prev_rank:
        return 'escalating'
    if curr_rank > prev_rank:
        return 'improving'
    return 'stable'


def session_emotion_summary(emotion_log):
    """Return dominant label, avg score, and escalation count for the session."""
    if not emotion_log:
        return {'dominant': 'NEUTRAL', 'avg_score': 0.0, 'escalation_count': 0}

    scores = [e['score'] for e in emotion_log]
    avg_score = sum(scores) / len(scores)

    label_counts = {}
    for e in emotion_log:
        label_counts[e['label']] = label_counts.get(e['label'], 0) + 1
    dominant = max(label_counts, key=label_counts.get)

    escalation_count = 0
    for i in range(1, len(emotion_log)):
        if detect_mood_shift(emotion_log[i - 1]['label'], emotion_log[i]['label']) == 'escalating':
            escalation_count += 1

    return {
        'dominant':         dominant,
        'avg_score':        avg_score,
        'escalation_count': escalation_count,
    }


# ================================================================
# Stage 3 — Intent Tracker & Conversation State Machine
# ================================================================

INTENT_KEYWORDS = {
    'COMPLAINT': ['problem', 'broken', 'error', 'fail', 'wrong', 'terrible', 'awful'],
    'INQUIRY':   ['what', 'how', 'when', 'where', 'explain', 'tell me', 'does'],
    'SUPPORT':   ['help', 'stuck', 'assist', 'cannot', 'unable'],
    'FEEDBACK':  ['suggest', 'improve', 'recommend', 'wish'],
    'BILLING':   ['charge', 'payment', 'invoice', 'refund', 'price', 'cost', 'money'],
    'FAREWELL':  ['bye', 'goodbye', 'thanks', 'thank you', 'done'],
}


def classify_intent(text):
    """Return best-matching intent using keyword scoring; 'GENERAL' if no match."""
    t = text.lower()
    scores = {}

    for intent, keywords in INTENT_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in t)
        if count > 0:
            scores[intent] = count

    if not scores:
        return 'GENERAL'
    return max(scores, key=scores.get)


def update_intent_log(intent_log, new_intent, turn_number):
    """Append a turn record; include a shift sub-dict when the intent changes.

    Each record: {'intent': str, 'turn': int} + optional 'shift': {'from', 'to', 'turn'}.
    Returns the updated log.
    """
    entry = {'intent': new_intent, 'turn': turn_number}

    if intent_log and intent_log[-1]['intent'] != new_intent:
        entry['shift'] = {
            'from': intent_log[-1]['intent'],
            'to':   new_intent,
            'turn': turn_number,
        }

    intent_log.append(entry)
    return intent_log


def generate_response(text, intent, emotion):
    """Return a response string based on intent + emotion combination."""
    label = emotion if isinstance(emotion, str) else emotion.get('label', 'NEUTRAL')

    # COMPLAINT
    if intent == 'COMPLAINT' and label == 'FRUSTRATED':
        return 'I sincerely apologise. Let me escalate this to our senior team right now.'
    if intent == 'COMPLAINT':
        return 'I understand there is an issue. Can you share more details?'

    # BILLING
    if intent == 'BILLING' and label in ('NEGATIVE', 'FRUSTRATED'):
        return 'I am sorry about the billing concern. I will review your account immediately.'
    if intent == 'BILLING':
        return 'I can help with your billing query. Let me pull up your account details.'

    # FAREWELL — any emotion
    if intent == 'FAREWELL':
        return 'Thank you for contacting us. Have a great day!'

    # INQUIRY — any emotion
    if intent == 'INQUIRY':
        return 'Great question. Here is what I know: please hold while I look that up for you.'

    # SUPPORT
    if intent == 'SUPPORT' and label in ('FRUSTRATED', 'NEGATIVE'):
        return 'I am here to help. Let me guide you through this step by step.'
    if intent == 'SUPPORT':
        return 'Of course! I am happy to assist you. What do you need help with?'

    # FEEDBACK
    if intent == 'FEEDBACK':
        return 'Thank you for your feedback. We will pass it on to our product team.'

    # GENERAL fallback with emotion awareness
    if label == 'FRUSTRATED':
        return 'I understand your frustration. I am here to help — please tell me more.'
    if label == 'NEGATIVE':
        return 'I am sorry you are having a difficult experience. How can I assist?'

    return 'Thank you for reaching out. How can I assist you today?'


# ================================================================
# Stage 4 — Replay & Audit
# ================================================================

CONVO_LOG = [
    'Hi, I need some help with my account.',
    'I have been trying to log in for two days but it keeps failing.',
    'This is really frustrating, I cannot believe this is still broken.',
    'Can you at least tell me when it will be fixed?',
    'Also I was charged twice last month — I want a refund.',
    'Actually you know what, forget it. I am going to cancel.',
    'Wait — actually I found the issue, it was my browser. Never mind!',
    'Thanks for your patience. Bye.',
]


def run_replay():
    """Process CONVO_LOG through the full ARIA pipeline (no microphone)."""
    emotion_log        = []
    intent_log         = []
    history            = []
    prev_emotion_label = None

    print('\n' + '=' * 60)
    print('  ARIA — REPLAY & AUDIT (Stage 4)')
    print('=' * 60 + '\n')

    for i, msg in enumerate(CONVO_LOG):
        emotion  = classify_emotion(msg)
        intent   = classify_intent(msg)
        shift    = detect_mood_shift(prev_emotion_label, emotion['label']) \
            if prev_emotion_label is not None else 'stable'
        response = generate_response(msg, intent, emotion['label'])

        emotion_log.append(emotion)
        intent_log = update_intent_log(intent_log, intent, i + 1)
        history.append({
            'text':    msg,
            'source':  'text',
            'intent':  intent,
            'emotion': emotion,
            'shift':   shift,
        })

        print(f"Turn {i+1}: {emotion['emoji']} {emotion['label']} | {intent} | {shift}")
        print(f"  User : {msg}")
        print(f"  ARIA : {response}\n")

        prev_emotion_label = emotion['label']

    print_health_report(history, emotion_log, intent_log)


# ================================================================
# Stage 5 — Session Health Report
# ================================================================

def print_health_report(history, emotion_log, intent_log):
    """Print the ARIA session health report."""
    print('=' * 50)
    print('     ARIA SESSION HEALTH REPORT')
    print('=' * 50)

    # Turn counts
    total_turns = len(history)
    voice_turns = sum(1 for h in history if h.get('source') == 'voice')
    text_turns  = total_turns - voice_turns
    print(f"Total Turns     : {total_turns}  (Voice: {voice_turns} | Text: {text_turns})")

    # Unique intents in order first seen
    seen_intents = list(dict.fromkeys(e['intent'] for e in intent_log))
    print(f"Intents Seen    : {', '.join(seen_intents)}")

    # Intent shift events
    shift_events = [e for e in intent_log if 'shift' in e]
    print(f"Intent Shifts   : {len(shift_events)}")

    # Emotion summary
    summary = session_emotion_summary(emotion_log)
    dominant_label = summary['dominant']
    print(f"Dominant Emotion: {dominant_label} {EMOJI_MAP.get(dominant_label, '')}")
    print(f"Avg Sentiment   : {summary['avg_score']:.2f}")
    print(f"Escalations     : {summary['escalation_count']}")

    # Session Health Score
    base_score     = 50
    emotion_bonus  = summary['avg_score'] * 30
    shift_penalty  = len(shift_events) * 5
    escalation_pen = summary['escalation_count'] * 8
    health_score   = base_score + emotion_bonus - shift_penalty - escalation_pen
    health_score   = max(0, min(100, round(health_score)))

    if health_score >= 70:
        recommendation = 'RESOLVED'
    elif health_score >= 40:
        recommendation = 'MONITOR'
    else:
        recommendation = 'ESCALATE'

    print(f"Health Score    : {health_score}/100")
    print(f"Recommendation  : {recommendation}")
    print('=' * 50)


# ================================================================
# Live ARIA Session (interactive)
# ================================================================

def run_live_session():
    """Run a live interactive ARIA session with voice + text input."""
    print('\n' + '=' * 60)
    print('  ARIA — AI Customer Support Agent  (Live Session)')
    print('  Type your message or press ENTER to speak.')
    print('  Say / type "bye" to end the session.')
    print('=' * 60 + '\n')

    emotion_log        = []
    intent_log         = []
    history            = []
    prev_emotion_label = None
    turn               = 0

    while True:
        turn += 1
        text, source = get_user_input()

        if not text or text == '[no speech detected]':
            print("ARIA: I did not catch that. Could you try again?\n")
            turn -= 1
            continue

        emotion  = classify_emotion(text)
        intent   = classify_intent(text)
        shift    = detect_mood_shift(prev_emotion_label, emotion['label']) \
            if prev_emotion_label is not None else 'stable'
        response = generate_response(text, intent, emotion['label'])

        emotion_log.append(emotion)
        intent_log = update_intent_log(intent_log, intent, turn)
        history.append({
            'text':    text,
            'source':  source,
            'intent':  intent,
            'emotion': emotion,
            'shift':   shift,
        })

        print(f"\n[{source.upper()}] Turn {turn}: "
              f"{emotion['emoji']} {emotion['label']} | {intent} | {shift}")
        print(f"ARIA: {response}\n")

        prev_emotion_label = emotion['label']

        if intent == 'FAREWELL':
            break

    print_health_report(history, emotion_log, intent_log)


# ================================================================
# Stage 1 verification helper (as specified in the assessment)
# ================================================================

def verify_stage1():
    text, source = get_user_input()
    assert isinstance(text, str)
    assert source in ('voice', 'text')
    print(f'[{source.upper()}] {text}')


# ================================================================
# Entry Point
# ================================================================

if __name__ == '__main__':
    if '--replay' in sys.argv:
        run_replay()
    elif '--live' in sys.argv:
        run_live_session()
    else:
        print("ARIA — AI Customer Support Agent")
        print("  1. Run Stage-4 Replay  (no microphone needed)")
        print("  2. Start Live Session  (voice + text)")
        choice = input("Choose (1/2): ").strip()
        if choice == '1':
            run_replay()
        else:
            run_live_session()

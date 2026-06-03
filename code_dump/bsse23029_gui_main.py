# ================================================================
# ITU Multi-Modal Chatbot System — HCI Assignment 5
# Single-file implementation
# ================================================================

import sys
import threading
import random
import pyttsx3
import speech_recognition as sr
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit, QStackedWidget, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QTextCursor

# ================================================================
# ── DESIGN TOKENS
# ================================================================

APP = {
    "title":       "NEXUS",
    "window_size": (1920, 1080),
}

# Single source of truth for all colours
C = {
    # Single interactive accent — the ONE "click me" signal
    "primary":          "#0066cc",
    "primary_hover":    "#0071e3",
    "primary_on_dark":  "#2997ff",   # blue on near-black surfaces

    # Surfaces — light ↔ near-black alternation is the section divider
    "canvas":           "#ffffff",
    "canvas_parchment": "#f5f5f7",   # off-white
    "surface_pearl":    "#fafafc",   # near-white for ghost button fills
    "surface_dark":     "#1d1d1f",   # near-black (global nav)
    "surface_dark_alt": "#2c2c2e",   # slightly lighter dark tile

    # Text
    "ink":              "#1d1d1f",   # main body on light surfaces
    "ink_secondary":    "#3a3a3c",
    "ink_muted":        "#6e6e73",   # captions, placeholders, status
    "on_dark":          "#ffffff",
    "on_dark_muted":    "#aeaeb2",   # secondary text on dark surfaces

    # Hairlines — never decorative, only structural
    "hairline":         "#d2d2d7",
    "divider":          "#e8e8ed",

    # system semantic colours (used sparingly)
    "system_green":     "#34c759",
    "system_orange":    "#ff9500",
    "system_red":       "#ff3b30",
    "system_yellow":    "#ffd60a",
}

# Typography scale — (family, pt-size, weight)
# SF Pro resolves to the real typeface on macOS; falls back gracefully elsewhere.
T = {
    "display":   ("", 24, QFont.Weight.DemiBold),
    "title_1":   ("", 20, QFont.Weight.DemiBold),
    "title_2":   ("", 17, QFont.Weight.DemiBold),
    "body":      ("", 15, QFont.Weight.Normal),
    "callout":   ("", 13, QFont.Weight.Normal),
    "subhead":   ("", 13, QFont.Weight.DemiBold),
    "footnote":  ("", 11, QFont.Weight.Normal),
    "caption":   ("", 10, QFont.Weight.Normal),
}

def _font(token: str, italic: bool = False) -> QFont:
    family, size, weight = T[token]
    f = QFont(family, size)
    f.setWeight(weight)
    if italic:
        f.setItalic(True)
    return f

TTS = {"rate": 160}

STT = {
    "timeout":             5,
    "phrase_time_limit":   7,
    "ambient_calibration": 0.5,
}

MODES = [
    ("Review Mode",  "Review"),
    ("Live Mode", "Live"),
    # ("Hybrid Mode",      "Hybrid"),
]

# ================================================================
# ── NLP DATA
# ================================================================

# SPAM_KEYWORDS = [
#     # "win prize", "crypto", "lottery", "free cash", "subscribe", "buy now",
#     # "bitcoin", "investment offer", "earn money", "click here", "limited offer",
#     # "exclusive deal", "make money fast", "get rich",
# ]

INTENT_MAP = {
'ACADEMIC': ['assignment','deadline','exam','grade','fail','pass',
'lecture','study','professor','submit'],
'WELLBEING': ['stress','anxious','depressed','lonely','overwhelmed',
'panic','cry','hopeless','afraid'],
'FINANCIAL': ['fees','scholarship','loan','afford','money','rent',
'bursary','payment','debt'],
'TECHNICAL': ['portal','login','password','system','error','access',
'email','VPN','reset'],
'SOCIAL': ['friends','roommate','belong','isolated','group',
'relationship','community'],
'ADMIN': ['enrolment','certificate','transcript','registration',
'form','office'],
}
# RESPONSES = {
#     "Greeting": [
#         "Hello! I'm your ITU Assistant. How can I help you today? I can answer questions about Admissions, Fees, Courses, and Schedules.",
#         "Hi there! Welcome to ITU's chatbot. Feel free to ask me about Admissions, Fees, Courses, or your Schedule.",
#         "Greetings! I'm here to assist you with ITU-related queries — Admissions, Fees, Courses, and Schedules.",
#         "Hey! Good to connect with you. Ask me anything about ITU Admissions, Fees, Courses, or your Schedule.",
#     ],
#     "Goodbye": [
#         "Goodbye! It was a pleasure assisting you. Returning to the main menu.",
#         "Take care! Feel free to come back anytime you need help. Returning to the main menu.",
#         "Bye! I hope I was helpful today. Returning to the main menu.",
#         "See you soon! Don't hesitate to reach out again. Returning to the main menu.",
#     ],
#     "Admission": [
#         "Regarding {kw}: ITU Admissions 2026 are merit-based. Apply via the online portal at itu.edu.pk.",
#         "About {kw}: Admissions are conducted through an NTS entry test. Check the portal for the latest deadlines and merit criteria.",
#         "For {kw} queries: ITU uses a competitive merit system. The online application form opens each spring — watch the official website.",
#         "On {kw}: Eligibility requires FSc Pre-Engineering or equivalent with at least 60% marks. Visit itu.edu.pk for the full criteria.",
#     ],
#     "Fee": [
#         "Regarding {kw}: The semester fee is approximately 150,000 PKR. Fee challans are generated on the LMS portal.",
#         "About {kw}: ITU charges around 150,000 PKR per semester. Merit and need-based scholarships are available for eligible students.",
#         "For {kw} information: The current semester fee is 150,000 PKR, due within the first two weeks. Contact the finance office for payment plans.",
#         "On {kw}: Tuition stands at 150,000 PKR per semester. Students with financial need can apply for the ITU scholarship programme.",
#     ],
#     "Courses": [
#         "Regarding {kw}: ITU offers BSSE, BSCE, and BSCS — all four-year undergraduate programmes under SEECS.",
#         "About {kw}: ITU's degree options include Software Engineering (BSSE), Computer Engineering (BSCE), and Computer Science (BSCS).",
#         "For {kw} details: All three programmes span eight semesters. The curriculum covers core CS theory, engineering fundamentals, and a final-year project.",
#         "On {kw}: BSCS focuses on theory and algorithms; BSSE on software development; BSCE on hardware-software integration. Choose based on your interests.",
#     ],
#     "Schedule": [
#         "Regarding {kw}: Timetables are posted on the student portal. Regular classes run 9:00 AM to 5:00 PM, Monday to Friday.",
#         "About {kw}: Log in to the ITU LMS to view your personalised timetable. Exam schedules are published two weeks before the exam period.",
#         "For {kw} information: The academic calendar is available on itu.edu.pk. Office hours are typically 8:00 AM to 5:00 PM.",
#         "On {kw}: Class timings vary by section — check the portal after registration. Mid-terms and finals dates are announced on the LMS.",
#     ],
#     "Unknown": [
#         "I can only assist with Admissions, Fees, Courses, and Schedules. Please try rephrasing your question.",
#         "I'm not sure I understood that. I'm best equipped to help with Admissions, Fees, Courses, or Schedule queries.",
#         "That's outside my area of expertise. Try asking about Admissions, Fees, available Courses, or your Schedule.",
#         "I didn't catch that. Could you ask about ITU Admissions, Fee details, Courses, or your timetable?",
#     ],
#     "Spam": [
#         "This message has been flagged as spam and cannot be processed.",
#         "I detected potentially harmful content. Please ask about academic topics only.",
#         "Spam detected. I can only assist with ITU-related academic queries.",
#     ],
# }

# ================================================================
# ── NLP MODULES
# ================================================================

# def check_spam(text: str) -> bool:
#     return any(kw in text.lower() for kw in SPAM_KEYWORDS)


# def identify_intent(text: str) -> tuple[str, str | None]:
#     t = text.lower()
#     for intent, kws in INTENT_MAP.items():
#         for kw in kws:
#             if kw in t:
#                 return intent, kw
#     return "Unknown", None

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
    # score = sia.polarity_scores(text)['compound']
    
    score = random.uniform(-1, 1)  

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

def classify_intent(text):
    """Return best-matching intent using keyword scoring; 'GENERAL' if no match."""
    t = text.lower()
    scores = {}

    for intent, keywords in INTENT_MAP.items():
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

# {
# 'ACADEMIC': ['assignment','deadline','exam','grade','fail','pass',
# 'lecture','study','professor','submit'],
# 'WELLBEING': ['stress','anxious','depressed','lonely','overwhelmed',
# 'panic','cry','hopeless','afraid'],
# 'FINANCIAL': ['fees','scholarship','loan','afford','money','rent',
# 'bursary','payment','debt'],
# 'TECHNICAL': ['portal','login','password','system','error','access',
# 'email','VPN','reset'],
# 'SOCIAL': ['friends','roommate','belong','isolated','group',
# 'relationship','community'],
# 'ADMIN': ['enrolment','certificate','transcript','registration',
# 'form','office'],
# }
def generate_response(text, intent, emotion):
    """Return a response string based on intent + emotion combination."""
    label = emotion if isinstance(emotion, str) else emotion.get('label', 'NEUTRAL')

    # COMPLAINT
    if intent == 'ACADEMIC' and label in ['FRUSTRATED', 'NEGATIVE']:
        return "I'm sorry to hear you're having academic difficulties. Can you tell me more about what's going on with your assignments or lectures?"
    if intent == 'WELLBEING' and label in ['FRUSTRATED', 'NEGATIVE']:
        return "It sounds like you're going through a tough time. I'm here to listen if you want to share more about how you're feeling."
    if intent == 'FINANCIAL' and label in ['FRUSTRATED', 'NEGATIVE']:
        return "Financial stress can be really hard. Are there specific fees or payment issues you're concerned about?"
    if intent == 'TECHNICAL' and label in ['FRUSTRATED', 'NEGATIVE']:
        return "Technical issues can be so frustrating. Can you describe the problem you're facing with the portal or system access?"
    if intent == 'SOCIAL' and label in ['FRUSTRATED', 'NEGATIVE']:
        return "Social challenges can be really tough. Do you want to talk more about how you're feeling with friends or roommates?"
    if intent == 'ADMIN' and label in ['FRUSTRATED', 'NEGATIVE']:
        return "Administrative tasks can be overwhelming. Is there a specific form or registration issue you're struggling with?"
    return 'Thank you for reaching out. How can I assist you today?'

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

    get_health_report(history, emotion_log, intent_log)


# ================================================================
# Stage 5 — Session Health Report
# ================================================================

def get_health_report(history, emotion_log, intent_log):

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
# ── THREAD-SAFE SIGNALS
# ================================================================

class Signals(QObject):
    status_update   = pyqtSignal(str, str)   # (text, colour) — from any thread
    speech_started  = pyqtSignal()
    speech_finished = pyqtSignal(str)
    stt_result      = pyqtSignal(str)         # recognized text → input bar
    stt_error       = pyqtSignal(str)
    listen_finished = pyqtSignal()

# ================================================================
# ── STYLE FACTORIES
# ================================================================

def _pill(bg: str, fg: str = "white", border: str = None,
          hover: str = None, size: int = 15) -> str:
    """Primary action button — the pill (border-radius 980px ≈ full capsule)."""
    bd = f"border: 1.5px solid {border};" if border else "border: none;"
    hv = f"QPushButton:hover {{ background-color: {hover}; }}" if hover else ""
    return f"""
        QPushButton {{
            background-color: {bg}; color: {fg};
            font-size: {size}px; border-radius: 980px;
            padding: 9px 22px; {bd}
        }}
        QPushButton:disabled {{
            background-color: {C['divider']}; color: {C['ink_muted']}; border: none;
        }}
        {hv}
    """

def _util(bg: str, fg: str = "white", hover: str = None, size: int = 13,
          radius: int = 8) -> str:
    """Compact utility button — rounded rect, not pill."""
    hv = f"QPushButton:hover {{ background-color: {hover}; }}" if hover else ""
    return f"""
        QPushButton {{
            background-color: {bg}; color: {fg};
            font-size: {size}px; border-radius: {radius}px;
            padding: 7px 14px; border: none;
        }}
        QPushButton:disabled {{ background-color: {C['divider']}; color: {C['ink_muted']}; }}
        {hv}
    """

def _entry_css() -> str:
    """Pill-shaped text entry with hairline border; blue focus ring; parchment when read-only."""
    return f"""
        QLineEdit {{
            background-color: {C['canvas']}; color: {C['ink']};
            border: 1.5px solid {C['hairline']}; border-radius: 980px;
            padding: 9px 20px; font-size: 15px;
            selection-background-color: {C['primary']};
        }}
        QLineEdit:focus {{
            border: 1.5px solid {C['primary']};
        }}
        QLineEdit:read-only {{
            background-color: {C['canvas_parchment']}; color: {C['ink_muted']};
            border: 1.5px solid {C['divider']};
        }}
    """

# ================================================================
# ── MAIN WINDOW
# ================================================================

class ITU_Chatbot_GUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP["title"])
        self.resize(*APP["window_size"])
        self.setMinimumSize(700, 560)
        self.setStyleSheet(f"background-color: {C['canvas_parchment']};")

        self.is_listening = False
        self.is_speaking  = False
        self.current_mode = "Review"
        self._in_confirm  = False

        self.sig = Signals()
        self._wire_signals()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self._build_menu()
        self._build_chat()
        self.stack.setCurrentWidget(self.menu_widget)

    # ── Signal wiring ─────────────────────────────────────────────

    def _wire_signals(self):
        s = self.sig
        s.status_update.connect(self._set_status)
        s.speech_started.connect(lambda: self._lock_ui(True))
        s.speech_finished.connect(self._on_speech_done)
        s.stt_result.connect(self._on_stt_result)
        s.stt_error.connect(self._on_stt_error)
        s.listen_finished.connect(self._on_listen_done)

    # ── Menu screen ───────────────────────────────────────────────

    def _build_menu(self):
        self.menu_widget = QWidget()
        self.menu_widget.setStyleSheet(f"background-color: {C['canvas_parchment']};")

        lay = QVBoxLayout(self.menu_widget)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(0)

        # Wordmark
        title = QLabel("NEXUS")
        title.setFont(_font("display"))
        title.setStyleSheet(f"color: {C['ink']};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)

        lay.addSpacing(6)

        sub = QLabel("Multi-Modal Chatbot System")
        sub.setFont(_font("callout"))
        sub.setStyleSheet(f"color: {C['ink_muted']};")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(sub)

        # Hairline divider — uses these to let content breathe
        lay.addSpacing(32)
        rule = QFrame()
        rule.setFixedHeight(1)
        rule.setFixedWidth(300)
        rule.setStyleSheet(f"background-color: {C['hairline']};")
        lay.addWidget(rule, alignment=Qt.AlignmentFlag.AlignCenter)
        lay.addSpacing(28)

        lbl = QLabel("Select a mode to begin")
        lbl.setFont(_font("footnote"))
        lbl.setStyleSheet(f"color: {C['ink_muted']};")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl)
        lay.addSpacing(16)

        for label, mode in MODES:
            b = QPushButton(label)
            b.setFixedWidth(280)
            b.setMinimumHeight(44)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFont(_font("body"))
            b.setStyleSheet(_pill(C["primary"], hover=C["primary_hover"]))
            b.clicked.connect(lambda _, m=mode: self._start(m))
            lay.addWidget(b, alignment=Qt.AlignmentFlag.AlignCenter)
            lay.addSpacing(10)

        lay.addSpacing(18)

        # Quit — ghost pill in system red; secondary grammar, never the hero action
        quit_b = QPushButton("Quit Application")
        quit_b.setFixedWidth(280)
        quit_b.setMinimumHeight(40)
        quit_b.setCursor(Qt.CursorShape.PointingHandCursor)
        quit_b.setFont(_font("callout"))
        quit_b.setStyleSheet(_pill(
            "transparent", fg=C["system_red"],
            border=C["system_red"], size=13,
        ))
        quit_b.clicked.connect(QApplication.instance().quit)
        lay.addWidget(quit_b, alignment=Qt.AlignmentFlag.AlignCenter)

        self.stack.addWidget(self.menu_widget)

    # ── Chat screen ───────────────────────────────────────────────

    def _build_chat(self):
        self.chat_widget = QWidget()
        self.chat_widget.setStyleSheet("background: transparent;")
        root = QVBoxLayout(self.chat_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Global nav bar — near-black surface, like persistent top strip
        nav = QFrame()
        nav.setStyleSheet(f"background-color: {C['surface_dark']};")
        nav.setFixedHeight(52)
        nl = QHBoxLayout(nav)
        nl.setContentsMargins(20, 0, 20, 0)

        self.back_btn = QPushButton("⬅  Back")
        self.back_btn.setFont(_font("callout"))
        # Transparent button on dark nav — text link style with primary-on-dark colour
        self.back_btn.setStyleSheet(_util(
            "transparent", fg=C["primary_on_dark"],
            hover=C["surface_dark_alt"], size=13,
        ))
        self.back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back_btn.clicked.connect(self._to_menu)
        nl.addWidget(self.back_btn)
        nl.addStretch()

        self.mode_label = QLabel()
        self.mode_label.setFont(_font("footnote"))
        self.mode_label.setStyleSheet(f"color: {C['on_dark_muted']};")
        nl.addWidget(self.mode_label)
        root.addWidget(nav)

        # Chat display — pure white canvas; typography does the heavy lifting
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(_font("body"))
        self.chat_display.setStyleSheet(
            f"QTextEdit {{ background-color: {C['canvas']}; color: {C['ink']};"
            " border: none; padding: 20px 24px; }"
        )
        root.addWidget(self.chat_display, stretch=1)

        # Status strip — italic footnote, structural hairline above
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setFont(_font("footnote", italic=True))
        self.status_lbl.setStyleSheet(
            f"color: {C['ink_muted']}; padding: 5px 24px;"
            f"background-color: {C['canvas']}; border-top: 1px solid {C['divider']};"
        )
        root.addWidget(self.status_lbl)

        # Input panel — parchment surface, separated by hairline divider
        panel = QFrame()
        panel.setStyleSheet(
            f"background-color: {C['canvas_parchment']};"
            f"border-top: 1px solid {C['divider']};"
        )
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(20, 12, 20, 16)
        pl.setSpacing(10)

        # Row 1 — pill entry + Send pill (hidden in Voice mode)
        tr = QWidget()
        tr.setStyleSheet("background: transparent;")
        tl = QHBoxLayout(tr)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(10)

        self.user_entry = QLineEdit()
        self.user_entry.setFont(_font("body"))
        self.user_entry.setStyleSheet(_entry_css())
        self.user_entry.setPlaceholderText("Message ITU Assistant…")
        self.user_entry.setMinimumHeight(42)
        self.user_entry.returnPressed.connect(self._send)
        tl.addWidget(self.user_entry)

        self.send_btn = QPushButton("Send")
        self.send_btn.setFont(_font("body"))
        self.send_btn.setStyleSheet(_pill(C["primary"], hover=C["primary_hover"]))
        self.send_btn.setFixedWidth(88)
        self.send_btn.setMinimumHeight(42)
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.clicked.connect(self._send)
        tl.addWidget(self.send_btn)
        pl.addWidget(tr)

        # Row 2 — voice / STT confirmation row
        vr = QWidget()
        vr.setStyleSheet("background: transparent;")
        vl = QHBoxLayout(vr)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(10)

        # Dark utility button — secondary grammar (not a blue pill)
        self.voice_btn = QPushButton("🎤  Voice Input")
        self.voice_btn.setFont(_font("callout"))
        self.voice_btn.setStyleSheet(_util(
            C["surface_dark"], fg=C["on_dark"],
            hover=C["surface_dark_alt"],
        ))
        self.voice_btn.setMinimumHeight(40)
        self.voice_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.voice_btn.clicked.connect(self._toggle_listen)
        vl.addWidget(self.voice_btn)
        vl.addStretch()

        # Ghost pill — pearl/outline style (secondary, not hero)
        self.retry_btn = QPushButton("↩  Retry")
        self.retry_btn.setFont(_font("callout"))
        self.retry_btn.setStyleSheet(_pill(
            "transparent", fg=C["ink_secondary"],
            border=C["hairline"], size=13,
        ))
        self.retry_btn.setMinimumHeight(40)
        self.retry_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.retry_btn.clicked.connect(self._retry)
        self.retry_btn.setVisible(False)
        vl.addWidget(self.retry_btn)

        # Primary blue pill — the single "yes, proceed" action
        self.confirm_btn = QPushButton("✓  Confirm")
        self.confirm_btn.setFont(_font("callout"))
        self.confirm_btn.setStyleSheet(_pill(C["primary"], hover=C["primary_hover"], size=13))
        self.confirm_btn.setMinimumHeight(40)
        self.confirm_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.confirm_btn.clicked.connect(self._send)
        self.confirm_btn.setVisible(False)
        vl.addWidget(self.confirm_btn)

        pl.addWidget(vr)
        root.addWidget(panel)
        self.stack.addWidget(self.chat_widget)

    # ── Navigation ────────────────────────────────────────────────

    def _start(self, mode: str):
        self.current_mode = mode
        self.mode_label.setText(mode)
        self.stack.setCurrentWidget(self.chat_widget)
        self._apply_mode_ui()
        self._run("hello")

    def _to_menu(self):
        self.is_listening = False
        self.is_speaking  = False
        self._in_confirm  = False
        self.chat_display.clear()
        self.stack.setCurrentWidget(self.menu_widget)

    # ── Mode UI ───────────────────────────────────────────────────

    def _apply_mode_ui(self):
        """Reset input panel to the default layout for the active mode."""
        m = self.current_mode
        self.user_entry.setVisible(False if m == "Review" else True)
        self.user_entry.setReadOnly(False)
        self.user_entry.setStyleSheet(_entry_css())
        self.user_entry.clear()
        self.send_btn.setVisible(False if m == "Review" else True)
        self.voice_btn.setVisible(False if m == "Review" else True)
        self.voice_btn.setText("🎤  Voice Input")
        self.voice_btn.setStyleSheet(_util(
            C["surface_dark"], fg=C["on_dark"], hover=C["surface_dark_alt"],
        ))
        self.retry_btn.setVisible(False if m == "Review" else True)
        self.confirm_btn.setVisible(False if m == "Review" else True)
        self._in_confirm = False

    def _enter_confirm_state(self, text: str):
        """
        Populate the input bar with the STT result and surface confirmation controls.
        Voice mode  → entry is read-only (parchment tint); Retry and Confirm only.
        Hybrid mode → entry is editable; user may correct before confirming.
        """
        m = self.current_mode
        self._in_confirm = True

        self.user_entry.setVisible(True)
        self.user_entry.setText(text)
        self.user_entry.setReadOnly(False)
        self.user_entry.setStyleSheet(_entry_css())   # :read-only rule applies automatically

        hint = (
            "Voice captured — edit if needed, then confirm"
        )
        self.send_btn.setVisible(False)
        self.voice_btn.setVisible(False)
        self.retry_btn.setVisible(True)
        self.confirm_btn.setVisible(True)
        self._set_status(hint, C["primary"])

    # ── Voice controls ────────────────────────────────────────────

    def _toggle_listen(self):
        if self.is_speaking:
            return
        if not self.is_listening:
            self.is_listening = True
            self.voice_btn.setText("■  Stop")
            self.voice_btn.setStyleSheet(_util(C["system_red"], fg=C["on_dark"], hover="#e0352b"))
            self._set_status("Listening…", C["system_orange"])
            threading.Thread(target=self._listen_thread, daemon=True).start()
        else:
            self.is_listening = False

    def _retry(self):
        self._apply_mode_ui()
        self._toggle_listen()

    def _send(self):
        """Shared handler for Send button, Confirm button, and Return key."""
        if self.is_speaking:
            return
        text = self.user_entry.text().strip()
        if not text:
            return
        self._apply_mode_ui()
        self._run(text)

    # ── Background threads ────────────────────────────────────────

    def _listen_thread(self):
        r = sr.Recognizer()
        try:
            with sr.Microphone() as src:
                r.adjust_for_ambient_noise(src, duration=STT["ambient_calibration"])
                audio = r.listen(
                    src,
                    timeout=STT["timeout"],
                    phrase_time_limit=STT["phrase_time_limit"],
                )
            if not self.is_listening:
                return
            self.sig.stt_result.emit(r.recognize_google(audio))
        except sr.WaitTimeoutError:
            self.sig.stt_error.emit("No speech detected — please try again.")
        except sr.UnknownValueError:
            self.sig.stt_error.emit("Could not understand audio — please try again.")
        except Exception as exc:
            self.sig.stt_error.emit(f"Microphone error: {exc}")
        finally:
            self.is_listening = False
            self.sig.listen_finished.emit()

    def _speak_thread(self, text: str, intent: str):
        self.is_speaking = True
        self.sig.speech_started.emit()
        self.sig.status_update.emit("Speaking…", C["system_yellow"])
        engine = pyttsx3.init()
        engine.setProperty("rate", TTS["rate"])
        engine.say(text)
        engine.runAndWait()
        self.is_speaking = False
        self.sig.speech_finished.emit(intent)

    # ── Signal handlers (always on main thread via Qt queued connection) ──

    def _on_stt_result(self, text: str):
        self._enter_confirm_state(text)

    def _on_stt_error(self, msg: str):
        self._set_status(msg, C["system_red"])

    def _on_listen_done(self):
        if not self._in_confirm:
            self.voice_btn.setText("🎤  Voice Input")
            self.voice_btn.setStyleSheet(
                _util(C["surface_dark"], fg=C["on_dark"], hover=C["surface_dark_alt"])
            )

    def _on_speech_done(self, intent: str):
        if intent == "FAREWELL":
            self._to_menu()
            return
        self._lock_ui(False)
        self._set_status("Ready", C["ink_muted"])

    # ── Pipeline (always called on main thread) ───────────────────

    def _run(self, query: str):
        if query.lower() != "hello":
            self._append_message("You", query)

        # if check_spam(query):
        #     resp = build_response("Spam", None)
        #     self._append_message("System", resp)
        #     threading.Thread(target=self._speak_thread, args=(resp, "Spam"), daemon=True).start()
        #     return

        intent= classify_intent(query)
        emotion = classify_emotion(query)
        
        print(intent)
        
        # self._set_status(f"Intent: {intent}  ·  matched '{kw}'", C["ink_muted"])
        
        if(self.current_mode == "Review"):
            self._set_status(f"Intent: Review", C["ink_muted"])
        
        
        resp = generate_response(query, intent, emotion['label'])
        
        
        
        
        self._append_message("Assistant", resp)
        
        
        # CONVO_LOG = [
        #     'Hi, I need some help with my account.',
        #     'I have been trying to log in for two days but it keeps failing.',
        #     'This is really frustrating, I cannot believe this is still broken.',
        #     'Can you at least tell me when it will be fixed?',
        #     'Also I was charged twice last month — I want a refund.',
        #     'Actually you know what, forget it. I am going to cancel.',
        #     'Wait — actually I found the issue, it was my browser. Never mind!',
        #     'Thanks for your patience. Bye.',
        # ]
        
        if(self.current_mode != "Review"):
            threading.Thread(target=self._speak_thread, args=(resp, intent), daemon=True).start()
            
        if(self.current_mode == "Live"):
            self._lock_ui(False)
            self._set_status(f"Intent: {intent} ·  Eatched '{emotion}' ", C["ink_muted"])
        
        if(self.current_mode == "Review"):
            intent = "Review"
            for i, msg in enumerate(CONVO_LOG):
                # emotion  = classify_emotion(msg)
                # intent   = classify_intent(msg)
                # shift    = detect_mood_shift(prev_emotion_label, emotion['label']) \
                #     if prev_emotion_label is not None else 'stable'
                print(intent)
                response = generate_response(msg, intent, emotion['label'])
                self._append_message("You", msg)
                self._append_message("Assistant", response)
                
                # self._lock_ui(False)
                # self._set_status("Ready", C["ink_muted"])
                
                
            

    # ── UI helpers ────────────────────────────────────────────────

    def _append_message(self, sender: str, msg: str):
        colour = {
            "You":           C["primary"],
            "Assistant": C["ink"],
            "System":        C["system_red"],
        }.get(sender, C["ink_muted"])

        # Uppercase micro-label above the message body
        html = (
            f'<span style="font-size: 10px; font-weight: bold; color: {colour};">'
            f'{sender.upper()}</span><br/>'
            f'<span style="font-size: 15px; color: {C["ink"]};">{msg}</span>'
        )
        self.chat_display.append(html)
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)

    def _set_status(self, text: str, color: str):
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(
            f"color: {color}; padding: 5px 24px;"
            f"background-color: {C['canvas']}; border-top: 1px solid {C['divider']};"
        )

    def _lock_ui(self, locked: bool):
        for w in (self.user_entry, self.send_btn, self.voice_btn,
                  self.retry_btn, self.confirm_btn, self.back_btn):
            w.setEnabled(not locked)

# ================================================================
# ── ENTRY POINT
# ================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ITU_Chatbot_GUI()
    win.show()
    sys.exit(app.exec())

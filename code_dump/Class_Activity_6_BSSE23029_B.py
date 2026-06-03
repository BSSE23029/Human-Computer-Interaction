import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

nltk.download('punkt')
nltk.download('stopwords')

# Text Preprocessing

def preprocess(text):
    """Convert to lowercase and remove extra spaces."""
    text = text.lower()           # lowercase
    text = text.strip()           # remove leading/trailing spaces
    text = ' '.join(text.split()) # collapse multiple spaces
    return text

# Quick test
print(preprocess("  Hello  WORLD  "))  # → "hello world"

# Food Delivery Chatbot Intents

intents = {
    "greeting": {
        "patterns": [
            "hello", "hi", "hey", "good morning",
            "good evening", "howdy", "what's up", "greetings"
        ],
        "responses": [
            "Hello! Welcome to FoodExpress. How can I help you today?",
            "Hi there! Ready to order some delicious food?",
            "Hey! What can I get for you today?"
        ]
    },
    "order_food": {
        "patterns": [
            "i want to order", "can i order", "place an order",
            "i'd like to buy", "order food", "i want food",
            "get me some food", "i want to place order"
        ],
        "responses": [
            "Great! What would you like to order? Check our menu first!",
            "Sure! Please tell me what you'd like from our menu.",
            "I'd be happy to take your order. What are you craving?"
        ]
    },
    "menu": {
        "patterns": [
            "show menu", "what do you have", "what's available",
            "see the menu", "food options", "what can i order",
            "list of food", "what food do you serve"
        ],
        "responses": [
            "Our menu includes: Burgers, Pizza, Pasta, Salads, Wraps, and Desserts!",
            "We offer Pizza, Burgers, Pasta, Salads, Sushi, and much more!",
            "Today's menu: Pizza, Burgers, Sandwiches, Salads, and Desserts."
        ]
    },
    "delivery_time": {
        "patterns": [
            "how long will it take", "delivery time", "when will my order arrive",
            "estimated time", "how soon", "when will food come",
            "time for delivery", "how fast is delivery"
        ],
        "responses": [
            "Delivery typically takes 30–45 minutes depending on your location.",
            "Estimated delivery time is 30–60 minutes. We'll notify you!",
            "Your order should arrive within 45 minutes. Thank you for waiting!"
        ]
    },
    "payment": {
        "patterns": [
            "how can i pay", "payment methods", "do you accept cash",
            "can i pay online", "credit card", "payment options",
            "how to pay", "what payment do you accept"
        ],
        "responses": [
            "We accept Cash, Credit/Debit cards, and online payments via JazzCash & EasyPaisa!",
            "Payment options: Cash on delivery, Visa, Mastercard, and mobile wallets.",
            "You can pay via cash, card, or any mobile payment app."
        ]
    },
    "contact": {
        "patterns": [
            "contact support", "customer service", "help",
            "talk to someone", "phone number", "how to contact",
            "reach you", "support team"
        ],
        "responses": [
            "Contact us at support@foodexpress.pk or call 0300-1234567.",
            "Our support team is available 24/7 at 0300-1234567.",
            "Reach us via email: help@foodexpress.pk or WhatsApp: 0300-1234567."
        ]
    }
}

# Extract patterns and labels from intents

training_patterns = []
training_intents  = []

for intent_name, intent_data in intents.items():
    for pattern in intent_data["patterns"]:
        training_patterns.append(preprocess(pattern))
        training_intents.append(intent_name)

print(f"Total training samples: {len(training_patterns)}")
print(f"Sample pattern : {training_patterns[0]}")
print(f"Sample label   : {training_intents[0]}")

# TF-IDF Vectorizer (unigrams + bigrams)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # unigrams and bigrams
    analyzer='word',
    min_df=1
)

X_train = vectorizer.fit_transform(training_patterns)

print(f"Feature matrix shape : {X_train.shape}")
print(f"Vocabulary size      : {len(vectorizer.vocabulary_)}")

# Train Multinomial Naive Bayes

classifier = MultinomialNB()
classifier.fit(X_train, training_intents)

print("Model trained successfully!")
print(f"Classes: {classifier.classes_}")

# Predict intent and return confidence score

def predict_intent(user_input):
    """
    Takes raw user input, vectorizes it, predicts intent,
    and returns (intent_label, confidence_score).
    """
    cleaned = preprocess(user_input)
    X       = vectorizer.transform([cleaned])
    intent  = classifier.predict(X)[0]
    probs   = classifier.predict_proba(X)[0]
    confidence = max(probs)
    return intent, confidence

# Quick test
intent, conf = predict_intent("show me the menu")
print(f"Intent: {intent}  |  Confidence: {conf:.2f}")

# Generate a response based on intent and confidence

CONFIDENCE_THRESHOLD = 0.3

def get_response(user_input):
    """
    Predicts intent from user input.
    Returns a random matching response if confidence is high enough,
    otherwise returns a fallback message.
    """
    intent, confidence = predict_intent(user_input)

    if confidence >= CONFIDENCE_THRESHOLD:
        responses = intents[intent]["responses"]
        return random.choice(responses)
    else:
        return ("I'm not sure I understood that. "
                "Try asking about our menu, orders, delivery time, or payment.")

# Quick test
print(get_response("how long does delivery take"))

# Run the chatbot — type "quit" to exit

print("FoodExpress Chatbot is ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() == "quit":
        print("Bot: Thank you for using FoodExpress. Goodbye!")
        break

    response = get_response(user_input)
    print(f"Bot: {response}\n")
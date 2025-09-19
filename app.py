from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
from textblob import TextBlob

app = Flask(__name__)
CORS(app)

# Gemini API config
GEMINI_API_KEY = "49de4768f1msh7a09ab21d046638p1dc42fjsn2c40aec517d1"
GEMINI_API_URL = "https://gemini-pro-ai.p.rapidapi.com"
HEADERS = {
    "Content-Type": "application/json",
    "X-RapidAPI-Key": GEMINI_API_KEY,
    "X-RapidAPI-Host": "gemini-pro-ai.p.rapidapi.com"
}

# Strongly negative words (custom lexicon)
custom_negative = {
    "die": -0.9,
    "suicide": -1.0,
    "kill": -0.9,
    "murder": -0.9,
    "self-harm": -1.0,
    "cut": -0.8,
    "hang": -0.9,
    "worthless": -0.8,
    "failure": -0.7,
    "hopeless": -0.9,
    "pointless": -0.8,
    "useless": -0.8,
    "meaningless": -0.8,
    "depressed": -0.8,
    "anxious": -0.6,
    "lonely": -0.7,
    "tired": -0.6,
    "stressed": -0.6,
    "broken": -0.7,
    "hate": -0.9,
    "anger": -0.8,
    "rage": -0.9,
    "cry": -0.7,
    "sad": -0.8,
    "grief": -0.9,
    "pain": -0.9,
    "suffer": -0.9,
}

def analyze_sentiment(text):
    """Custom sentiment analysis with strong negative overrides."""
    lower_text = text.lower()

    # Check custom strong negatives first
    for word in custom_negative:
        if word in lower_text:
            return -0.9  # force strong negative

    # Otherwise use TextBlob polarity
    return TextBlob(text).sentiment.polarity

def classify_sentiment(polarity):
    """Convert polarity score to label + prefix message."""
    if polarity >= 0.8:
        return "strongly positive", "The user feels very positive. Reply with an inspiring and joyful message. User said: "
    elif polarity > 0.1:
        return "positive", "The user feels positive. Reply with an encouraging and uplifting message. User said: "
    elif polarity <= -0.1:
        return "negative", "The user feels negative. Reply with an empathetic and supportive message. User said: "
    else:  # -0.1 to 0.1
        return "neutral", "The user is in between positive and negative emotion. Help them reflect and choose a better option. User said: "

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Message is empty"}), 400

    # Sentiment analysis
    polarity = analyze_sentiment(user_message)
    sentiment, prefix = classify_sentiment(polarity)

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prefix + user_message}]}
        ]
    }

    try:
        response = requests.post(GEMINI_API_URL, json=payload, headers=HEADERS, timeout=20)
        response.raise_for_status()
        result = response.json()

        print("Gemini raw response:", result)

        ai_message = (
            result.get("candidates", [{}])[0]
                  .get("content", {})
                  .get("parts", [{}])[0]
                  .get("text", "No reply from AI.")
        )

        return jsonify({
            "sentiment": sentiment,
            "reply": ai_message
        })

    except Exception as e:
        print("Error from backend:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("✅ Backend is running on http://127.0.0.1:5000")
    app.run(port=5000, debug=True)

from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
from textblob import TextBlob

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = "49de4768f1msh7a09ab21d046638p1dc42fjsn2c40aec517d1"
GEMINI_API_URL = "https://gemini-pro-ai.p.rapidapi.com"
HEADERS = {
    "Content-Type": "application/json",
    "X-RapidAPI-Key": GEMINI_API_KEY,
    "X-RapidAPI-Host": "gemini-pro-ai.p.rapidapi.com"
}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Message is empty"}), 400

    analysis = TextBlob(user_message)
    polarity = analysis.sentiment.polarity

    if polarity > 0.1:
        sentiment = "positive"
        prefix = "The user feels positive. Reply with an encouraging and uplifting message. User said: "
    elif polarity < -0.1:
        sentiment = "negative"
        prefix = "The user feels negative. Reply with an empathetic and supportive message. User said: "
    else:
        sentiment = "neutral"
        prefix = "The user feels neutral. Reply with a friendly and engaging message. User said: "

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
    app.run(port=5000, debug=True)

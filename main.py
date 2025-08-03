from flask import Flask, request
import openai
from twilio.twiml.messaging_response import MessagingResponse
import os
from retriever import get_best_match

app = Flask(__name__)
openai.api_key = os.environ['OPENAI_API_KEY']

print("OPENAI KEY:", openai.api_key)  # 👈 just for debug


@app.route("/webhook", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.values.get("Body", "").strip()
    print("Received:", incoming_msg)

    # Use vector retriever to get context
    try:
        matches = get_best_match(incoming_msg, top_k=1)
        context = matches[0][
            1] if matches else "Dr. Garg's Dental Care is a dental clinic in Ludhiana."
    except Exception as e:
        print("Retriever error:", e)
        context = "Dr. Garg's Dental Care is a dental clinic in Ludhiana."

    # System prompt with context included
    system_prompt = f"""
You are the assistant for Dr Garg's Dental Care, Model Town, Ludhiana.

Context to help you:
{context}

Clinic hours: Mon–Sat, 10 AM – 7 PM. Closed Sundays.
Services: Root canal, aligners, scaling, implants, smile design, kids dentistry, and cosmetic dental procedures.
Fees: ₹200 consultation. Teeth cleaning starts ₹1,200.
Booking: Patients can walk in or call +91 99158 35290.

Speak in a warm, reassuring tone. Use emojis occasionally. Always answer as a clinic assistant.
"""

    try:
        client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

        completion = client.chat.completions.create(model="gpt-3.5-turbo",
                                                    messages=[{
                                                        "role":
                                                        "system",
                                                        "content":
                                                        system_prompt
                                                    }, {
                                                        "role":
                                                        "user",
                                                        "content":
                                                        incoming_msg
                                                    }])

        reply = completion.choices[0].message.content.strip()

    except Exception as e:
        print("OpenAI error:", e)
        reply = "Sorry, there was a problem fetching your answer."

    response = MessagingResponse()
    response.message(reply)
    return str(response)


# 🔁 Fix for Replit: Don't use app.run() directly
if __name__ == "__main__":
    from flask import send_from_directory

    @app.route("/")
    def root():
        return "Dr Garg's chatbot is live."

    app.run(host="0.0.0.0", port=3000)

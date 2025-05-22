
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = Flask(__name__)

# Load tokenizer and model
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Label mapping
labels = ['negative', 'neutral', 'positive']
emojis = {'negative': '😞', 'neutral': '😐', 'positive': '😊'}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probabilities = None
    text_input = ""
    if request.method == "POST":
        text_input = request.form["review"]

        # Tokenize and run through model
        inputs = tokenizer(text_input, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1).squeeze().tolist()

        # Find top class
        max_idx = int(torch.argmax(logits))
        label = labels[max_idx]
        prediction = f"{label.title()} {emojis[label]}"

        # Format probabilities
        probabilities = {
            "positive": round(probs[2] * 100, 2),
            "neutral": round(probs[1] * 100, 2),
            "negative": round(probs[0] * 100, 2)
        }

    return render_template("index.html", prediction=prediction, text_input=text_input, probabilities=probabilities)

if __name__ == "__main__":
    app.run(debug=True, port=8000)

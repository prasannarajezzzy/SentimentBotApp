from flask import Flask, render_template, request, jsonify
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import os

app = Flask(__name__)


# Load the model
class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)


# Initialize model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentClassifier().to(device)
model_path = "model/sentiment_analysis_model.pth"

if os.path.exists(model_path):
    model.load_state_dict(
        torch.load(
            model_path, map_location=device if not torch.cuda.is_available() else None
        )
    )
else:
    print(f"Error: Model file '{model_path}' not found.")

model.eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Function to predict sentiment
def predict_sentiment(text, model, tokenizer):
    with torch.no_grad():
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        output = model(input_ids, attention_mask)
        prediction = torch.sigmoid(output).item()
        return "positive" if prediction > 0.5 else "negative"


# Route to handle home page
@app.route("/")
def home():
    return render_template("index.html")


# Route to handle form submission and sentiment analysis
@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form["text"]
    print(text)
    sentiment = predict_sentiment(text, model, tokenizer)
    print(sentiment)
    sentiment_class = "success" if sentiment == "positive" else "danger"
    # Generate HTML for result
    result_html = f"""
    <div class="card">
        <div class="card-body">
            <p class="card-text"><strong>Input text:</strong> {text}</p>
            <p class="card-text"><strong>Sentiment:</strong> <span class="badge badge-{sentiment_class}">{sentiment}</span></p>
        </div>
    </div>
    """

    return result_html


if __name__ == "__main__":
    app.run(debug=True)

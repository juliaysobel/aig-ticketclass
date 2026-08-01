"""
IT Support Ticket Classifier — Inference App
Course: AIG230NAA — NLP Final Project

Loads both the TF-IDF + Logistic Regression baseline and the fine-tuned
DistilBERT classifier, and lets you compare their predictions on any ticket
text you type in. Satisfies the assignment's inference-app requirement:
loads a saved model, predicts on new text without retraining, shows
confidence scores, in real time.

Run with:  python app.py
Then open the local URL it prints (usually http://127.0.0.1:7860).
"""

import json
import os

import joblib
import torch
import torch.nn as nn
import gradio as gr
from transformers import DistilBertModel, DistilBertTokenizerFast

# ------------------------------------------------------------------
# Paths — assumes this file sits at the repo root, next to models/
# ------------------------------------------------------------------
BASELINE_DIR = "models/baseline"
DISTILBERT_DIR = "models/distilbert"

# ------------------------------------------------------------------
# Illustrative routing lookup (category -> team). This is deliberately a
# plain dictionary, not a trained model — routing is a business-rule
# mapping, not something that needs ML. Update the team names once you
# know your actual co-op's real team structure.
# ------------------------------------------------------------------
ROUTING_MAP = {
    "Account": "Identity & Access Management",
    "Communication": "Collaboration Tools Team",
    "Hardware": "Desktop Support",
    "Network": "Network Operations",
    "Other": "General IT Support (Triage)",
    "RemoteWork": "Remote Access / VPN Team",
    "Security": "Security Team",
    "Software": "Application Support",
}


# ------------------------------------------------------------------
# Custom model class — must match Notebook 3's architecture exactly,
# since we're loading a state_dict that was saved from this exact class.
# ------------------------------------------------------------------
class TicketClassifier(nn.Module):
    def __init__(self, n_classes, dropout=0.3, encoder_name='distilbert-base-uncased'):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        x = self.relu(self.dense(x))
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


# ------------------------------------------------------------------
# Load baseline (TF-IDF + Logistic Regression) — fast, local files only
# ------------------------------------------------------------------
print("Loading baseline model...")
baseline_vectorizer = joblib.load(f"{BASELINE_DIR}/tfidf_vectorizer.joblib")
baseline_model = joblib.load(f"{BASELINE_DIR}/logreg_model.joblib")
print("Baseline loaded.")

# ------------------------------------------------------------------
# Load fine-tuned DistilBERT
# ------------------------------------------------------------------
print("Loading DistilBERT model (this can take a moment on first run)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(f"{DISTILBERT_DIR}/classifier_config.json") as f:
    db_config = json.load(f)

db_tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_DIR)
db_model = TicketClassifier(
    n_classes=db_config['n_classes'],
    dropout=db_config.get('dropout', 0.3),
)
db_model.load_state_dict(
    torch.load(f"{DISTILBERT_DIR}/pytorch_model.bin", map_location=device)
)
db_model.to(device)
db_model.eval()
print(f"DistilBERT model loaded on {device}.")


# ------------------------------------------------------------------
# Prediction functions
# ------------------------------------------------------------------
def predict_baseline(text):
    X = baseline_vectorizer.transform([text])
    probs = baseline_model.predict_proba(X)[0]
    return {str(cls): float(p) for cls, p in zip(baseline_model.classes_, probs)}


def predict_distilbert(text):
    max_length = db_config['max_length']
    encoding = db_tokenizer(
        text, max_length=max_length, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = db_model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    class_labels = db_config['class_labels']
    return {str(cls): float(p) for cls, p in zip(class_labels, probs)}


def classify_ticket(text, model_choice):
    if not text or not text.strip():
        return {}, "Enter some ticket text first."

    if model_choice == "TF-IDF + Logistic Regression (baseline)":
        probs = predict_baseline(text)
    else:
        probs = predict_distilbert(text)

    top_category = max(probs, key=probs.get)
    team = ROUTING_MAP.get(top_category, "General IT Support (Triage)")
    routing_note = f"**Predicted category:** {top_category}  \n**\u2192 Routed to:** {team}"

    return probs, routing_note


# ------------------------------------------------------------------
# Gradio interface
# ------------------------------------------------------------------
EXAMPLE_TICKETS = [
    ["My laptop screen keeps flickering and won't stay on.", "DistilBERT (fine-tuned)"],
    ["I can't connect to the company VPN from home.", "DistilBERT (fine-tuned)"],
    ["Requesting access to the shared marketing drive.", "DistilBERT (fine-tuned)"],
    ["Email keeps bouncing back when I send to external clients.", "DistilBERT (fine-tuned)"],
]

with gr.Blocks(title="IT Support Ticket Classifier") as demo:
    gr.Markdown("# IT Support Ticket Classifier")
    gr.Markdown(
        "Enter internal IT support ticket text to see the predicted category, "
        "confidence scores, and a suggested routing team. Compare the fine-tuned "
        "DistilBERT model against the TF-IDF + Logistic Regression baseline."
    )

    with gr.Row():
        with gr.Column():
            ticket_input = gr.Textbox(
                label="Ticket text",
                placeholder="e.g. My laptop won't connect to the office wifi...",
                lines=5,
            )
            model_choice = gr.Radio(
                choices=[
                    "DistilBERT (fine-tuned)",
                    "TF-IDF + Logistic Regression (baseline)",
                ],
                value="DistilBERT (fine-tuned)",
                label="Model",
            )
            classify_btn = gr.Button("Classify", variant="primary")

        with gr.Column():
            output_label = gr.Label(label="Category probabilities", num_top_classes=8)
            routing_output = gr.Markdown()

    classify_btn.click(
        fn=classify_ticket,
        inputs=[ticket_input, model_choice],
        outputs=[output_label, routing_output],
    )

    gr.Examples(
        examples=EXAMPLE_TICKETS,
        inputs=[ticket_input, model_choice],
    )

if __name__ == "__main__":
    demo.launch()

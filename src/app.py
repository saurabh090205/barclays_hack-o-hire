from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import numpy as np

app = FastAPI(title="Barclays Email Fraud Detection Engine")

MODEL_PATH = "models/best_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


class EmailRequest(BaseModel):
    email_text: str


def risk_band(score):
    if score >= 85:
        return "CRITICAL"
    elif score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    else:
        return "LOW"


def recommended_action(band):
    actions = {
        "CRITICAL": "Immediately quarantine email and alert SOC.",
        "HIGH": "Flag email and require user verification.",
        "MEDIUM": "Mark as suspicious and monitor activity.",
        "LOW": "Allow delivery."
    }
    return actions[band]


@app.post("/score-email")
def score_email(request: EmailRequest):
    inputs = tokenizer(
        request.email_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).numpy()[0]

    fraud_prob = float(probabilities[1])
    risk_score = round(fraud_prob * 100, 2)
    prediction = "FRAUD" if fraud_prob >= 0.5 else "LEGIT"

    band = risk_band(risk_score)
    action = recommended_action(band)

    return {
        "prediction": prediction,
        "fraud_probability": fraud_prob,
        "risk_score_0_to_100": risk_score,
        "risk_band": band,
        "recommended_action": action,
        "confidence": float(max(probabilities)),
        "model_version": "DistilBERT-Barclays-v1"
    }

# ==================================================
#  CLAIMS DESCRIPTION NORMALIZER - TERMINAL CLI
# ==================================================
import os
import json
import uuid
import logging
from dotenv import load_dotenv
from typing import Optional
from datetime import datetime
import re

# ML + NLP tools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Azure OpenAI
from openai import AzureOpenAI
from langdetect import detect

# ==================================================
# ENV CONFIG
# ==================================================
load_dotenv()
logging.basicConfig(level=logging.INFO)

logging.getLogger("httpx").setLevel(logging.WARNING)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# ==================================================
# FALLBACK ML MODEL
# ==================================================
texts = ["car accident", "bike stolen", "house fire", "mobile damaged", "flood water"]
labels = ["Accident", "Theft", "Fire", "Damage", "Flood"]

vec = TfidfVectorizer()
X = vec.fit_transform(texts)
clf = LogisticRegression().fit(X, labels)


def ml_predict(text: str) -> str:
    return clf.predict(vec.transform([text]))[0]


# ==================================================
# RULE ENGINES
# ==================================================
LOSS_RULES = {
    "accident": "Accident", "hit": "Accident", "crash": "Accident", "collision": "Accident",
    "theft": "Theft", "stolen": "Theft", "robbed": "Theft",
    "fire": "Fire", "burn": "Fire",
    "flood": "Flood", "water": "Flood"
}

ASSET_RULES = {
    "car": "Car", "vehicle": "Car", "bike": "Bike", "motorcycle": "Bike",
    "house": "House", "home": "House",
    "mobile": "Mobile", "phone": "Mobile"
}


def rule_based_loss(text: str):
    t = text.lower()
    for k, v in LOSS_RULES.items():
        if k in t:
            return v
    return None


def rule_based_asset(text: str):
    t = text.lower()
    for k, v in ASSET_RULES.items():
        if k in t:
            return v
    return None


# ==================================================
# LANGUAGE AGENT
# ==================================================
def language_agent(text: str):
    lang = detect(text)
    if lang == "hi" or lang == "mr":
        resp = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": f"Translate Hindi/Marathi to English:\n{text}"}]
        )
        return resp.choices[0].message.content, "Translated"
    return text, "English"


# ==================================================
# GPT EXTRACTION AGENT
# ==================================================
def extraction_agent(text: str) -> dict:
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": f"""
Return STRICT JSON only:
loss_type, severity (Low/Medium/High), affected_asset

Text:
{text}
"""
            }
        ],
        temperature=0.1
    )
    return json.loads(resp.choices[0].message.content)


# ==================================================
# OTHER MODULES
# ==================================================
def severity_rule(text: str) -> str:
    t = text.lower()
    score = 0
    if any(k in t for k in ["small", "minor", "scratch"]):
        score += 1
    if any(k in t for k in ["broken", "damaged", "bumper"]):
        score += 2
    if any(k in t for k in ["fire", "exploded", "burned", "total loss"]):
        score += 3
    if "stolen" in t or "theft" in t:
        return "Medium"
    if score >= 3:
        return "High"
    if score == 2:
        return "Medium"
    if score == 1:
        return "Low"
    return "Medium"


def fraud_agent(text: str, severity: str) -> str:
    score = 0
    if len(text) < 40:
        score += 1
    if severity == "High":
        score += 2
    return "High" if score >= 3 else "Medium" if score == 2 else "Low"


def confidence_agent(data: dict, text: str) -> float:
    score = 0.3
    if data["loss_type"] != "Other": score += 0.25
    if data["affected_asset"] != "Other": score += 0.25
    if len(text.split()) > 10: score += 0.1
    if len(text.split()) < 5: score -= 0.1
    return round(max(min(score, 0.98), 0.10), 2)


# ==================================================
# MAIN ORCHESTRATOR
# ==================================================
def normalize_claim(text: str):
    fallback = False
    try:
        extracted = extraction_agent(text)
    except:
        extracted = {
            "loss_type": ml_predict(text),
            "severity": "Medium",
            "affected_asset": "Other"
        }
        fallback = True

    if rule_based_loss(text):
        extracted["loss_type"] = rule_based_loss(text)
    if rule_based_asset(text):
        extracted["affected_asset"] = rule_based_asset(text)

    extracted["severity"] = severity_rule(text)
    fraud = fraud_agent(text, extracted["severity"])
    confidence = confidence_agent(extracted, text)

    policy = re.search(r"\b\d{5,12}\b", text)
    policy_no = policy.group(0) if policy else "UNKNOWN"
    claim_id = f"POL-{policy_no}-CLM-{str(uuid.uuid4())[:4]}"

    return extracted, fraud, confidence, fallback, claim_id, policy_no


# ==================================================
# TERMINAL EXECUTION
# ==================================================
if __name__ == "__main__":
    print("------- CLAIM NORMALIZER CLI -------")
    user_text = input("Enter Claim Text: ")

    translated, lang = language_agent(user_text)
    data, fraud, conf, fallback, claim_id, policy_no = normalize_claim(translated)

    summary = f"This claim involves a {data['loss_type']} affecting a {data['affected_asset']} with severity level {data['severity']}."
    action = "Start repair process." if data["severity"] != "High" else "Send tow truck & begin claim approval."

    human_msg = f"""
CLAIM RESULT
============
Claim ID: {claim_id}
Policy Number: {policy_no}
Language: {lang}
Fallback Used: {fallback}

Loss Type: {data['loss_type']}
Asset: {data['affected_asset']}
Severity: {data['severity']}
Fraud Risk: {fraud}
Confidence Score: {conf}

Summary:
{summary}

Recommended Action:
{action}
"""

    json_output = {
        "claim_id": claim_id,
        "policy_number": policy_no,
        "language": lang,
        "fallback_used": fallback,
        "incident": data,
        "ai_risk": {"fraud": fraud, "confidence_score": conf},
        "summary": summary,
        "recommended_action": action,
        "raw_input": user_text,
        "normalized_on": datetime.utcnow().isoformat() + "Z"
    }

    print("\n--- JSON OUTPUT ---")
    print(json.dumps(json_output, indent=2))
    print("\n--- HUMAN SUMMARY ---")
    print(human_msg)

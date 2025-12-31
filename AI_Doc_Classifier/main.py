# =======================================================
# Insurance Document Classification Agent – CLI VERSION
# =======================================================

import os
import re
import pdfplumber
import pandas as pd
from PIL import Image
import pytesseract
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# -----------------------------
# MODELS & LABELS
# -----------------------------
LABELS = ["Invoice", "Claim Form", "Insurance Policy", "Inspection Report"]

print("Loading AI models... please wait...")

# Zero-shot LLM
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Sentence transformer model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Embedding labels
LABEL_SEED_TEXT = {
    "Invoice": "This document contains payment details, bill numbers, GST, total amount due",
    "Claim Form": "Insurance claim request including claimant name, policy holder, incident details",
    "Insurance Policy": "Coverage details, exclusions, policy period, premium payable, deductible",
    "Inspection Report": "Vehicle / property survey assessment details, surveyor comments, damage report"
}
LABEL_EMBED = embedder.encode(list(LABEL_SEED_TEXT.values()))

# Load ML model if exists
ML_MODEL_PATH = "document_classifier.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

trained_model = None
tfidf_vectorizer = None

if os.path.exists(ML_MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    trained_model = joblib.load(ML_MODEL_PATH)
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
    print("Custom ML model loaded successfully.")
else:
    print("⚠ ML model not found – using Zero-Shot + Embedding only.")


# -------------------------------------------------------
# PDF / OCR text extractor
# -------------------------------------------------------
def extract_text(file_path):
    file_path = file_path.lower()
    text = ""

    # PDF
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for pg in pdf.pages:
                extracted = pg.extract_text() or ""
                text += extracted
        return text
    # Image
    else:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text


# -------------------------------------------------------
# Embedding classifier
# -------------------------------------------------------
def embedding_classifier(text):
    text_vec = embedder.encode([text])
    sims = cosine_similarity(text_vec, LABEL_EMBED)[0]
    index = np.argmax(sims)
    return LABELS[index], float(sims[index])


# -------------------------------------------------------
# ML model classifier
# -------------------------------------------------------
def ml_classifier(text):
    if trained_model is None or tfidf_vectorizer is None:
        return None, 0.0
    vec = tfidf_vectorizer.transform([text])
    pred = trained_model.predict(vec)[0]
    conf = max(trained_model.predict_proba(vec)[0])
    return pred, float(conf)


# -------------------------------------------------------
# Regex extraction for fields
# -------------------------------------------------------
def extract_fields(text):
    invoice_no = re.findall(r"(INV[- ]?\d+)", text, re.IGNORECASE)
    claim_id = re.findall(r"(CLM[- ]?\d+)", text, re.IGNORECASE)
    policy_id = re.findall(r"(PLC[- ]?\d+)", text, re.IGNORECASE)
    dates = re.findall(r"\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b", text)

    return {
        "invoice_no": invoice_no[0] if invoice_no else None,
        "claim_id": claim_id[0] if claim_id else None,
        "policy_id": policy_id[0] if policy_id else None,
        "date": dates[0] if dates else None
    }


# -------------------------------------------------------
# Main classification logic
# -------------------------------------------------------
def classify_document(file_path):
    text = extract_text(file_path)
    if not text.strip():
        return ("Unknown", 0.0, "No Text Found", "", {})

    # 1️⃣ Zero-shot
    zero_shot_result = zero_shot(text, LABELS)
    zs_label = zero_shot_result["labels"][0]
    zs_score = float(zero_shot_result["scores"][0])

    # 2️⃣ Embedding
    emb_label, emb_score = embedding_classifier(text)

    # 3️⃣ ML (if exists)
    ml_label, ml_score = ml_classifier(text)

    # Select best score
    best_label = zs_label
    best_score = zs_score
    best_method = "Zero-Shot LLM"

    if emb_score > best_score:
        best_label = emb_label
        best_score = emb_score
        best_method = "Embedding Classifier"

    if ml_label is not None and ml_score > best_score:
        best_label = ml_label
        best_score = ml_score
        best_method = "TF-IDF ML Classifier"

    # Extract metadata fields
    fields = extract_fields(text)

    return best_label, best_score, best_method, text[:600], fields


# -------------------------------------------------------
# Batch mode – classify entire folder
# -------------------------------------------------------
def classify_folder(folder):
    results = []
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path):
            label, score, method, preview, fields = classify_document(path)
            results.append([f, label, round(score*100,2), method,
                            fields.get("invoice_no"), fields.get("claim_id"),
                            fields.get("policy_id"), fields.get("date")])
    return results


# -------------------------------------------------------
# Export Excel + PDF
# -------------------------------------------------------
def export_results(results, excel="results.xlsx", pdf="results.pdf"):
    df = pd.DataFrame(results, columns=[
        "File", "Label", "Confidence%", "Method",
        "Invoice No", "Claim ID", "Policy ID", "Date"
    ])
    df.to_excel(excel, index=False)
    print(f"Excel file saved → {excel}")

    c = canvas.Canvas(pdf, pagesize=letter)
    y = 750
    c.drawString(40,770,"Document Classification Report")
    for row in df.values.tolist():
        c.drawString(40,y,str(row))
        y -= 20
        if y < 40:  # Create new page when space runs out
            c.showPage()
            y = 750
    c.save()
    print(f"PDF report saved → {pdf}")


# -------------------------------------------------------
# CLI RUN
# -------------------------------------------------------
if __name__ == "__main__":
    print("\n===============================================")
    print("📌 Insurance Document Classification – CLI Tool")
    print("===============================================\n")

    mode = input("Choose mode:\n1 – Single File\n2 – Batch Folder\nEnter: ")

    if mode == "1":
        file = input("Enter file path (PDF/Image): ")

        if not os.path.exists(file):
            print("❌ File not found")
            exit()

        label, score, method, preview, fields = classify_document(file)
        print("\n----- RESULT -----")
        print(f"Prediction: {label}")
        print(f"Confidence: {round(score*100,2)}%")
        print(f"Method Used: {method}")

        print("\n----- Extracted Fields -----")
        for k,v in fields.items():
            print(f"{k}: {v}")

        print("\n----- Extracted Text Preview -----")
        print(preview)

    elif mode == "2":
        folder = input("Enter folder path: ")
        results = classify_folder(folder)
        print("\nBatch completed. Exporting...")

        export_results(results)
        print("DONE.")

    else:
        print("Invalid Option")

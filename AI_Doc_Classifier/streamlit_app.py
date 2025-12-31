import streamlit_app as st
import os
import pdfplumber
import io
from PIL import Image
import pytesseract
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pandas as pd
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas 
import sqlite3
from datetime import datetime

# -----------------------------
# LOAD MODELS
# -----------------------------
LABELS = ["Invoice", "Claim Form", "Insurance Policy", "Inspection Report"]

st.title("🧾 AI Insurance Document Classification Tool")
st.caption("OCR | Zero-Shot AI | ML Model | Metadata Extraction | Batch Mode | Excel/PDF Export | History Dashboard")

@st.cache_resource
def load_models():
    zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    LABEL_SEED_TEXT = {
        "Invoice": "This document contains payment details, bill numbers, GST, total amount due",
        "Claim Form": "Insurance claim request including claimant name, policy holder, incident details",
        "Insurance Policy": "Coverage details, exclusions, policy period, premium payable, deductible",
        "Inspection Report": "Vehicle / property survey assessment details, surveyor comments, damage report"
    }
    LABEL_EMBED = embedder.encode(list(LABEL_SEED_TEXT.values()))

    trained_model = None
    tfidf_vectorizer = None
    if os.path.exists("document_classifier.pkl") and os.path.exists("tfidf_vectorizer.pkl"):
        trained_model = joblib.load("document_classifier.pkl")
        tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return zero_shot, embedder, LABEL_EMBED, trained_model, tfidf_vectorizer

zero_shot, embedder, LABEL_EMBED, trained_model, tfidf_vectorizer = load_models()

# -----------------------------
# DATABASE (ADDED)
# -----------------------------
def init_db():
    conn = sqlite3.connect("history.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            label TEXT,
            confidence REAL,
            method TEXT,
            invoice TEXT,
            claim TEXT,
            policy TEXT,
            date_detected TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    return conn

conn = init_db()

def save_history(filename, label, confidence, method, fields):
    conn.execute("""
        INSERT INTO history (filename, label, confidence, method, invoice, claim, policy, date_detected, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        filename, label, confidence, method,
        fields.get("Invoice No"), fields.get("Claim ID"),
        fields.get("Policy ID"), fields.get("Date"),
        str(datetime.now())
    ))
    conn.commit()

# -----------------------------
# EXTRACT OCR / TEXT
# -----------------------------
def extract_text_bytes(bytes_data, name):
    name = name.lower()
    if name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(bytes_data)) as pdf:
            for pg in pdf.pages:
                extracted = pg.extract_text() or ""
                text += extracted
        return text
    else:
        img = Image.open(io.BytesIO(bytes_data))
        text = pytesseract.image_to_string(img)
        return text

# -----------------------------
# Classifiers
# -----------------------------
def embedding_classifier(text):
    text_vec = embedder.encode([text])
    sims = cosine_similarity(text_vec, LABEL_EMBED)[0]
    index = np.argmax(sims)
    return LABELS[index], float(sims[index])

def ml_classifier(text):
    if trained_model is None or tfidf_vectorizer is None:
        return None, 0.0
    vec = tfidf_vectorizer.transform([text])
    pred = trained_model.predict(vec)[0]
    conf = max(trained_model.predict_proba(vec)[0])
    return pred, float(conf)

# -----------------------------
# Regex metadata extraction
# -----------------------------
def extract_fields(text):
    invoice_no = re.findall(r"(INV[- ]?\d+)", text, re.IGNORECASE)
    claim_id = re.findall(r"(CLM[- ]?\d+)", text, re.IGNORECASE)
    policy_id = re.findall(r"(PLC[- ]?\d+)", text, re.IGNORECASE)
    dates = re.findall(r"\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b", text)
    return {
        "Invoice No": invoice_no[0] if invoice_no else None,
        "Claim ID": claim_id[0] if claim_id else None,
        "Policy ID": policy_id[0] if policy_id else None,
        "Date": dates[0] if dates else None
    }

# -----------------------------
# CLASSIFICATION LOGIC
# -----------------------------
def classify_doc(bytes_data, name):
    text = extract_text_bytes(bytes_data, name)
    if not text.strip():
        return ("Unknown", 0.0, "No Text Found", "", {})

    zero_res = zero_shot(text, LABELS)
    zs_label = zero_res["labels"][0]
    zs_score = float(zero_res["scores"][0])

    emb_label, emb_score = embedding_classifier(text)
    ml_label, ml_score = ml_classifier(text)

    best_label = zs_label
    best_score = zs_score
    best_method = "Zero-Shot AI"

    if emb_score > best_score:
        best_label = emb_label
        best_score = emb_score
        best_method = "Embeddings AI"

    if ml_label is not None and ml_score > best_score:
        best_label = ml_label
        best_score = ml_score
        best_method = "TF-IDF ML"

    fields = extract_fields(text)
    return best_label, best_score, best_method, text[:600], fields

# -----------------------------
# File Upload UI
# -----------------------------
uploaded_files = st.file_uploader("Upload document(s)", accept_multiple_files=True, type=["pdf", "png", "jpg", "jpeg"])

results = []

if uploaded_files:
    for file in uploaded_files:
        with st.spinner(f"Processing {file.name} ..."):
            label, score, method, preview, fields = classify_doc(file.read(), file.name)

        st.success(f"{file.name} → {label} ({round(score*100,2)}%) [{method}]")
        st.json(fields)
        st.text_area("Text Preview", preview, height=150, key=file.name)

        results.append([file.name, label, round(score*100,2), method,
                        fields.get("Invoice No"), fields.get("Claim ID"),
                        fields.get("Policy ID"), fields.get("Date")])

        # SAVE TO DB (ADDED)
        save_history(file.name, label, round(score*100,2), method, fields)

# -----------------------------
# EXPORT RESULTS
# -----------------------------
def export_pdf(df, pdf_filename):
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    y = 750
    c.drawString(40,770,"Document Classification Report")
    for row in df.values.tolist():
        c.drawString(40,y,str(row))
        y -= 20
        if y < 40:
            c.showPage()
            y = 750
    c.save()
    return pdf_filename

if results:
    df = pd.DataFrame(results, columns=[
        "File","Label","Confidence%","Method","Invoice No","Claim ID","Policy ID","Date"
    ])

    st.dataframe(df)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Excel"):
            df.to_excel("results.xlsx", index=False)
            st.success("Excel saved → results.xlsx")
            st.download_button("Download results.xlsx", open("results.xlsx","rb"), "results.xlsx")

    with col2:
        if st.button("Export PDF"):
            export_pdf(df, "results.pdf")
            st.success("PDF saved → results.pdf")
            st.download_button("Download results.pdf", open("results.pdf","rb"), "results.pdf")

# -----------------------------
# HISTORY DASHBOARD (ADDED)
# -----------------------------
st.subheader("📜 Upload History Dashboard")

if st.button("Show Upload History"):
    hist = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    st.dataframe(hist)

    if st.button("Clear History"):
        conn.execute("DELETE FROM history")
        conn.commit()
        st.warning("History Cleared Successfully")


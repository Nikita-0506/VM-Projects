AI Insurance Document Classification Agent

A hybrid NLP-powered system that analyzes PDF or scanned image documents and automatically classifies them into standard insurance document types:

Invoice

Claim Form

Insurance Policy

Inspection Report

Using OCR extraction + Zero-Shot LLM classification + Sentence Embedding matching + Optional ML classifier, this tool enables batch folder processing, Excel/PDF reporting, and a full Streamlit dashboard with history logging.

This project demonstrates:

PDF & Image OCR (pdfplumber + pytesseract)

Zero-shot text classification (facebook/bart-large-mnli)

Sentence-Transformer embedding model (all-MiniLM-L6-v2)

Optional ML classifier (TF-IDF + Logistic Regression)

Regex metadata extraction (Invoice number, Claim ID, Date)

Streamlit UI with results viewer & export

SQLite database history tracking

1️ Architecture Overview
Component	Technology
Backend Engine	Python – CLI Script
OCR Engine	pdfplumber, pytesseract
NLP Classification	facebook/bart-large-mnli
Embedding Similarity	Sentence Transformers
Optional ML Classifier	TF-IDF Vectorizer + Joblib Model
UI Interface	Streamlit (app.py)
Storage	SQLite history.db
Reporting	ReportLab – PDF Export
2️ Core Features
🧠 AI Classification Modes (Priority-Based)

When classifying a document, system chooses BEST-match using:

Zero-shot model

Embedding similarity

ML TF-IDF classifier (if exists)

Output includes:

Best predicted label

Confidence %

Which AI method was used

🔍 Metadata Extraction (Regex)

Extracts:

Invoice No – INV-XXXX

Claim ID – CLM-XXXX

Policy ID – PLC-XXXX

Date – dd/mm/yyyy or dd-mm-yyyy

📁 Batch Folder Processing

Run classification for multiple documents in a directory → auto-generate:

results.xlsx

results.pdf

🖥 Streamlit Application

Upload & classify multiple files

View extracted fields & preview text

Export Excel & PDF

History Dashboard (SQLite DB)

Clear records

A. Install Dependencies
pip install -r requirements.txt


OCR dependency (Windows example):

choco install tesseract

B. Run CLI Script (Terminal)
python main.py


CLI example:

📌 Insurance Document Classification – CLI Tool

Choose mode:
1 – Single File
2 – Batch Folder

Enter: 1
Enter file path: tests/invoice.pdf

----- RESULT -----
Prediction: Invoice
Confidence: 93.55%
Method Used: Embedding Classifier
Invoice No: INV-2203
Date: 22/10/2024


Batch mode:

python main.py
Enter folder path: ./docs/
Excel file saved → results.xlsx
PDF report saved → results.pdf

C. Run Streamlit Dashboard
streamlit run app.py


Visit UI:

http://localhost:8501

Dashboard Features
Feature	Description
File Upload	Upload multiple PDF / scanned images
Auto Classification	AI model runs in background
JSON Results	Label, confidence, metadata
Download	Excel + PDF output
History	View previous uploaded results
Database	Saves logs to history.db
D. Example JSON Output

(Not actual run output — formatted for documentation)

{
  "file": "car_policy.pdf",
  "label": "Insurance Policy",
  "confidence": 0.91,
  "method": "Zero-Shot AI",
  "fields": {
    "invoice_no": null,
    "claim_id": null,
    "policy_id": "PLC-1982",
    "date": "12-08-2024"
  }
}

3 Folder Structure
│── main.py                     # CLI document classifier
│── app.py                      # Streamlit dashboard
│── document_classifier.pkl     # Optional ML model (if trained)
│── tfidf_vectorizer.pkl        # Optional vectorizer
│── history.db                  # SQLite logs
│── results.xlsx                # Generated output
│── results.pdf                 # Generated PDF report
│── requirements.txt
│── README.md

4 Future Enhancements

FastAPI backend → REST endpoints (/classify, /upload, /results)

Azure Blob Storage for file archival

JWT based authentication for dashboard

OCR language pack support – Hindi/Marathi

Smart section extraction → e.g., “Policy Period”, “Premium Due”, “Claim Reason”

✨ Developed By

Nikita Pachkate
Data Scientist – Insurance AI Automation Specialist

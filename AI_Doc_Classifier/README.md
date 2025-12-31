## AI Insurance Document Classification Agent 

A hybrid NLP-powered system that analyzes PDF or scanned image documents and automatically classifies them into standard insurance document types:

   a. Invoice

   b. Claim Form

   c. Insurance Policy

   d. Inspection Report

Using OCR extraction + Zero-Shot LLM classification + Sentence Embedding matching + Optional ML classifier, this tool enables batch folder processing, Excel/PDF reporting, and a full Streamlit dashboard with history logging.

# This project demonstrates:

   a. PDF & Image OCR (pdfplumber + pytesseract)

   b. Zero-shot text classification (facebook/bart-large-mnli)

   c. Sentence-Transformer embedding model (all-MiniLM-L6-v2)

   d. Optional ML classifier (TF-IDF + Logistic Regression)

   e. Regex metadata extraction (Invoice number, Claim ID, Date)

   f. Streamlit UI with results viewer & export

   g. SQLite database history tracking
   

# 1️ Architecture Overview

Component                      	Technology

Backend Engine              	Python – CLI Script
OCR Engine	                  pdfplumber, pytesseract
NLP Classification	          facebook/bart-large-mnli
Embedding Similarity	        Sentence Transformers
Optional ML Classifier      	TF-IDF Vectorizer + Joblib Model
UI Interface                	Streamlit (app.py)
Storage                     	SQLite history.db
Reporting                   	ReportLab – PDF Export


# 2️ Core Features

# 🧠 AI Classification Modes (Priority-Based)

When classifying a document, system chooses BEST-match using:

   a. Zero-shot model

   b. Embedding similarity

   c. ML TF-IDF classifier (if exists)

Output includes:

   a. Best predicted label

   b. Confidence %

   c. Which AI method was used
   
# 🔍 Metadata Extraction (Regex)

Extracts:

   a. Invoice No – INV-XXXX

   b. Claim ID – CLM-XXXX

   c. Policy ID – PLC-XXXX

   d. Date – dd/mm/yyyy or dd-mm-yyyy

# 📁 Batch Folder Processing

Run classification for multiple documents in a directory → auto-generate:

   a. results.xlsx

   b. results.pdf

# 🖥 Streamlit Application

   a. Upload & classify multiple files

   b. View extracted fields & preview text

   c. Export Excel & PDF

   d. History Dashboard (SQLite DB)

   e. Clear records

# A. Install Dependencies
   pip install -r requirements.txt


# OCR dependency (Windows example):

   choco install tesseract

# B. Run CLI Script (Terminal)
   python main.py


# CLI example:

" [  📌 Insurance Document Classification – CLI Tool

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
Date: 22/10/2024  ] "


# Batch mode:

"  [ python main.py
Enter folder path: ./docs/
Excel file saved → results.xlsx
PDF report saved → results.pdf ]  "

# C. Run Streamlit Dashboard
   streamlit run app.py


# Visit UI:

   http://localhost:8501


# Dashboard Features

Feature            Description
File Upload	       Upload multiple PDF / scanned images
Auto               Classification	AI model runs in background
JSON Results       Label, confidence, metadata
Download           Excel + PDF output
History            View previous uploaded results
Database	         Saves logs to history.db


# D. Example JSON Output

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

# 3 Folder Structure

│── main.py                     # CLI document classifier
│── app.py                      # Streamlit dashboard
│── document_classifier.pkl     # Optional ML model (if trained)
│── tfidf_vectorizer.pkl        # Optional vectorizer
│── history.db                  # SQLite logs
│── results.xlsx                # Generated output
│── results.pdf                 # Generated PDF report
│── requirements.txt
│── README.md

# 4 Future Enhancements

   a. FastAPI backend → REST endpoints (/classify, /upload, /results)

   b. Azure Blob Storage for file archival

   c. JWT based authentication for dashboard

   d. OCR language pack support – Hindi/Marathi

   e. Smart section extraction → e.g., “Policy Period”, “Premium Due”, “Claim Reason”

# ✨ Developed By

Nikita Pachkate
Data Scientist – Insurance AI Automation Specialist


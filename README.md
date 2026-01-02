🏦 Insurance GenAI Solutions
End-to-End AI Automation Suite for the Insurance Industry

Insurance GenAI Solutions is a comprehensive collection of enterprise-grade AI systems designed to automate and enhance critical insurance workflows using Generative AI, Machine Learning, Computer Vision, and NLP.

This repository consolidates four production-ready insurance AI applications into a single platform, each solving a real-world insurance problem with both CLI tools and Streamlit dashboards.

## 🎯 What This Platform Solves
#### Insurance Challenges vs AI Solutions

| Insurance Area     | Traditional Problem                     | AI-Driven Solution                          |
|--------------------|------------------------------------------|----------------------------------------------|
| Document Intake    | Manual classification & delays           | Automated document categorization             |
| Claims Processing  | Unstructured, multilingual inputs        | Structured claim normalization                |
| Fraud Detection    | Late fraud identification                | Proactive risk scoring & alerts               |
| Policy Understanding | Long, complex policy PDFs             | Plain-English summaries + chatbot             |

🧠 Included AI Systems
1️⃣ 📄🤖 AI Insurance Document Classification Agent

Purpose:
Automatically classifies insurance documents from PDFs or scanned images into standard document types.

Supported Document Types

Invoice

Claim Form

Insurance Policy

Inspection Report

Key Capabilities

PDF & Image OCR (pdfplumber, pytesseract)

Zero-Shot LLM classification (facebook/bart-large-mnli)

Sentence-Transformer embedding similarity

Optional ML fallback (TF-IDF + Logistic Regression)

Regex-based metadata extraction

Batch folder processing

Excel & PDF reporting

Streamlit dashboard with SQLite history

AI Decision Logic (Priority-Based)

Zero-Shot LLM

Embedding Similarity

ML TF-IDF Classifier (if available)

Output Includes

Predicted document label

Confidence score

AI method used

Extracted metadata

2️⃣ 📑 Claims Description Normalizer (Agentic Claims AI)

Purpose:
Transforms messy insurance claim inputs (text, voice, images) into structured, actionable claim intelligence.

Core Features

Loss type, severity & affected asset detection

Auto-translation (Hindi / Marathi → English)

Fraud risk estimation

ML fallback classifier

YOLO-based vehicle damage detection

Voice-to-text claim intake

Enterprise-grade PDF claim reports

Automated email delivery

SQLite history logging

Analytics dashboard (fraud & severity trends)

Supported Inputs

Free-text claims

Voice recordings

Accident images

Mixed-language claims

3️⃣ 🛡️ Fraud Detection Copilot

Purpose:
AI-powered fraud scoring and analytics system for identifying high-risk insurance claims.

Technical Highlights

XGBoost ML model with SMOTE balancing

Hybrid risk scoring (ML + rules + text similarity)

TF-IDF cosine similarity against known fraud cases

Automatic threshold-based risk categorization

PDF fraud summary reports

Email alerts for high-risk claims

Streamlit analytics dashboard

SQLite-based audit storage

Fraud Risk Categories

Low

Medium

High

4️⃣ 📘🤖 Agentic Policy Summary Assistant

Purpose:
Converts lengthy insurance policy PDFs into easy-to-understand summaries and structured compliance data.

Key Outputs

Plain-English policy summaries

Structured JSON:

Coverage

Exclusions

Limits

Eligibility

Waiting periods

Disclaimers

Hindi & Marathi translations

Downloadable PDF reports

Master report generation (10+ PDFs)

Policy-aware chatbot

Full history tracking & analytics

Chatbot Modes

Normal Chatbot

Policy-Based Only (answers strictly from uploaded policy)

🏗️ Architecture Overview

┌──────────────────────────────────────────────────────────────────────────┐
│                    INSURANCE GENAI SOLUTIONS SUITE                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────┐  ┌─────────────────────────┐              │
│  │ DOCUMENT CLASSIFICATION │  │ CLAIMS NORMALIZATION     │              │
│  │ AGENT                   │  │ ENGINE                  │              │
│  │                         │  │                         │              │
│  │ • PDF / Image OCR       │  │ • NLP Claim Parsing     │              │
│  │ • Zero-Shot AI          │  │ • Loss & Severity AI   │              │
│  │ • Embedding Similarity  │  │ • Fraud Risk Scoring   │              │
│  │ • Metadata Extraction  │  │ • YOLO Damage Detection│              │
│  │ • Batch Processing     │  │ • Voice-to-Text Intake │              │
│  └─────────────────────────┘  └─────────────────────────┘              │
│                                                                          │
│  ┌─────────────────────────┐  ┌─────────────────────────┐              │
│  │ FRAUD DETECTION         │  │ POLICY SUMMARY           │              │
│  │ COPILOT                 │  │ ASSISTANT                │              │
│  │                         │  │                         │              │
│  │ • XGBoost ML Scoring    │  │ • Policy Summarization  │              │
│  │ • Rule-Based Risk Logic │  │ • Structured JSON Output│              │
│  │ • Text Similarity AI   │  │ • Multi-language Support│              │
│  │ • PDF & Email Alerts   │  │ • RAG-based Chatbot     │              │
│  │ • Analytics Dashboard  │  │ • Master Report Builder │              │
│  └─────────────────────────┘  └─────────────────────────┘              │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                  COMMON ENTERPRISE AI PLATFORM                      │ │
│  ├────────────────────────────────────────────────────────────────────┤ │
│  │ • Azure OpenAI (GPT) Integration    • Streamlit UI Framework        │ │
│  │ • OCR & PDF Processing Utilities   • SQLite Persistence             │ │
│  │ • ML Models (TF-IDF, XGBoost)      • Audit Logs & History Tracking  │ │
│  │ • Modular & Scalable Architecture • Email & Report Automation      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘


🧩 High-Level Architecture

PDF / Image / Text / Voice / CSV
            ↓
     OCR / NLP / Vision
            ↓
   LLM + ML + Rule Engines
            ↓
 Structured JSON Intelligence
            ↓
 Reports • Dashboards • Emails
            ↓
 SQLite Logs & Analytics

🧠 System Comparison Matrix

| Feature                    | Document Classification Agent     | Claims Description Normalizer     | Fraud Detection Copilot           | Agentic Policy Summary Assistant   |
| -------------------------- | --------------------------------- | --------------------------------- | --------------------------------- | ---------------------------------- |
| **Primary Focus**          | Insurance document type detection | Claim understanding & structuring | Fraud risk prediction & alerts    | Policy understanding & compliance  |
| **Core AI Techniques**     | Zero-shot LLM + Embeddings + ML   | NLP extraction + Rules + YOLO     | XGBoost + Rules + Text Similarity | LLM summarization + RAG            |
| **Input Data**             | PDF & scanned images              | Text, images, voice               | Claims CSV data                   | Policy PDFs                        |
| **Key Output**             | Document label + metadata         | Structured claim JSON + PDF       | Fraud score & risk category       | Policy summary + structured JSON   |
| **Decision Style**         | Assistive classification          | Semi-automated with review        | Automated with human override     | Assistive (non-decisional)         |
| **Explainability**         | Confidence + method used          | Summary + severity reasoning      | Probability + rules applied       | Human-readable summaries           |
| **UI Interface**           | Streamlit dashboard               | Streamlit multi-page UI           | Streamlit analytics app           | Streamlit policy portal            |
| **Automation Level**       | Medium                            | High                              | High                              | Medium                             |
| **Regulatory Sensitivity** | Medium                            | High                              | Very High                         | High                               |
| **Enterprise Readiness**   | Batch + audit logs                | End-to-end claim pipeline         | Production fraud monitoring       | Compliance & customer transparency |


🗂 Combined Project Structure
insurance-genai-solutions/
│
├── document_classifier/
│   ├── main.py
│   ├── app.py
│   └── history.db
│
├── claims_normalizer/
│   ├── main.py
│   ├── streamlit.py
│   └── claims.db
│
├── fraud_detection_copilot/
│   ├── main.py
│   ├── app.py
│   └── fraud_results.db
│
├── policy_summary_assistant/
│   ├── main.py
│   ├── streamlit_app.py
│   └── policy_history.db
│
├── requirements.txt
├── .env
└── README.md

⚙️ Technology Stack

LLMs: Azure OpenAI (GPT-4 / GPT-4o)

NLP: HuggingFace Transformers, Sentence-Transformers

ML: XGBoost, Scikit-Learn, SMOTE

Vision: YOLOv8, OpenCV

OCR: pdfplumber, pytesseract

UI: Streamlit

Storage: SQLite

Reporting: ReportLab, FPDF

Email: SMTP (Yagmail)

🚀 Setup & Execution
Install Dependencies
pip install -r requirements.txt

Configure Environment Variables (.env)
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_VERSION=
AZURE_OPENAI_DEPLOYMENT=

EMAIL_USER=
EMAIL_PASSWORD=
NOTIFY_EMAIL_TO=

Run Any Module

cd <module_folder>
python main.py
# OR
streamlit run app.py

## 📈 Future Enhancements

FastAPI microservices for all modules

Centralized RAG knowledge base

JWT authentication & role-based access

SHAP-based fraud explainability

Cloud deployment (Azure / AWS)

Real-time claim ingestion (Kafka)

## ✨ Developed By

#### Nikita Pachkate
Insurance AI Engineer & Data Scientist
Specializing in GenAI-powered insurance automation, risk intelligence, and decision-support systems.

# 📘🤖 Agentic Policy Summary Assistant

### A full-stack AI system that reads insurance policy PDF documents and automatically generates:

   a. Plain-English policy summaries

   b. Structured JSON explaining coverage, exclusions, limits, waiting periods, eligibility, disclaimers

   c. Optional translation into Hindi / Marathi

   d. PDF downloadable reports

   e. Automated email delivery

   f. Analytics visualizations

   g. Long-term history storage

   h. Chatbot capable of answering questions based ONLY on uploaded policy content

### This project demonstrates:

   a. Azure OpenAI GPT model for summarization & RAG-style structuring

   b. PDF text extraction (pdfplumber)

   c. PDF export (FPDF / ReportLab Platypus)

   d. Multi-language translation

   e. Interactive Streamlit App UI

   f. Chat session logging into SQLite

   g. Multi-file master PDF report generation (10+ policies in one document)

   h. Email workflow using SMTP + Yagmail

## 1️ Architecture Overview

#### Component Technology Stack

| Component        | Technology                                                |
|------------------|-----------------------------------------------------------|
| Backend Engine   | CLI (Python)                                              |
| Frontend UI     | Streamlit                                                 |
| AI Model         | Azure OpenAI GPT (Chat Completions)                       |
| PDF OCR          | pdfplumber                                                |
| PDF Rendering    | ReportLab, FPDF                                          |
| Multi-Language   | GPT-based Translation (Hindi / Marathi)                   |
| Email Service    | Yagmail (SMTP)                                           |
| DB Storage       | SQLite (policy_history.db, chat_history.db)               |
| Deployment       | Localhost, Azure VM, Docker capable                       |


## 2️ Core Features

### Policy Summarization

A. Converts full PDF → summarised text (200-word adjustable)

B. Converts full policy → structured JSON including:
```text
   "coverage": []

   "exclusions": []

   "limits": []

   "eligibility": []

   "waiting_periods": []

   "disclaimers": []
```
### Multi-language Output

   a. Hindi summary

   b. Marathi summary

   c. English only

### Policy Compliance Dashboard (Streamlit)

   a. Upload multiple PDF files

   b. Generate summaries instantly

   c. Download individual PDF reports

   d. Track all generated outputs in history database

### Master Report Generator (10+ PDFs)

   a. Upload 10+ policies → one consolidated MASTER_POLICY_REPORT.pdf

   b. Optional email sending of master report

### Email Delivery

   a. Send single PDF summary or master report

   b. Uses SMTP via .env credentials

### Chatbot (Policy-Aware)

Modes:

   a. Normal Chatbot (generic GPT-like)

   b. Policy-Based Only (answers strictly using uploaded policy text)

#### All chat history is saved in a database & can be reopened later.

### A. Install Dependencies

pip install -r requirements.txt

#### Recommended PDF font support (Linux):

sudo apt-get install fonts-noto-cjk

## B. Environment Variables (.env)

#### Create .env with:


AZURE_OPENAI_API_KEY=xxxx
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-model-name
EMAIL_USER=your@gmail.com
EMAIL_PASSWORD=your_app_password
NOTIFY_EMAIL_TO=recipient@example.com


## C. Run CLI Version (main.py)

python main.py

#### Example usage:

```text
[INFO] Reading PDF...
[INFO] Generating JSON structured summary...
[INFO] Generating English summary...
[INFO] Translating (Hindi)...
PDF saved → policy_summary.pdf
Do you want to send this PDF by email? (yes/no):
]
```
### Arguments available:

python main.py --pdf HealthPolicy.pdf --length 150 --lang hindi

## D. Run Streamlit Dashboard

streamlit run streamlit_app.py

### Visit in browser:

http://localhost:8501


### Pages (Tabs):

#### Application Tabs

| Tab               | Description                                       |
|-------------------|---------------------------------------------------|
| 🧾 Summarize Policy | Upload PDF → Summary + PDF + Email               |
| 📚 Master Report   | Upload 10+ PDFs → Consolidated Report             |
| 📜 History         | View all generated reports                        |
| 📊 Analytics       | Chart view of coverage & exclusions               |
| 🤖 Chatbot         | Ask questions about policy contents               |



## 3️ Example Output – JSON Summary
```text
{
  "coverage": ["Hospitalization", "Room Rent", "Pre-Post Care"],
  "exclusions": ["Pre-existing diseases", "Dental cosmetic care"],
  "limits": ["Max 5L per year"],
  "eligibility": ["Age 18-60"],
  "waiting_periods": ["30 days general", "2 years maternity"],
  "disclaimers": ["No guarantee of policy renewal"]
}
```
## 4 Folder Structure

```text
project/
│── main.py                        # CLI summarizer + email
│── streamlit_app.py               # Web dashboard UI
│── policy_history.db              # Generated policy history
│── chat_history.db                # Chat conversations
│── requirements.txt
│── .env
│── README.md
│── /MASTER_POLICY_REPORT.pdf      # Generated (optional)
│── /<file>_summary.pdf            # Auto-generated report(s)
```
## 5 Analytics Capabilities

#### Extracts JSON → builds charts:

   a. Most common coverage terms

   b. Most frequent exclusions

   c. Date-wise usage tracking

#### Uses: Plotly Express

## 6 Future Enhancements

   a. Add OCR for scanned PDF policy images

   b. Add JWT admin login & session auth

   c. FASTAPI REST endpoints for automation pipelines

   d. Export JSON data to Blob Storage or Salesforce Insurance Cloud

   e. Integrate GPT-RAG document search across multiple policies

## License

This is a demo project for educational purposes.

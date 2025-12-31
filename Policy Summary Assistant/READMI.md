Agentic Policy Summary Assistant

A full-stack AI system that reads insurance policy PDF documents and automatically generates:

Plain-English policy summaries

Structured JSON explaining coverage, exclusions, limits, waiting periods, eligibility, disclaimers

Optional translation into Hindi / Marathi

PDF downloadable reports

Automated email delivery

Analytics visualizations

Long-term history storage

Chatbot capable of answering questions based ONLY on uploaded policy content

This project demonstrates:

Azure OpenAI GPT model for summarization & RAG-style structuring

PDF text extraction (pdfplumber)

PDF export (FPDF / ReportLab Platypus)

Multi-language translation

Interactive Streamlit App UI

Chat session logging into SQLite

Multi-file master PDF report generation (10+ policies in one document)

Email workflow using SMTP + Yagmail

1️ Architecture Overview
Component	Technology
Backend Engine	CLI (Python)
Frontend UI	Streamlit
AI Model	Azure OpenAI GPT (Chat Completions)
PDF OCR	pdfplumber
PDF Rendering	ReportLab, FPDF
Multi-Language	GPT-based Translation (Hindi / Marathi)
Email Service	Yagmail (SMTP)
DB Storage	SQLite (policy_history.db, chat_history.db)
Deployment	Localhost, Azure VM, Docker capable
2️ Core Features
Policy Summarization

Converts full PDF → summarised text (200-word adjustable)

Converts full policy → structured JSON including:

"coverage": []

"exclusions": []

"limits": []

"eligibility": []

"waiting_periods": []

"disclaimers": []

Multi-language Output

Hindi summary

Marathi summary

English only

Policy Compliance Dashboard (Streamlit)

Upload multiple PDF files

Generate summaries instantly

Download individual PDF reports

Track all generated outputs in history database

Master Report Generator (10+ PDFs)

Upload 10+ policies → one consolidated MASTER_POLICY_REPORT.pdf

Optional email sending of master report

Email Delivery

Send single PDF summary or master report

Uses SMTP via .env credentials

Chatbot (Policy-Aware)

Modes:

Normal Chatbot (generic GPT-like)

Policy-Based Only (answers strictly using uploaded policy text)

All chat history is saved in a database & can be reopened later.

A. Install Dependencies
pip install -r requirements.txt


Recommended PDF font support (Linux):

sudo apt-get install fonts-noto-cjk

B. Environment Variables (.env)

Create .env with:

AZURE_OPENAI_API_KEY=xxxx
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-model-name
EMAIL_USER=your@gmail.com
EMAIL_PASSWORD=your_app_password
NOTIFY_EMAIL_TO=recipient@example.com

C. Run CLI Version (main.py)
python main.py


Example usage:

[INFO] Reading PDF...
[INFO] Generating JSON structured summary...
[INFO] Generating English summary...
[INFO] Translating (Hindi)...
PDF saved → policy_summary.pdf
Do you want to send this PDF by email? (yes/no):


Arguments available:

python main.py --pdf HealthPolicy.pdf --length 150 --lang hindi

D. Run Streamlit Dashboard
streamlit run streamlit_app.py


Visit in browser:

http://localhost:8501


Pages (Tabs):

Tab	Description
🧾 Summarize Policy	Upload PDF → Summary + PDF + Email
📚 Master Report	Upload 10+ PDFs → Consolidated Report
📜 History	View all generated reports
📊 Analytics	Chart view of coverage & exclusions
🤖 Chatbot	Ask questions about policy contents
3️ Example Output – JSON Summary
{
  "coverage": ["Hospitalization", "Room Rent", "Pre-Post Care"],
  "exclusions": ["Pre-existing diseases", "Dental cosmetic care"],
  "limits": ["Max 5L per year"],
  "eligibility": ["Age 18-60"],
  "waiting_periods": ["30 days general", "2 years maternity"],
  "disclaimers": ["No guarantee of policy renewal"]
}

4 Folder Structure
│── main.py                        # CLI summarizer + email
│── streamlit_app.py               # Web dashboard UI
│── policy_history.db              # Generated policy history
│── chat_history.db                # Chat conversations
│── requirements.txt
│── .env
│── README.md
│── /MASTER_POLICY_REPORT.pdf      # Generated (optional)
│── /<file>_summary.pdf            # Auto-generated report(s)

5 Analytics Capabilities

Extracts JSON → builds charts:

Most common coverage terms

Most frequent exclusions

Date-wise usage tracking

Uses: Plotly Express

6 Future Enhancements

Add OCR for scanned PDF policy images

Add JWT admin login & session auth

FASTAPI REST endpoints for automation pipelines

Export JSON data to Blob Storage or Salesforce Insurance Cloud

Integrate GPT-RAG document search across multiple policies

✨ Developed By

Nikita Pachkate
Insurance AI Engineer & Data Scientist
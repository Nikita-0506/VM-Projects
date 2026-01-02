# CLAIMS DESCRIPTION NORMALIZER – AI INSURANCE ENGINE

A production-ready AI system that converts raw insurance claim text into structured JSON, identifies loss type, severity, affected asset, evaluates fraud probability, and auto-generates claim summaries + PDF reports.
Includes ML fallback classifier, rule-based engines, YOLO-based image damage analysis, voice-to-text claim intake, history dashboard, and email PDF delivery.


## 1. Overview

Insurance claims often arrive as messy natural-language text, voice calls, or damaged-vehicle photos. Manual review is slow and error-prone.

#### This solution provides:

| Capability           | Description                                            |
| -------------------- | ------------------------------------------------------ |
| NLP Text Parsing     | Detects loss type, severity, asset affected            |
| Entity Extraction    | Extracts policy number and generates unique Claim IDs  |
| Language Detection   | Auto-translates Hindi / Marathi → English              |
| Fraud Scoring        | AI-based risk score & severity intelligence            |
| ML Fallback          | Logistic Regression classifier when GPT is unavailable |
| YOLO Image Damage AI | Detects vehicle count, auto-infers severity            |
| Voice Claims         | Speech transcription → Claim parser                    |
| PDF Generator        | Enterprise-grade formatted PDF claim report            |
| Email Delivery       | Sends PDF to customer via Gmail                        |
| Database Storage     | Logs history into SQLite                               |
| Analytics Dashboard  | Visual charts (severity, fraud, daily count)           |

## 2. System Architecture
```text
+----------------------+          +-------------------------+
|  User Input (Text)   | -------> | Azure OpenAI GPT Agent  |
+----------------------+          +-------------------------+
             | Fallback if fails
             v
+----------------------+          +-------------------------+
| ML Classifier (TF-IDF| -------> | Rule Engine (Regex /   |
| + Logistic Regression)|          | Loss & Asset Rules)    |
+----------------------+          +-------------------------+

Voice --------> Speech-to-Text ---------------+
                                              |
Images -------> YOLOv8 Car Detection ---------+
                                              v
                               Final JSON Normalization
                                              |
                                              v
                                  Fraud / Severity Scoring
                                              |
                                              v
                                  PDF Builder + Email Sender
                                              |
                                              v
                                     SQLite History + Analytics

```
## 3. Project Structure
```text
project/
├── main.py               # Terminal CLI normalizer
├── streamlit.py          # Web UI with OCR, Image, Voice features
├── claims.db             # Local DB (auto-created)
├── .env                  # API Keys + Email creds
├── README.md             # Documentation
```
## 4. Requirements

A. System Dependencies

   a. Python 3.9+

   b. GPU optional (for YOLO acceleration)

B. Python Libraries

   a.  pip install streamlit openai python-dotenv sklearn langdetect reportlab ultralytics
   b.  pip install pillow numpy yagmail speechrecognition pydub opencv-python

## 5. Environment Configuration (.env)

### Create a .env file at project root:

AZURE_OPENAI_API_KEY=xxxxxxxx
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_ENDPOINT=https://xxxx.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-model-name

EMAIL_USER=yourgmail@gmail.com
EMAIL_PASSWORD=your-app-password

## 6. Running the Application
A. CLI Mode (main.py)

python main.py

 a. Enter Claim Text: My car hit a bike yesterday, bumper broken. Policy 9043321

 b .Output:

   1. JSON result

   2. Human-readable summary

B. Streamlit Web Portal
   
   1. streamlit run streamlit.py


   2. UI will open with 5 pages:

       a. Normalize Claim

       b. Image Claim (YOLO damage detection)

       c. Voice Claim Intake

       d.  History Log

       e. Analytics Dashboard

## 7. Demo Example

   Input Text

      Kal mera car accident hua bike se takkar. Bumper damage. Policy number 9938833


   Output JSON

      {
        "loss_type": "Accident",
        "severity": "Medium",
        "affected_asset": "Car",
        "fraud": "Low",
        "confidence": 0.72
      }

    Generated Action
        
        Tow vehicle immediately OR Start repair process

## 8. PDF Report Output

### Automated enterprise-layout PDF contains:

   a. Claim ID + Policy

   b. AI language notes

   c. Incident details

   d. Fraud Level

   e. Recommended Next Action

   f. Summary Text

### Example PDF name:

POL-9938833-CLM-A73E.pdf

## 9. Email Sending

   a. PDF is attached and emailed automatically:

   b. Subject: Insurance Claim Report – POL-xxxx-CLM-yyyy
       Customer receives PDF report

## 10. Database Logging (SQLite)

### All processed claims are stored:

   a. claim_id, policy_no, summary, fraud_score, severity, asset, created_on


   b. History screen displays full log table.

## 11. Analytics Dashboard

### Visual charts:

   a. Claims count per day

   b. Fraud-level distribution

   c. Severity distribution

## 12. Use Cases

### Industry Use Cases

| Industry           | Use Case                                   |
|--------------------|---------------------------------------------|
| Auto Insurance     | Accident claims automation                  |
| Property Insurance | Fire / theft damage documentation           |
| BPO Call-Centers   | Voice-based intake + transcription          |
| Inspection Teams   | Field photo upload → instant AI assessment  |
| Fraud Control      | Auto-risk-flagging pipeline                 |


## 13. Future Enhancements 

   a. RAG knowledge-base policy validation

   b. Multi-language voice call ingestion (Telugu, Tamil)

   c. OCR extraction from documents/invoices

   d. Integration with CRM systems (Salesforce / Guidewire)

## ✨ Developed By:

**Nikita Pachkate**  
Data Scientist & AI Project Developer

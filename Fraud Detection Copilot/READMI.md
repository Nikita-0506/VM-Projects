# 🛡️ Fraud Detection Copilot

AI-powered claims fraud scoring & analytics system that combines Machine Learning (XGBoost) + Rule-based risk logic + Text-similarity AI to detect high-risk fraudulent claims.
Includes CLI automation, Streamlit dashboard, PDF fraud summary report generator, SQLite storage, and Email alerts for high-risk predictions.

## This project demonstrates:

   a. ML Model Training (XGBoost + SMOTE balancing)

   b. Hybrid Risk Scoring (ML + rule engine + text similarity)

   c. TF-IDF text vectorization for suspicious claim description detection

   d. Model persistence (joblib)

   e. Streamlit Analytics Dashboard

   f. SQLite logging of uploads & results

   g. Auto PDF generation

   h. Auto Email notification workflow
   

## 🧠 1️ Architecture Overview

#### Component Technology Stack

| Component                 | Technology                                      |
|---------------------------|--------------------------------------------------|
| ML Model                  | XGBoost Classifier                               |
| Vectorization             | TF-IDF + Scikit-Learn ColumnTransformer          |
| Oversampling              | SMOTE                                            |
| Backend Training Script   | Python (main.py)                                 |
| UI                        | Streamlit                                        |
| Storage                   | SQLite (fraud_results.db)                        |
| Email Alerts              | Yagmail (SMTP)                                   |
| Reporting                 | ReportLab PDF Generator                          |
| Deployment                | Local, Docker, Cloud VM                          |



## 🚀 2️ Core Features

### ✔️ Machine-Learning Fraud Scoring (XGBoost)

   a. Predicts fraud probability per claim

   b. Automatic threshold selection

   c. Stores probabilities + model prediction outputs
   

## 🧩 Hybrid Rule-Based Risk Enhancer

#### Adds extra risk % based on:

   a. repair_estimate > 2 × claim_amount

   b. previous_claims >= 3

   c. high text similarity against historical fraud descriptions

## 🧬 Text Similarity Engine

Cosine-similarity using TF-IDF embeddings

## 📑 PDF Fraud Summary Report

   a. Includes high-risk flagged claim list

   b. Auto-formatted PDF saved locally

   c. One-click download or send-to-email

## 📧 Email Alerts

When high-risk claims detected → prompts option to send summary to client/team

## 🧾 Streamlit Dashboard

   a. Upload CSV → instant risk scoring

   b. Download PDF

   c. Send Email Report

   d. Past uploads log

   e. Trend analytics & pie chart breakdown

## 🛠 A. Install Dependencies

pip install -r requirements.txt


### Additional recommended installs:

pip install xgboost imbalanced-learn joblib yagmail reportlab streamlit seaborn matplotlib

## 🔐 B. Environment Variables (.env)

Create .env in root:

EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
NOTIFY_EMAIL_TO=management@example.com  

## 🧪 C. Train Model & Run ML (CLI Mode)

python main.py


### CLI Output Example:
```text
=== XGBoost Model Results ===
ROC-AUC: 0.93
F1 Score: 0.87
Confusion Matrix:
[[180  12]
 [ 20  60]]
=== Predictions Saved to fraud_model_predictions.csv ===
=== Results Saved to SQLite fraud_results.db ===
=== PDF Generated: fraud_report.pdf ===
*** ALERT: High-risk fraud cases detected ***
Do you want to email this PDF fraud report? (y/n):
```

#### PDF file generated:

fraud_report.pdf


#### SQLite result stored:

fraud_results.db

## 🖥 D. Run Web App (Streamlit UI)

streamlit run app.py


#### Visit Dashboard:

http://localhost:8501

### UI Tabs

#### Application Tabs

| Tab                    | Description                                      |
|------------------------|--------------------------------------------------|
| 🚀 Run Fraud Detection | Upload CSV → Score claims → PDF / Email          |
| 📂 Upload History     | View stored uploads from SQLite                  |
| 📊 Analytics Dashboard | Trend line graph, pie-chart, category breakdown |



## 📊 Example Output CSV (Model Prediction)
```text
claim_amount,repair_estimate,previous_claims,days_since_last_claim,description,fraud_probability,risk
20000,50000,3,12,"rear bumper hit, no witnesses",0.84,"High"
```

## 📁 2 Folder Structure
```text
project/
│── main.py                         # ML training + CLI + PDF email sender
│── app.py                          # Streamlit dashboard
│── fraud_model.joblib              # Saved XGB model
│── preprocessor.joblib             # Saved preprocessing pipeline
│── fraud_results.db                # SQLite database
│── fraud_report.pdf                # Generated PDF report (auto-created)
│── fraud_model_predictions.csv     # Output predictions
│── requirements.txt
│── .env
│── README.md
```
## 📊 Analytics Features

Uses Streamlit + Matplotlib + SQLite:

   a. High-risk trend over time

   b. Pie chart breakdown

   c. Avg high-risk per dataset

   d. Upload total counts

## 💡 Suggested Enhancements (Future Scope)

#### Future Ideas & Business Value

| Idea                              | Value                                             |
|-----------------------------------|---------------------------------------------------|
| Add FastAPI inference API         | Real-time risk scoring endpoint                   |
| Add incremental retraining pipeline | Improve accuracy over time                        |
| Deploy on Azure / AWS Lambda      | Serverless risk alerts                            |
| Add Explainability (SHAP)         | Let underwriting teams understand model reasoning |
| Real-time Kafka ingestion         | Fraud alert automation in production              |



## ✨ Developed By

#### Nikita Pachkate

AI & Data Science Engineer – Insurance Intelligence Solutions

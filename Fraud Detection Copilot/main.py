# ============================
# Fraud Detection Copilot 
# ============================

import pandas as pd
import numpy as np
import sqlite3
import yagmail
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import SMOTE
from joblib import dump
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

# ======================
# Load ENV
# ======================
load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASSWORD")
NOTIFY_TO = os.getenv("NOTIFY_EMAIL_TO")

# ======================
# Load Dataset
# ======================
df = pd.read_csv("fraud_claims_dataset.csv")

# ======================
# Feature Split
# ======================
X = df.drop("fraud_reported", axis=1)
y = df["fraud_reported"]

text_feature = "description"
numeric_features = ["claim_amount", "repair_estimate", "previous_claims", "days_since_last_claim"]
categorical_features = ["claim_type", "location"]

# ======================
# Preprocessing Pipelines
# ======================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('txt', TfidfVectorizer(), text_feature)
    ]
)

numeric_pre = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# ======================
# Train-Test Split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Transform numeric+categorical only (for XGBoost prediction)
X_train_num = numeric_pre.fit_transform(X_train)
X_test_num = numeric_pre.transform(X_test)

# ======================
# Handle Imbalance
# ======================
X_train_enc = preprocessor.fit_transform(X_train)
sm = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train_enc, y_train)

# ======================
# Train XGBoost/ML Model 
# ======================
xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight = (len(y_train)/sum(y_train))  # class weight auto-adjust
)

xgb_model.fit(X_train_balanced, y_train_balanced)

# ML Prediction + Threshold Optimization
X_test_enc = preprocessor.transform(X_test)
raw_probs = xgb_model.predict_proba(X_test_enc)[:,1]

best_threshold = 0.30
preds = (raw_probs >= best_threshold).astype(int)

print("\n=== XGBoost Model Results ===")
print(classification_report(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, raw_probs))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
print("F1 Score:", f1_score(y_test, preds))

# Save model
dump(xgb_model, "fraud_model.joblib")
dump(preprocessor, "preprocessor.joblib")

# ======================
# Text Similarity
# ======================
fraud_texts = df[df["fraud_reported"] == 1][text_feature].astype(str).tolist()
tfidf = TfidfVectorizer().fit(fraud_texts)
fraud_vecs = tfidf.transform(fraud_texts)

def text_similarity_score(text):
    vec = tfidf.transform([text])
    sim = cosine_similarity(vec, fraud_vecs)
    return np.max(sim) if len(sim) else 0

# ======================
# Hybrid Rule + ML Risk Logic
# ======================
def hybrid_risk(row, ml_prob):
    rule_score = 0
    if row["repair_estimate"] > 2 * row["claim_amount"]:
        rule_score += 0.25
    if row["previous_claims"] >= 3:
        rule_score += 0.25
    if text_similarity_score(str(row[text_feature])) > 0.4:
        rule_score += 0.25
    return max(ml_prob, rule_score)

ml_probs = raw_probs
X_test_copy = X_test.copy()
X_test_copy["fraud_probability_ml"] = ml_probs
X_test_copy["fraud_probability_final"] = [
    hybrid_risk(X_test_copy.iloc[i], ml_probs[i]) for i in range(len(X_test_copy))
]

def categorize(prob):
    if prob >= 0.7: return "High"
    if prob >= 0.4: return "Medium"
    return "Low"

X_test_copy["risk"] = X_test_copy["fraud_probability_final"].apply(categorize)
X_test_copy.to_csv("fraud_model_predictions.csv", index=False)
print("=== Predictions Saved to fraud_model_predictions.csv ===")

# ======================
# Save Results to SQLite DB
# ======================
conn = sqlite3.connect("fraud_results.db")
X_test_copy.to_sql("fraud_results", conn, if_exists="replace", index=False)
conn.close()
print("=== Results Saved to SQLite fraud_results.db ===")

# ======================
# Generate Combined PDF Summary
# ======================
high_risk = X_test_copy[X_test_copy["risk"] == "High"]

def generate_pdf():
    file_name = "fraud_report.pdf"
    c = canvas.Canvas(file_name, pagesize=letter)
    width, height = letter
    y = height - 40

    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, y, "Fraud Detection Summary Report")
    y -= 30

    c.setFont("Helvetica", 10)
    c.drawString(30, y, f"Total Claims Tested: {len(X_test_copy)}")
    y -= 15
    c.drawString(30, y, f"High-Risk Claims: {len(high_risk)}")
    y -= 25

    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, y, "High-Risk Claims:")
    y -= 20

    c.setFont("Helvetica", 9)
    for idx, row in high_risk.iterrows():
        data = f"ID: {row.name} | Risk: {row['risk']} | Prob: {round(row['fraud_probability_final'],2)}"
        c.drawString(30, y, data)
        y -= 12
        if y < 30:
            c.showPage()
            y = height - 40

    c.save()
    return file_name

pdf_file = generate_pdf()
print("=== PDF Generated:", pdf_file, "===")

# ======================
# EMAIL SEND 
# ======================
if len(high_risk) > 0:
    print("\n*** ALERT: High-risk fraud cases detected ***")
    choice = input("Do you want to email this PDF fraud report? (y/n): ").strip().lower()

    if choice == "y":
        # Ask user for recipient email
        receiver = input("Enter recipient email address: ").strip()
        if receiver == "":
            print("No email entered. Email cancelled.")
        else:
            print("\n=== Sending Email with Fraud Report… ===")
            yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASS)

            email_body = f"""
Dear Client / Team,

Please find attached the latest Fraud Detection Summary Report generated by the Fraud Detection Copilot System.

Report includes:
• Total claims analyzed: {len(X_test_copy)}
• High-risk fraud-flagged cases: {len(high_risk)}
• ML probability + rule-based fraud score
• Suggested actions for manual review

This report is auto-generated using AI-based scoring + hybrid risk analysis.
If you have any questions or need deeper analytics dashboards, feel free to contact the Data Science team.

Regards,
Fraud Detection Copilot System
(Automated Email)
"""

            yag.send(
                to=receiver,
                subject="Fraud Detection – High-Risk Claims Summary (Auto-Generated Report)",
                contents=email_body,
                attachments=pdf_file
            )
            print("=== Email Sent Successfully ===")
    else:
        print("Email was NOT sent.")
else:
    print("No high-risk records → email send option skipped.")

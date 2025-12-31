# ============================
# Fraud Detection Copilot – Streamlit App (PRO Version)
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import yagmail
from datetime import datetime
from joblib import load
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Load ENV
load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASSWORD")

# Load ML model & preprocessor
model = load("fraud_model.joblib")
preprocessor = load("preprocessor.joblib")

# UI setup
st.set_page_config(page_title="Fraud Detection Copilot", layout="wide")
st.markdown("<h1 style='text-align:center;'>🛡️ Fraud Detection Copilot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered claim risk scoring, reporting, storage & analytics</p>", unsafe_allow_html=True)

# Top Navigation Tabs
tabs = st.tabs(["🚀 Run Fraud Detection", "📂 Upload History", "📊 Analytics Dashboard"])

# ======================================
# DATABASE FUNCTIONS
# ======================================
def init_db():
    conn = sqlite3.connect("fraud_results.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS uploads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    uploaded_at TEXT,
                    record_count INTEGER)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS fraud_results_history (
                    upload_id INTEGER,
                    record_json TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS pdf_reports (
                    upload_id INTEGER,
                    file_name TEXT,
                    created_at TEXT)""")
    conn.close()

def save_upload(filename, count):
    conn = sqlite3.connect("fraud_results.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO uploads (filename, uploaded_at, record_count) VALUES (?,?,?)",
                   (filename, datetime.now().strftime("%Y-%m-%d %H:%M"), count))
    upload_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return upload_id

def save_result(upload_id, df):
    conn = sqlite3.connect("fraud_results.db")
    conn.execute("INSERT INTO fraud_results_history (upload_id, record_json) VALUES (?,?)",
                 (upload_id, df.to_json()))
    conn.commit()
    conn.close()

def save_pdf(upload_id, file):
    conn = sqlite3.connect("fraud_results.db")
    conn.execute("INSERT INTO pdf_reports (upload_id, file_name, created_at) VALUES (?,?,?)",
                 (upload_id, file, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    conn.close()

# Initialize DB
init_db()

# ============================
# GLOBAL PDF GENERATOR 
# ============================
def generate_pdf(df, upload_id):
    file_name = f"fraud_report_{upload_id}.pdf"
    c = canvas.Canvas(file_name, pagesize=letter)
    width, height = letter
    y = height - 40

    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, y, "Fraud Detection Summary Report")
    y -= 25
    c.setFont("Helvetica", 10)
    c.drawString(30, y, f"Total Claims: {len(df)}")
    y -= 15
    c.drawString(30, y, f"High-Risk Flags: {len(df[df['risk']=='High'])}")
    y -= 20

    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, y, "High-Risk Records:")
    y -= 20

    c.setFont("Helvetica", 9)
    for idx, row in df[df["risk"]=="High"].iterrows():
        c.drawString(30, y, f"ID: {idx} | Prob: {round(row['fraud_probability'],2)}")
        y -= 12
        if y < 30:
            c.showPage()
            y = height - 40

    c.save()
    return file_name

# ======================================
# TAB 1 – RUN FRAUD DETECTION
# ======================================
with tabs[0]:
    st.subheader("🚀 Upload Claims File & Run AI Fraud Risk Analysis")

    colA, colB = st.columns([2, 1])
    uploaded_file = colA.file_uploader("Upload CSV File", type=["csv"])
    email_id = colB.text_input("Send report to email (optional):")

    run_btn = st.button("Run Analysis", use_container_width=True)

    if run_btn and uploaded_file:
        df = pd.read_csv(uploaded_file)
        enc = preprocessor.transform(df)
        probs = model.predict_proba(enc)[:, 1]
        df["fraud_probability"] = probs
        df["risk"] = df["fraud_probability"].apply(lambda p: "High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low")

        st.success("✔ Analysis Complete")
        st.dataframe(df, use_container_width=True)

        upload_id = save_upload(uploaded_file.name, len(df))
        save_result(upload_id, df)

        # ======================
        # REPORT OPTIONS
        # ======================
        st.markdown("### 📑 Report Options")

        # Save pdf in session
        if "last_pdf" not in st.session_state:
            st.session_state.last_pdf = None

        colPDF, colEmail = st.columns([1, 1])

        # Generate PDF
        if colPDF.button("📄 Generate PDF Report", use_container_width=True, key="genpdf"):
            pdf = generate_pdf(df, upload_id)
            st.session_state.last_pdf = pdf
            save_pdf(upload_id, pdf)
            st.success("✔ PDF successfully generated and stored")

        # Download PDF
        if st.session_state.last_pdf:
            with open(st.session_state.last_pdf, "rb") as f:
                st.download_button("📥 Download PDF", f, file_name=st.session_state.last_pdf,
                                   mime="application/pdf", use_container_width=True)

        # Email PDF
        if colEmail.button("📧 Send Email Report", use_container_width=True, key="sendmail"):
            if not email_id:
                st.error("⚠ Enter email address first.")
            else:
                if not st.session_state.last_pdf:
                    pdf = generate_pdf(df, upload_id)
                    st.session_state.last_pdf = pdf

                yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASS)
                body = f"""
Hello Team,

Your AI-powered Fraud Detection Report is ready.

Summary:
• Total Claims: {len(df)}
• High-Risk: {len(df[df['risk']=='High'])}
• Medium-Risk: {len(df[df['risk']=='Medium'])}
• Low-Risk: {len(df[df['risk']=='Low'])}

Attached is the full PDF with detailed findings.

Login to Fraud Detection Dashboard for trend analytics & history.

Regards,  
Fraud Detection Copilot – Automated AI System
"""
                yag.send(
                    to=email_id,
                    subject="Fraud Detection Report – High Risk Alerts Found",
                    contents=body,
                    attachments=st.session_state.last_pdf
                )
                st.success(f"📩 Email successfully sent to {email_id}!")

# ======================================
# TAB 2 – HISTORY VIEW
# ======================================
with tabs[1]:
    st.subheader("📂 Upload File History")
    conn = sqlite3.connect("fraud_results.db")
    df_hist = pd.read_sql_query("SELECT * FROM uploads ORDER BY id DESC", conn)
    conn.close()

    if df_hist.empty:
        st.info("No uploaded history found.")
    else:
        st.dataframe(df_hist, use_container_width=True)
        sel = st.selectbox("Select Upload ID", df_hist["id"])
        if sel:
            conn = sqlite3.connect("fraud_results.db")
            result_json = conn.execute(f"SELECT record_json FROM fraud_results_history WHERE upload_id={sel}").fetchone()
            conn.close()
            st.write("### Records:")
            st.dataframe(pd.read_json(result_json[0]), use_container_width=True)

# ======================================
# TAB 3 – ANALYTICS
# ======================================
with tabs[2]:
    st.subheader("📊 Analytics – Easy to Understand Summary")

    conn = sqlite3.connect("fraud_results.db")
    uploads = pd.read_sql_query("SELECT * FROM uploads ORDER BY uploaded_at", conn)
    conn.close()

    if uploads.empty:
        st.info("No analytics available yet.")
    else:
        trend = uploads.copy()
        high, med, low = [], [], []
        for uid in trend["id"]:
            conn = sqlite3.connect("fraud_results.db")
            r = conn.execute(f"SELECT record_json FROM fraud_results_history WHERE upload_id={uid}").fetchone()
            conn.close()
            df = pd.read_json(r[0])
            high.append(len(df[df["risk"]=="High"]))
            med.append(len(df[df["risk"]=="Medium"]))
            low.append(len(df[df["risk"]=="Low"]))

        trend["High"] = high
        trend["Medium"] = med
        trend["Low"] = low

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Uploads", len(trend))
        c2.metric("High-Risk Alerts", sum(high))
        c3.metric("Avg High-Risk per Upload", round(sum(high)/len(trend),2))

        st.markdown("#### 📈 Fraud Trend Over Time")
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(trend["uploaded_at"], trend["High"], marker="o", color="red")
        ax.set_xticklabels(trend["uploaded_at"], rotation=45)
        st.pyplot(fig)

        st.markdown("#### 📊 Category Breakdown")
        fig, ax = plt.subplots(figsize=(8,4))
        width = 0.25
        x = np.arange(len(trend))
        ax.bar(x-width, trend["High"], width, label="High", color="red")
        ax.bar(x, trend["Medium"], width, label="Medium", color="orange")
        ax.bar(x+width, trend["Low"], width, label="Low", color="green")
        ax.set_xticks(x)
        ax.set_xticklabels(trend["uploaded_at"], rotation=45)
        ax.legend()
        st.pyplot(fig)

        st.markdown("#### 🎯 Overall Percentage")
        fig, ax = plt.subplots()
        ax.pie([sum(high), sum(med), sum(low)], labels=["High","Medium","Low"], autopct="%1.1f%%")
        ax.axis("equal")
        st.pyplot(fig)

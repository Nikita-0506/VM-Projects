import os
import pdfplumber
import sqlite3
import json
import yagmail
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from fpdf import FPDF
from openai import AzureOpenAI

from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
# ======================================
# LOAD ENVIRONMENT + CLIENT
# ======================================
load_dotenv()
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-01"
)
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("NOTIFY_EMAIL_TO")

# ======================================
# DATABASE FILES
# ======================================
DB = "policy_history.db"
CHAT_DB = "chat_history.db"

# Create Chat Database Tables
def init_chat_tables():
    conn = sqlite3.connect(CHAT_DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chats(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            user TEXT,
            assistant TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            summary TEXT,
            json_summary TEXT,
            hindi TEXT,
            marathi TEXT,
            pdf_file TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()
init_chat_tables()

def create_chat(title="Chat"):
    conn = sqlite3.connect(CHAT_DB)
    cur = conn.cursor()
    cur.execute("INSERT INTO chats(title, created_at) VALUES (?,?)",
                (title, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    chat_id = cur.lastrowid
    conn.close()
    return chat_id

def save_chat_message(chat_id, user_msg, assistant_msg):
    conn = sqlite3.connect(CHAT_DB)
    cur = conn.cursor()
    cur.execute("INSERT INTO chat_messages(chat_id, user, assistant, created_at) VALUES (?,?,?,?)",
                (chat_id, user_msg, assistant_msg, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def fetch_chats():
    conn = sqlite3.connect(CHAT_DB)
    cur = conn.cursor()
    cur.execute("SELECT id, title FROM chats ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows

def fetch_chat_messages(chat_id):
    conn = sqlite3.connect(CHAT_DB)
    cur = conn.cursor()
    cur.execute("SELECT user, assistant FROM chat_messages WHERE chat_id=? ORDER BY id ASC", (chat_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

# ======================================
# DATABASE
# ======================================
def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            summary TEXT,
            json_summary TEXT,
            hindi TEXT,
            marathi TEXT,
            pdf_file TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(filename, summary, j, h, m, pdf_name):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO history(filename, summary, json_summary, hindi, marathi, pdf_file, created_at) VALUES (?,?,?,?,?,?,?)",
        (filename, summary, j, h, m, pdf_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

def fetch_history():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT * FROM history ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows

# ======================================
# EMAIL SEND FUNCTION (HEALTHY + SAFE)
# ======================================
def send_email(pdf_path, email_to=None):
    try:
        if email_to is None or email_to.strip() == "":
            email_to = EMAIL_TO  # default email from .env

        yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASSWORD)
        yag.send(
            to=email_to,
            subject="Policy Summary Document – AI Generated",
            contents="Hello,\n\nPlease find attached the generated insurance policy summary.\n\nRegards,\nAI Policy Assistant",
            attachments=[pdf_path]
        )
        return True
    except Exception as e:
        return str(e)

# ======================================
# EXTRACTION + GENAI FUNCTIONS
# ======================================
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text.strip()

def chunk_text(text, size=2000):
    return [text[i:i+size] for i in range(0, len(text), size)]

def call_llm(prompt):
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_json(text):
    prompt = f"""
    Convert the following into STRICT JSON:
    {{
      "coverage": [],
      "exclusions": [],
      "limits": [],
      "eligibility": [],
      "waiting_periods": [],
      "disclaimers": []
    }}

    Policy:
    {text}

    Reply JSON only.
    """
    return call_llm(prompt)

def english_summary(text, words=200):
    prompt = f"""
    Summarize the insurance policy in {words} words including coverage, exclusions, limits, disclaimers.
    Text:
    {text}
    """
    return call_llm(prompt)

def translate(text, lang):
    return call_llm(f"Translate into {lang}:\n{text}")

# ======================================
# CHAT_DB
# ======================================

CHAT_DB = "chat_history.db"

def init_chat_db():
    conn = sqlite3.connect(CHAT_DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            message TEXT,
            status TEXT,
            pdf_path TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


# Correct status-save function (used for email + save chat logs)
def save_chat_status(chat_id, message, status, pdf=None):
    conn = sqlite3.connect(CHAT_DB)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_history(chat_id, message, status, pdf_path, created_at) VALUES (?,?,?,?,?)",
        (chat_id, message, status, pdf, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(CHAT_DB)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_history(chat_id, message, status, pdf_path, created_at) VALUES (?,?,?,?,?)",
        (chat_id, message, status, pdf, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()


def fetch_chat_history():
    conn = sqlite3.connect(CHAT_DB)
    cur = conn.cursor()
    cur.execute("SELECT * FROM chat_history ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows

# ======================================
# PDF EXPORT
# ======================================

def generate_pdf(e, j, h, m, filename):
    pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))  # built-in Unicode font

    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    style = styles["Normal"]

    story = []
    story.append(Paragraph("=== ENGLISH SUMMARY ===", style))
    story.append(Paragraph(e.replace("\n","<br/>"), style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("=== JSON SUMMARY ===", style))
    story.append(Paragraph(j.replace("\n","<br/>"), style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("=== HINDI SUMMARY ===", style))
    story.append(Paragraph(h.replace("\n","<br/>"), style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("=== MARATHI SUMMARY ===", style))
    story.append(Paragraph(m.replace("\n","<br/>"), style))

    doc.build(story)
    return filename

# ======================================
# STREAMLIT CONFIG
# ======================================
st.set_page_config(page_title="Policy Summary Assistant", layout="wide")
init_db()

# ---------- FORCE HORIZONTAL TABS UI ----------
st.markdown("""
<style>
div[data-baseweb="tab-list"] {
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: nowrap !important;
}
div[data-baseweb="tab"] {
    padding: 12px 18px !important;
    border-radius: 10px !important;
    background-color: #111 !important;
    color: #fff !important;
    font-size: 16px !important;
}
div[data-baseweb="tab"][aria-selected="true"] {
    background-color: #4a4dda !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ============ SIDEBAR – Saved Chats ===============

if "active_chat" not in st.session_state:
    st.session_state.active_chat = create_chat("Chat #1")

with st.sidebar:
    st.header("📁 Saved Chats")

    chats = fetch_chats()
    for cid, title in chats:
        if st.button(title, key=f"open_chat_{cid}"):
            st.session_state.active_chat = cid

    if st.button("➕ New Chat"):
        new_id = create_chat(f"Chat #{len(chats)+1}")
        st.session_state.active_chat = new_id


# ============= UI – TOP TABS =====================
st.title("Insurance Policy Compliance AI – Dashboard")

tab_home, tab_master, tab_history, tab_analytics, tab_chat = st.tabs(
    ["🧾 Summarize Policy", "📚 Master Report", "📜 History", "📊 Analytics", "🤖 Chatbot"]
)

# ======================================
# TAB 1 – HOME
# ======================================
with tab_home:
    st.header("Policy Summary – AI Assistant")
    uploaded = st.file_uploader("Upload Multiple PDFs", type=["pdf"], accept_multiple_files=True)
    length = st.slider("English Summary Length", 100, 400, 200)
    lang_choice = st.radio(
        "Choose summary output language:",
        ["None (English Only)", "Hindi", "Marathi"],
        key="home_lang_choice"
    )

    if uploaded and st.button("Generate Summary"):
        for file in uploaded:
            text = extract_text_from_pdf(file)
            st.write("[INFO] Reading PDF...")
            chunks = chunk_text(text)
            selected = chunks[0]

            st.write("[INFO] Generating JSON structured summary...")
            j = generate_json(selected)

            st.subheader("=== STRUCTURED JSON SUMMARY ===")
            st.code(j, language="json")

            st.write("[INFO] Generating English summary...")
            e = english_summary(selected, length)

            # Apply chosen language
            if lang_choice == "Hindi":
                summary_out = translate(e, "Hindi")
            elif lang_choice == "Marathi":
                summary_out = translate(e, "Marathi")
            else:
                summary_out = e

            st.subheader("=== HUMAN SUMMARY ===")
            st.write(summary_out)

            # DB values
            h = translate(e, "Hindi") if lang_choice == "Hindi" else ""
            m = translate(e, "Marathi") if lang_choice == "Marathi" else ""

            pdf_name = f"{file.name}_summary.pdf"
            generate_pdf(e, j, h, m, pdf_name)
            save_to_db(file.name, e, j, h, m, pdf_name)

            st.success(f"Summary generated for {file.name}")

            # PDF Download
            with open(pdf_name, "rb") as f:
                st.download_button("Download PDF", f, file_name=pdf_name, key=f"dl_{file.name}")

            # ===== EMAIL SEND – FIXED UI =====
            email_choice = st.radio(
                f"Do you want to send email for {file.name}?",
                ["No", "Yes"],
                key=f"email_choice_{file.name}"
            )

            if email_choice == "Yes":
                with st.form(key=f"email_form_{file.name}"):
                    email_to = st.text_input("Enter recipient email", EMAIL_TO)
                    submit_email = st.form_submit_button("Send Email")
                    if submit_email:
                        send_email(pdf_name, email_to)
                        save_chat_status(st.session_state.active_chat, f"PDF sent to {email_to}", "Mail Sent")
                        st.success(f"Email sent successfully to: {email_to}")
            else:
                if st.button("Save Chat", key=f"save_only_{file.name}"):
                    save_chat_status(st.session_state.active_chat, f"PDF generated: {file.name}", "Saved")
                    st.info("Chat saved.")

# ======================================
# TAB 2 – MASTER REPORT
# ======================================
with tab_master:
    st.header("Master Combined Report (10+ PDFs)")

    lang_choice_m = st.radio(
        "Choose summary output language:",
        ["None (English Only)", "Hindi", "Marathi"],
        key="master_lang_choice"
    )

    uploaded_m = st.file_uploader("Upload 10+ PDFs", type=["pdf"], accept_multiple_files=True)
    length_m = st.slider("Summary Length", 100, 400, 200, key="len_master")

    if uploaded_m and len(uploaded_m) >= 10 and st.button("Create Master Report"):
        sections = []

        for file in uploaded_m:
            text = extract_text_from_pdf(file)
            chunks = chunk_text(text)
            selected = chunks[0]

            e = english_summary(selected, length_m)
            j = generate_json(selected)

            # Apply language
            if lang_choice_m == "Hindi":
                e = translate(e, "Hindi")
            elif lang_choice_m == "Marathi":
                e = translate(e, "Marathi")

            sections.append({"file": file.name, "english": e, "json": j})

        master_pdf = "MASTER_POLICY_REPORT.pdf"

        pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))
        doc = SimpleDocTemplate(master_pdf, pagesize=A4)
        styles = getSampleStyleSheet()
        style = styles["Normal"]

        story = []
        for sec in sections:
            story.append(Paragraph(f"===== {sec['file']} =====", style))
            story.append(Paragraph(sec["english"].replace("\n", "<br/>"), style))
            story.append(Spacer(1, 12))
            story.append(Paragraph("JSON:", style))
            story.append(Paragraph(sec["json"].replace("\n", "<br/>"), style))
            story.append(Spacer(1, 20))

        doc.build(story)

        st.success("Master Report Created Successfully!")
        with open(master_pdf, "rb") as f:
            st.download_button("Download Master PDF", f, file_name=master_pdf, key="dl_master")

        # ===== EMAIL SEND – FIXED UI =====
        email_choice2 = st.radio(
            "Do you want to send the Master Report by email?",
            ["No", "Yes"],
            key="master_email_choice"
        )

        if email_choice2 == "Yes":
            with st.form(key="master_email_form"):
                email_to2 = st.text_input("Enter recipient email", EMAIL_TO)
                submit2 = st.form_submit_button("Send Master Email")
                if submit2:
                    send_email(master_pdf, email_to2)
                    save_chat_status(st.session_state.active_chat, f"Master PDF emailed to {email_to2}", "Mail Sent")
                    st.success(f"Master Report sent successfully to: {email_to2}")
        else:
            if st.button("Save Master Chat Only", key="save_master_chat"):
                save_chat_status(st.session_state.active_chat, "Master Report Generated", "Saved")
                st.info("Master Report entry saved.")
    else:
        st.info("Upload minimum 10 PDFs to generate master report.")

# ======================================
# TAB 3 – HISTORY
# ======================================
with tab_history:
    st.header("History Log")
    rows = fetch_history()
    if rows:
        df = pd.DataFrame(rows, columns=["ID","File","Summary","JSON","Hindi","Marathi","PDF","Created"])
        st.dataframe(df[['File','Created','PDF']])
    else:
        st.info("No history found.")

# ======================================
# TAB 4 – ANALYTICS
# ======================================
with tab_analytics:
    st.header("Policy Analytics")
    rows = fetch_history()
    if rows:
        df = pd.DataFrame(rows, columns=["ID","File","Summary","JSON","Hindi","Marathi","PDF","Created"])
        coverage, exclusions = [], []
        for _, row in df.iterrows():
            try:
                j = json.loads(row["JSON"])
                coverage.extend(j.get("coverage", []))
                exclusions.extend(j.get("exclusions", []))
            except:
                pass
        cov_df = pd.DataFrame(coverage, columns=["Coverage"])
        exc_df = pd.DataFrame(exclusions, columns=["Exclusions"])
        if not cov_df.empty:
            chart = cov_df["Coverage"].value_counts().reset_index()
            chart.columns = ["Coverage","Count"]
            st.plotly_chart(px.bar(chart, x="Coverage", y="Count"))
        if not exc_df.empty:
            chart2 = exc_df["Exclusions"].value_counts().reset_index()
            chart2.columns = ["Exclusions","Count"]
            st.plotly_chart(px.bar(chart2, x="Exclusions", y="Count"))
    else:
        st.info("No data available.")

# ======================================
# TAB 5 – CHATBOT
# ======================================

chat_id = st.session_state.active_chat
for u,a in fetch_chat_messages(chat_id):
    st.write(f"User: {u}")
    st.write(f"Assistant: {a}")

with tab_chat:
    st.header("Chatbot – Ask Questions")
    uploaded = st.file_uploader("Upload Policy PDF", type=["pdf"])
    mode = st.radio("Chat Mode", ["Normal Chatbot", "Policy-Based Only"])
    if "chat" not in st.session_state:
        st.session_state.chat = []
    policy_text = extract_text_from_pdf(uploaded) if uploaded else ""
    msg = st.chat_input("Ask...")
    if msg:
      prompt = msg if mode == "Normal Chatbot" else f"Answer ONLY using this text:\n{policy_text}\n\nQ: {msg}"
      answer = call_llm(prompt)
      st.session_state.chat.append((msg, answer))
      save_chat_message(st.session_state.active_chat, msg, answer)

    for q, a in st.session_state.chat:
        st.write(f"User: {q}")
        st.write(f"Assistant: {a}")

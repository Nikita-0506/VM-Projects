# ===========================================
#  AGENTIC CLAIMS NORMALIZER
# ===========================================
import streamlit as st
import os, json, uuid, logging, sqlite3, tempfile, io, re, cv2
from email.message import EmailMessage
from dotenv import load_dotenv
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from openai import AzureOpenAI
from langdetect import detect
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from ultralytics import YOLO
from PIL import Image
import numpy as np
import speech_recognition as sr
import yagmail
import time
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment


# ===============================
# TEMP NOTIFICATION HELPER
# ===============================
def show_temp_message(message_fn, text, seconds=3):
    msg = message_fn(text)
    time.sleep(seconds)
    msg.empty()


# ✅ MUST be first Streamlit call
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

st.session_state["_sidebar_open"] = True

# 🔥 REMOVE EXTRA SPACE ABOVE TITLE
st.markdown("""
<style>

/* Remove ALL top padding/margin before content */
section.main {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Remove phantom container spacing */
section.main > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Remove block container spacing */
section.main .block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Remove app-level spacing */
[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Kill invisible vertical block gap */
[data-testid="stVerticalBlock"]:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* FORCE SIDEBAR VISIBILITY */
section[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    transform: none !important;
}

/* Restore layout balance */
section.main {
    margin-left: auto !important;
}

</style>
""", unsafe_allow_html=True)

# ===========================================
# ENV + OPENAI SETUP
# ===========================================
load_dotenv()
logging.basicConfig(level=logging.INFO)
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Gmail / Email Credentials (Required for Send Mail)
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")


# ===========================================
# DB INIT
# ===========================================
def init_db():
    conn = sqlite3.connect("claims.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS claims (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim_id TEXT,
        policy_number TEXT,
        summary TEXT,
        fraud TEXT,
        confidence REAL,
        loss_type TEXT,
        severity TEXT,
        asset TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()
init_db()

def save_history(json_out):
    conn = sqlite3.connect("claims.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO claims (claim_id, policy_number, summary, fraud, confidence, loss_type, severity, asset, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            json_out["claim_id"],
            json_out["policy_number"],
            json_out["summary"],
            json_out["ai_risk"]["fraud"],
            json_out["ai_risk"]["confidence_score"],
            json_out["incident"]["loss_type"],
            json_out["incident"]["severity"],
            json_out["incident"]["affected_asset"],
            json_out["normalized_on"]
        )
    )
    conn.commit()
    conn.close()

def fetch_history():
    conn = sqlite3.connect("claims.db")
    df = conn.execute("SELECT * FROM claims ORDER BY id DESC").fetchall()
    conn.close()
    return df

# ===========================================
#  Chatbot Memory
# ===========================================
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ===========================================
# ML FALLBACK
# ===========================================
texts = ["car accident","bike stolen","house fire","mobile damaged","flood water"]
labels = ["Accident","Theft","Fire","Damage","Flood"]
vec = TfidfVectorizer()
X = vec.fit_transform(texts)
clf = LogisticRegression().fit(X, labels)
def ml_predict(text): return clf.predict(vec.transform([text]))[0]

LOSS_RULES = {
    "accident":"Accident","hit":"Accident","crash":"Accident","collision":"Accident",
    "theft":"Theft","stolen":"Theft","robbed":"Theft",
    "fire":"Fire","burn":"Fire",
    "flood":"Flood","water":"Flood"
}
ASSET_RULES = {
    "car":"Car","vehicle":"Car","bike":"Bike","motorcycle":"Motorcycle",
    "house":"House","home":"House","mobile":"Mobile","phone":"Mobile"
}
def rule_based_loss(text):
    for k,v in LOSS_RULES.items():
        if k in text.lower(): return v
    return None
def rule_based_asset(text):
    for k,v in ASSET_RULES.items():
        if k in text.lower(): return v
    return None

# ===========================================
# GENERAL CHATBOT AGENT
# ===========================================
def chatbot_agent(user_query, history):

    # ✅ Fetch claim history for context
    claims_data = fetch_history()

    system_prompt = f"""
You are an AI assistant for an insurance platform.

User claim history (latest first):
{claims_data}

You can:
- Answer general questions
- Answer insurance & claim-related questions
- Summarize or explain claim history if asked
- Help with policies, claims, and app usage
- Answer unrelated questions normally (like ChatGPT)

Rules:
- Be clear and professional for insurance queries
- Be friendly for general questions
- If data is missing, say so honestly
"""

    messages = [{"role": "system", "content": system_prompt}]

    # Add chat memory
    for h in history[-10:]:
        messages.append(h)

    messages.append({"role": "user", "content": user_query})

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        temperature=0.6
    )

    return resp.choices[0].message.content

# ===========================================
# AI AGENTS
# ===========================================
def language_agent(text):
    lang = detect(text)
    if lang in ["hi","mr"]:
        resp = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role":"user","content":f"Translate Hindi/Marathi to English:\n{text}"}]
        )
        return resp.choices[0].message.content,"Translated"
    return text,"English"

def extraction_agent(text):
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{
            "role":"user",
            "content":f"Return STRICT JSON only: loss_type, severity, affected_asset\n{text}"
        }],
        temperature=0.1
    )
    return json.loads(resp.choices[0].message.content)

def severity_rule(text):
    t=text.lower(); score=0
    if any(k in t for k in ["small","minor","scratch"]): score+=1
    if any(k in t for k in ["broken","damaged","bumper"]): score+=2
    if any(k in t for k in ["fire","exploded","burned","total loss"]): score+=3
    if "stolen" in t or "theft" in t: return "Medium"
    return "High" if score>=3 else "Medium" if score==2 else "Low" if score==1 else "Medium"

def fraud_agent(text,severity):
    score=0
    if len(text)<40: score+=1
    if severity=="High": score+=2
    return "High" if score>=3 else "Medium" if score==2 else "Low"

def confidence_agent(data,text):
    score=0.3
    if data["loss_type"]!="Other": score+=0.25
    if data["affected_asset"]!="Other": score+=0.25
    if len(text.split())>10: score+=0.1
    if len(text.split())<5: score-=0.1
    return round(max(min(score,0.98),0.10),2)

# ===========================================
# NORMALIZER DAMAGE 
# ===========================================
def normalize_claim(text):
    fallback=False
    try: extracted=extraction_agent(text)
    except:
        extracted={"loss_type":ml_predict(text),"severity":"Medium","affected_asset":"Other"}
        fallback=True
    if rule_based_loss(text): extracted["loss_type"]=rule_based_loss(text)
    if rule_based_asset(text): extracted["affected_asset"]=rule_based_asset(text)
    extracted["severity"]=severity_rule(text)
    fraud=fraud_agent(text,extracted["severity"])
    conf=confidence_agent(extracted,text)
    policy=re.search(r"\b\d{5,12}\b",text)
    policy_no=policy.group(0) if policy else "UNKNOWN"
    claim_id=f"POL-{policy_no}-CLM-{uuid.uuid4().hex[:4]}"
    return extracted,fraud,conf,fallback,claim_id,policy_no

# ===========================================
# IMAGE DAMAGE
# ===========================================
damage_model=None
def get_yolo():
    global damage_model
    if damage_model is None: damage_model=YOLO("yolov8n.pt")
    return damage_model

def detect_damage(img:Image.Image):
    model=get_yolo()
    img_np=cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
    results=model.predict(img_np, imgsz=640)
    labels=[model.names[int(b.cls)] for b in results[0].boxes]
    car_count=labels.count("car")+labels.count("vehicle")
    severity="High" if car_count>=2 else "Medium" if car_count==1 else "Low"
    auto_text=f"A car accident involving {car_count} vehicles. Severity {severity}. Body damage visible."
    return auto_text
# ===========================================
# VIDEO DAMAGE 
# ===========================================
def extract_frames_from_video(video_file, every_n_sec=1):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp.write(video_file.read())
    temp.flush()

    cap = cv2.VideoCapture(temp.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * every_n_sec)

    frames = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        frame_id += 1

    cap.release()
    return frames


def analyze_video_damage(video_file):
    frames = extract_frames_from_video(video_file)
    severities = []

    for img in frames[:5]:  # limit to first 5 frames for speed
        auto_text = detect_damage(img)
        if "High" in auto_text:
            severities.append("High")
        elif "Medium" in auto_text:
            severities.append("Medium")
        else:
            severities.append("Low")

    if "High" in severities:
        final_severity = "High"
    elif "Medium" in severities:
        final_severity = "Medium"
    else:
        final_severity = "Low"

    summary_text = f"Video evidence shows vehicle damage with overall severity {final_severity}."
    return summary_text, final_severity

# ===========================================
# VOICE DAMAGE 
# ===========================================
def transcribe_audio(uploaded_file):
    temp=tempfile.NamedTemporaryFile(delete=False,suffix=".wav")
    temp.write(uploaded_file.read()); temp.flush()
    recog=sr.Recognizer()
    with sr.AudioFile(temp.name) as src:
        audio_data=recog.record(src)
        try: return recog.recognize_google(audio_data, language="hi-IN")
        except:
            try: return recog.recognize_google(audio_data, language="en-IN")
            except: return ""

# ===========================================
# PDF MAKER + PREVIEW
# ===========================================
def make_pdf(json_output,human_msg):
    buf=io.BytesIO()
    c=canvas.Canvas(buf,pagesize=A4)
    t=c.beginText(50,800)
    t.setFont("Helvetica",11)
    t.textLine("AI CLAIM REPORT")
    t.textLine("----------------------------------")
    for k,v in json_output.items():
        t.textLine(f"{k}: {v}")
    t.textLine("")
    t.textLine("HUMAN SUMMARY:")
    t.textLines(human_msg)
    c.drawText(t)
    c.save()
    buf.seek(0)
    return buf
# ===========================================
# ENTERPRISE PDF GENERATOR (Used in Normalize Claim)
# ===========================================
def make_enterprise_pdf(json_out, human_msg):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 50

    def write(x, txt, bold=False, size=11):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(x, y, txt)
        y -= 16

    # Header
    write(50, "INSURANCE CLAIM REPORT", bold=True, size=16)
    y -= 5
    c.line(50, y, width - 50, y)
    y -= 20

    # Metadata
    write(50, f"Claim ID: {json_out['claim_id']}")
    write(50, f"Policy Number: {json_out['policy_number']}")
    write(50, f"Processed On: {json_out['normalized_on']}")
    write(50, f"Language: {json_out.get('language', 'English')}")

    y -= 20
    write(50, "Incident Details", bold=True, size=13)
    write(60, f"Loss Type: {json_out['incident']['loss_type']}")
    write(60, f"Severity: {json_out['incident']['severity']}")
    write(60, f"Affected Asset: {json_out['incident']['affected_asset']}")

    y -= 20
    write(50, "AI Fraud Check", bold=True, size=13)
    write(60, f"Fraud Level: {json_out['ai_risk']['fraud']}")
    write(60, f"Confidence Score: {json_out['ai_risk']['confidence_score']}")

    y -= 20
    write(50, "Summary", bold=True, size=13)
    for line in json_out["summary"].split("\n"):
        write(60, line)

    y -= 20
    write(50, "Recommended Action", bold=True, size=13)
    action_text = json_out.get("recommended_action", "Further assessment required.")
    for line in action_text.split("\n"):
      write(60, line)


    c.save()
    buf.seek(0)
    return buf

# ===========================================
# ✅ COMMON NORMALIZED OUTPUT (ALL MODES)
# ===========================================
def render_claim_output(json_out, human_output, email=None):

    st.markdown("---")

    # LEFT: JSON | RIGHT: HUMAN
    col1, col2 = st.columns(2)

    with col1:
        st.json(json_out)

    with col2:
        st.text(human_output)

    # PDF GENERATION
    pdf = make_enterprise_pdf(json_out, human_output)
    pdf_path = os.path.join(
        tempfile.gettempdir(),
        f"{json_out['claim_id']}.pdf"
    )

    with open(pdf_path, "wb") as f:
        f.write(pdf.getvalue())

    # ACTION BUTTONS (exact like Normalize)
    st.markdown(f"[📄 View PDF Before Sending]({pdf_path})")

    st.download_button(
        "⬇ Download PDF",
        pdf,
        file_name=f"{json_out['claim_id']}.pdf"
    )

    send_to = email if email else EMAIL_USER

    if st.button(
    f"✉ Send Email – {json_out['claim_id']}",
    key=f"send_{json_out['claim_id']}"
):

     show_temp_message(st.warning, "Please upload a file first", 3)


    # ❌ NO AUTO SEND — only after click
     if not send_to:
      show_temp_message(st.error, "Email address not provided", 3)
      return


    with st.spinner("Sending email..."):
        yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASSWORD)
        yag.send(
            to=send_to,
            subject=f"Insurance Claim Report – {json_out['claim_id']}",
            contents=f"""
Dear Customer,

Your insurance claim has been successfully processed.

Claim ID: {json_out['claim_id']}
Loss Type: {json_out['incident']['loss_type']}
Severity: {json_out['incident']['severity']}
Asset: {json_out['incident']['affected_asset']}

PDF Report Attached.

Regards,
Insurance AI Bot
""",
            attachments=[pdf_path]
        )

    show_temp_message(st.success, "Email sent successfully!", 3)



    if st.button("🧹 Clear All Results"):
        st.session_state["history_outputs"] = []
        st.rerun()


def pdf_to_img(buffer):
    import base64
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:application/pdf;base64,{b64}"

# ---------- SESSION STATE INIT ----------
if "active_page" not in st.session_state:
    st.session_state.active_page = "Normalize Claim"

# ===========================================
#  MULTI-PAGE UI + SIDEBAR
# ===========================================

with st.sidebar:

    # =========================
    # SIDEBAR STYLING (UI ONLY)
    # =========================
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #020617);
        padding-top: 1.2rem;
    }

    .sb-title {
        font-size: 22px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.4rem;
        text-align: center;
    }

    .sb-subtitle {
        font-size: 14px;
        font-weight: 600;
        color: #cbd5e1;
        margin: 1rem 0 0.5rem 0;
    }

    .sb-divider {
        border-bottom: 1px solid #1e293b;
        margin: 1rem 0;
    }

    section[data-testid="stSidebar"] button {
        border-radius: 12px !important;
        font-size: 14px !important;
        padding: 0.45rem 0.75rem !important;
    }

    .stButton > button {
        background: linear-gradient(180deg, #1e293b, #020617);
        border: 1px solid #334155;
        color: white;
    }

    .stButton > button:hover {
        border-color: #38bdf8;
    }
    </style>
    """, unsafe_allow_html=True)

    # =========================
    # TITLE
    # =========================
    st.markdown('<div class="sb-title">Claims AI</div>', unsafe_allow_html=True)

    # =========================
    # LOAD CLAIMS
    # =========================
    claims = fetch_history()

    if "selected_claim_id" not in st.session_state:
        st.session_state.selected_claim_id = None

    if "history_outputs" not in st.session_state:
        st.session_state.history_outputs = []

    claim_to_delete = None

    # =========================
    # SAVED CLAIMS
    # =========================
    st.markdown('<div class="sb-subtitle">Saved Claims</div>', unsafe_allow_html=True)

    if claims:
        for row in claims[:20]:
            _, claim_id, _, _, _, _, _, severity, _, _ = row

            col_open, col_del = st.columns([5, 1])

            with col_open:
                if st.button(
                    f"{claim_id} · {severity}",
                    key=f"open_{claim_id}",
                    use_container_width=True
                ):
                    st.session_state.selected_claim_id = claim_id
                    st.session_state.active_page = "Normalize Claim"
                    st.rerun()

            with col_del:
                if st.button("✕", key=f"del_{claim_id}"):
                    claim_to_delete = claim_id
    else:
        st.caption("No claims saved yet.")

    # =========================
    # DIVIDER
    # =========================
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # =========================
    # ✅ NEW CLAIM (FULL RESET)
    # =========================
    if st.button("＋ New Claim", use_container_width=True):

        st.session_state.selected_claim_id = None
        st.session_state.claims_buffer = ""
        st.session_state.history_outputs = []
        st.session_state.active_page = "Normalize Claim"
        st.session_state["_new_claim"] = True  # UX flag

        st.rerun()

    # =========================
    # DELETE CLAIM
    # =========================
    if claim_to_delete:
        conn = sqlite3.connect("claims.db")
        conn.execute(
            "DELETE FROM claims WHERE claim_id = ?",
            (claim_to_delete,)
        )
        conn.commit()
        conn.close()
        st.rerun()

# ================================
# GLOBAL STICKY DASHBOARD BAR
# ================================

st.markdown("""
<style>

/* =========================================
   SAFE MAIN LAYOUT (DO NOT FORCE WIDTH)
   ========================================= */

/* Let Streamlit manage sidebar width */
section.main {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}

/* Main content container */
section.main .block-container {
    max-width: 100% !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    padding-top: 0rem !important;
}

/* Remove Streamlit internal centering + padding */
[data-testid="stAppViewContainer"] {
    max-width: 100vw !important;
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* =========================================
   STICKY TOP NAVIGATION
   ========================================= */

.global-nav {
    position: sticky;
    top: 0;
    z-index: 9999;
    background: linear-gradient(180deg, #0f0f0f, #0b0b0b);
    border-bottom: 1px solid #2a2a2a;
    padding: 12px 0 14px 0;
}

/* Inner container */
.global-nav-inner {
    max-width: 1400px;
    margin: auto;
}

/* Title */
.global-title {
    text-align: center;
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 10px;
}

/* Button spacing */
.global-nav-inner div[data-testid="column"] {
    padding: 6px;
}

/* Buttons */
.global-nav-inner button {
    height: 50px;
    width: 100%;
    border-radius: 14px;
    border: 1px solid #2f2f2f;
    background: linear-gradient(180deg, #171717, #0f0f0f);
    font-size: 15px;
    font-weight: 500;
    color: white;
    white-space: nowrap;
}

/* Hover */
.global-nav-inner button:hover {
    border-color: #4c4c4c;
    background: linear-gradient(180deg, #1d1d1d, #141414);
}

/* Remove phantom spacing below navbar */
section.main > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------- NAV HTML ----------
st.markdown("""
<style>
.hero-header {
    width: 100%;
    padding: 3rem 1rem;
    text-align: center;
    background: transparent;   /* ✅ removed blue/gradient box */
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.6rem;
    letter-spacing: 0.5px;
}

.hero-subtitle {
    font-size: 1.25rem;
    font-weight: 400;
    color: #cbd5e1;
    letter-spacing: 0.3px;
}
</style>

<div class="hero-header">
    <div class="hero-title">📊 ClaimStruct AI</div>
    <div class="hero-subtitle">Claims Description Normalization System</div>
</div>
""", unsafe_allow_html=True)


cols = st.columns(7)

with cols[0]:
    if st.button("📝 Normalize", use_container_width=True):
        st.session_state.active_page = "Normalize Claim"
        st.rerun()

with cols[1]:
    if st.button("🖼 Image", use_container_width=True):
        st.session_state.active_page = "Image Claim"
        st.rerun()

with cols[2]:
    if st.button("🎥 Video", use_container_width=True):
        st.session_state.active_page = "Video Claim"
        st.rerun()

with cols[3]:
    if st.button("🎙 Voice", use_container_width=True):
        st.session_state.active_page = "Voice Claim"
        st.rerun()

with cols[4]:
    if st.button("🤖 Chatbot", use_container_width=True):
        st.session_state.active_page = "Chatbot"
        st.rerun()

with cols[5]:
    if st.button("📜 History", use_container_width=True):
        st.session_state.active_page = "History"
        st.rerun()

with cols[6]:
    if st.button("📈 Analytics", use_container_width=True):
        st.session_state.active_page = "Analytics Dashboard"
        st.rerun()

st.markdown("""
  </div>
</div>
""", unsafe_allow_html=True)

# ===========================================
# PAGE – NORMALIZE CLAIM
# ===========================================
page = st.session_state.active_page

if page == "Normalize Claim":
    st.title("Text Claim Normalizer")

    # ===============================
    # SHOW SELECTED CLAIM (FROM SIDEBAR)
    # ===============================
    if st.session_state.get("selected_claim_id"):
        cid = st.session_state.selected_claim_id
        conn = sqlite3.connect("claims.db")
        row = conn.execute(
            "SELECT * FROM claims WHERE claim_id = ?", (cid,)
        ).fetchone()
        conn.close()

        if row:
            st.info(f"📄 Selected Claim: {cid}")
            st.json({
                "claim_id": row[1],
                "policy_number": row[2],
                "summary": row[3],
                "fraud": row[4],
                "confidence": row[5],
                "loss_type": row[6],
                "severity": row[7],
                "asset": row[8],
                "created_at": row[9],
            })

    # Init Session
    if "claims_buffer" not in st.session_state:
        st.session_state["claims_buffer"] = ""
    if "history_outputs" not in st.session_state:
        st.session_state["history_outputs"] = []

    # Input box (auto-reset)
    st.session_state["claims_buffer"] = st.text_area(
        "Enter Claim Text",
        value=st.session_state["claims_buffer"],
        key="claim_box",
        placeholder="Example: My car hit a bike yesterday, bumper damaged. Claim number 9938833"
    )
    email = st.text_input("Email To Send PDF (Optional)")

    # PROCESS CLAIM
    if st.button("Process Claim"):
        claim = st.session_state["claims_buffer"]
        if not claim.strip():
            st.warning("Enter claim text")
            st.stop()

        translated, lang = language_agent(claim)
        data, fraud, conf, fallback, cid, policy = normalize_claim(translated)

        summary = f"Claim involves {data['loss_type']} affecting {data['affected_asset']} severity {data['severity']}."
        action = "Tow vehicle immediately" if data["severity"] == "High" else "Start repair process"

        human_output = f"""
Claim ID: {cid}
Loss Type: {data['loss_type']}
Asset: {data['affected_asset']}
Severity: {data['severity']}
Fraud Risk: {fraud}
Confidence: {conf}

Summary:
{summary}

Recommended Action:
{action}
"""

        json_out = {
            "claim_id": cid,
            "policy_number": policy,
            "language": lang,
            "fallback_used": fallback,
            "incident": data,
            "ai_risk": {
                "fraud": fraud,
                "confidence_score": conf
            },
            "summary": summary,
            "recommended_action": action,
            "raw_input": claim,
            "normalized_on": datetime.utcnow().isoformat() + "Z"
        }

        save_history(json_out)
        st.session_state["history_outputs"].append((json_out, human_output))

        st.session_state["claims_buffer"] = ""
        st.rerun()

    # DISPLAY RESULTS
    for json_out, human_output in reversed(st.session_state["history_outputs"]):
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1: st.json(json_out)
        with col2: st.text(human_output)

        # Create PDF + Save Path
        pdf = make_enterprise_pdf(json_out, human_output)
        pdf_path = os.path.join(tempfile.gettempdir(), f"{json_out['claim_id']}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf.getvalue())

        st.markdown(f"[📄 View PDF Before Sending]({pdf_path})")
        st.download_button("⬇ Download PDF", pdf, file_name=f"{json_out['claim_id']}.pdf")

        # EMAIL SEND (PATH BASED – ZERO ERRORS)
        send_to = email if email else EMAIL_USER
        if st.button(f"Send Email – {json_out['claim_id']}"):
            import time
            yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASSWORD)
            yag.send(
                to=send_to,
                subject=f"Insurance Claim Report – {json_out['claim_id']}",
                contents=f"""
Dear Customer,

Your insurance claim has been successfully processed.

Claim ID: {json_out['claim_id']}
Loss: {json_out['incident']['loss_type']}
Severity: {json_out['incident']['severity']}
Asset: {json_out['incident']['affected_asset']}

PDF Report Attached.

Regards,
Insurance AI Bot
""",
                attachments=[pdf_path]   # <--- BEST & SAFE
            )
            msg = st.success("Email sent successfully!")
            time.sleep(3)
            msg.empty()

    # CLEAR ALL
    if st.button("🧹 Clear All Results"):
        st.session_state["history_outputs"] = []
        st.rerun()

# ===========================================
# PAGE – IMAGE CLAIM
# ===========================================
elif page =="Image Claim":
    st.title("Image-Based Claim")
    img_u=st.file_uploader("Upload Accident Image",type=["jpg","png"])
    if img_u and st.button("Analyze"):
        img=Image.open(img_u).convert("RGB")
        st.image(img)
        auto=detect_damage(img)
        t,lang=language_agent(auto)
        data,fraud,conf,fallback,cid,policy=normalize_claim(t)
        summary=f"Detected vehicle claim severity {data['severity']}"
        action="Tow vehicle" if data["severity"]=="High" else "Repair"
        human=f"Loss:{data['loss_type']}\nAsset:{data['affected_asset']}\nSeverity:{data['severity']}"
        json_out={
            "claim_id":cid,"policy_number":policy,"auto_text":auto,
            "incident":data,"ai_risk":{"fraud":fraud,"confidence_score":conf},
            "summary":summary,"human_friendly":human,
            "normalized_on":datetime.utcnow().isoformat()+"Z"
        }
        save_history(json_out)
        email = st.text_input("Email To Send PDF (Optional)")
        render_claim_output(json_out, human, email=email)

# ===========================================
# PAGE – VIDEO CLAIM
# ===========================================
elif page == "Video Claim":
    st.title("Video-Based Insurance Claim")

    video_file = st.file_uploader(
        "Upload Accident Video (MP4 / AVI / MOV)",
        type=["mp4", "avi", "mov"]
    )

    if video_file and st.button("Analyze Video Claim"):
        with st.spinner("Extracting frames and analyzing damage..."):
            auto_text, detected_severity = analyze_video_damage(video_file)

        st.success("Video analysis completed")
        st.write("Auto-detected description:", auto_text)

        translated, lang = language_agent(auto_text)
        data, fraud, conf, fallback, cid, policy = normalize_claim(translated)

        summary = f"Video-based claim with severity {data['severity']}."
        action = "Immediate towing and approval" if data["severity"] == "High" else "Initiate repair process"

        human_output = f"""
Claim ID: {cid}
Loss Type: {data['loss_type']}
Asset: {data['affected_asset']}
Severity: {data['severity']}
Fraud Risk: {fraud}
Confidence: {conf}

Summary:
{summary}

Recommended Action:
{action}
"""

        json_out = {
            "claim_id": cid,
            "policy_number": policy,
            "language": lang,
            "video_analysis": auto_text,
            "incident": data,
            "ai_risk": {
                "fraud": fraud,
                "confidence_score": conf
            },
            "summary": summary,
            "recommended_action": action,
            "normalized_on": datetime.utcnow().isoformat() + "Z"
        }

        save_history(json_out)
        
        email = st.text_input("Email To Send PDF (Optional)")
        render_claim_output(json_out, human_output, email=email)

# ===========================================
# PAGE – VOICE CLAIM
# ===========================================
elif page =="Voice Claim":
    st.title("Voice Based Claims")
    audio=st.file_uploader("Upload WAV or MP3")
    if audio and st.button("Process Audio"):
        text=transcribe_audio(audio)
        st.write("Speech Text:",text)
        t,lang=language_agent(text)
        data,fraud,conf,fallback,cid,policy=normalize_claim(t)
        summary=f"Voice claim severity {data['severity']}"
        action="Tow vehicle" if data["severity"]=="High" else "Repair"
        human=f"Loss:{data['loss_type']} Severity:{data['severity']}"
        json_out={
            "claim_id":cid,"policy_number":policy,"speech_text":text,
            "incident":data,"ai_risk":{"fraud":fraud,"confidence_score":conf},
            "summary":summary,"human_friendly":human,"normalized_on":datetime.utcnow().isoformat()+"Z"
        }
        save_history(json_out)
        email = st.text_input("Email To Send PDF (Optional)")
        render_claim_output(json_out, human, email=email)

# ===========================================
# PAGE – CHATBOT (ADVANCED)
# ===========================================
elif page == "Chatbot":

    st.title("AI Assistant Chatbot 🤖")

    # ===============================
    # SHOW SELECTED CLAIM (FROM SIDEBAR)
    # ===============================
    if st.session_state.get("selected_claim_id"):
        cid = st.session_state.selected_claim_id
        conn = sqlite3.connect("claims.db")
        row = conn.execute(
            "SELECT * FROM claims WHERE claim_id = ?", (cid,)
        ).fetchone()
        conn.close()

        if row:
            st.info(f"📄 Selected Claim: {cid}")
            st.json({
                "claim_id": row[1],
                "policy_number": row[2],
                "summary": row[3],
                "fraud": row[4],
                "confidence": row[5],
                "loss_type": row[6],
                "severity": row[7],
                "asset": row[8],
                "created_at": row[9],
            })

    # ---------- Session Defaults ----------
    if "show_uploader" not in st.session_state:
        st.session_state.show_uploader = False
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""
    if "detected_lang" not in st.session_state:
        st.session_state.detected_lang = "English"

    # ---------- CHAT HISTORY ----------
    chat_box = st.container()
    with chat_box:
        for msg in st.session_state.chat_history[-40:]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # ---------- FIXED BOTTOM BAR STYLE ----------
    st.markdown("""
    <style>
    .chat-bottom-container{
        position:fixed;
        bottom:0;
        left:50%;
        transform:translateX(-50%);
        width:min(1100px, 95%);
        padding:10px;
        background:#111;
        border-top:2px solid #444;
        border-radius:12px 12px 0 0;
        z-index:9999;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chat-bottom-container">', unsafe_allow_html=True)
    with st.container():
        col_attach, col_input, col_mic, col_send, col_clear = st.columns([1, 7, 1, 1, 1])

        # 📎 Attachment Toggle
        with col_attach:
            if st.button("📎"):
                st.session_state.show_uploader = not st.session_state.show_uploader

        # 🎤 MIC INPUT
        with col_mic:
            mic = mic_recorder(start_prompt="🎤", stop_prompt="⏹")
            if mic and "bytes" in mic:
                audio_bytes = io.BytesIO(mic["bytes"])
                audio = AudioSegment.from_file(audio_bytes)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    audio.export(tmp.name, format="wav")
                recog = sr.Recognizer()
                with sr.AudioFile(tmp.name) as src:
                    audio_data = recog.record(src)
                try:
                    text = recog.recognize_google(audio_data, language="en-IN")
                    st.session_state.chat_input = text
                except:
                    pass

        # ⌨️ TEXT INPUT
        with col_input:
            st.session_state.chat_input = st.text_input(
                "",
                value=st.session_state.chat_input,
                placeholder="Type or speak...",
                label_visibility="collapsed"
            )

        # ➤ SEND
        with col_send:
            send = st.button("➤")

        # 🗑 CLEAR
        with col_clear:
            if st.button("🗑"):
                st.session_state.chat_history.clear()
                st.session_state.chat_input = ""
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- FILE UPLOADER ----------
    if st.session_state.show_uploader:
        files = st.file_uploader(
            "Upload files",
            accept_multiple_files=True,
            type=["pdf", "png", "jpg", "jpeg", "docx", "mp4"]
        )
        if files:
            st.success("Files uploaded successfully")
            st.session_state.show_uploader = False

    # ---------- SEND MESSAGE ----------
    if send and st.session_state.chat_input.strip():
        query = st.session_state.chat_input.strip()

        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = chatbot_agent(
                    query,
                    st.session_state.chat_history
                )
                st.write(reply)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": reply
        })

        st.session_state.chat_input = ""
        st.rerun()

# ===========================================
# PAGE – HISTORY
# ===========================================
elif page =="History":
    st.title("History Log")
    data=fetch_history()
    st.table(data)

# ===========================================
# PAGE – ANALYTICS DASHBOARD (IMAGE STYLE)
# ===========================================
elif page == "Analytics Dashboard":

    import streamlit as st
    import pandas as pd
    import sqlite3
    import altair as alt

    st.title("📊 Analytics Dashboard – Claims Insights")
    st.caption("Executive-style visual summary of claims data")

    # -------------------------------
    # Load Data
    # -------------------------------
    conn = sqlite3.connect("claims.db")
    try:
        df = pd.read_sql_query("SELECT * FROM claims", conn)
    except:
        df = None
    conn.close()

    if df is None or df.empty:
        st.info("No claims data available.")
    else:
        df["created_at"] = pd.to_datetime(df["created_at"])

        # ===============================
        # KPI CARDS (TOP – LIKE IMAGE)
        # ===============================
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Claims", len(df))
        col2.metric(
            "Fraud Rate (%)",
            round((df["fraud"] == "Yes").mean() * 100, 1)
        )
        col3.metric(
            "High Severity Claims",
            (df["severity"] == "High").sum()
        )

        st.divider()

        # ===============================
        # BAR CHART – SEVERITY (LIKE IMAGE BAR)
        # ===============================
        st.subheader("Claims Severity Distribution")

        severity_df = (
            df.groupby("severity")
              .size()
              .reset_index(name="Total Claims")
        )

        severity_chart = alt.Chart(severity_df).mark_bar(
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6
        ).encode(
            x=alt.X(
                "severity:N",
                title="Severity Level",
                axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y(
                "Total Claims:Q",
                title="Number of Claims"
            ),
            tooltip=["severity", "Total Claims"],
            color=alt.Color(
                "severity:N",
                legend=None
            )
        ).properties(
            height=320
        )

        st.altair_chart(severity_chart, use_container_width=True)

        st.divider()

        # ===============================
        # DONUT CHART – FRAUD (LIKE IMAGE)
        # ===============================
        st.subheader("Fraud Detection Overview")

        fraud_df = (
            df.groupby("fraud")
              .size()
              .reset_index(name="Total Claims")
        )

        fraud_chart = alt.Chart(fraud_df).mark_arc(
            innerRadius=90
        ).encode(
            theta="Total Claims:Q",
            color=alt.Color(
                "fraud:N",
                legend=alt.Legend(title="Claim Type")
            ),
            tooltip=["fraud", "Total Claims"]
        ).properties(
            height=320
        )

        st.altair_chart(fraud_chart, use_container_width=True)


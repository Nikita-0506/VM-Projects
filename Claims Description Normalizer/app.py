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
# NORMALIZER
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
# VOICE
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
    write(50, f"Language: {json_out['language']}")

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
    for line in json_out["recommended_action"].split("\n"):
        write(60, line)

    c.save()
    buf.seek(0)
    return buf

def pdf_to_img(buffer):
    import base64
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:application/pdf;base64,{b64}"

# ===========================================
# STREAMLIT MULTI-PAGE UI
# ===========================================
st.set_page_config(page_title="Claims AI", layout="wide")
pages=["Normalize Claim","Image Claim","Voice Claim","History","Analytics Dashboard"]
menu=st.sidebar.radio("Menu", pages)

# ===========================================
# PAGE – NORMALIZE 
# ===========================================
if menu == "Normalize Claim":
    st.title("Text Claim Normalizer")

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
# PAGE – IMAGE
# ===========================================
elif menu=="Image Claim":
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
        c1,c2=st.columns(2)
        c1.json(json_out); c2.text(human)
        pdf=make_pdf(json_out,human)
        st.components.v1.iframe(pdf_to_img(pdf), height=400)
        st.download_button("Download PDF",pdf,f"{cid}.pdf")

# ===========================================
# PAGE – VOICE
# ===========================================
elif menu=="Voice Claim":
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
        c1,c2=st.columns(2)
        c1.json(json_out); c2.text(human)
        pdf=make_pdf(json_out,human)
        st.components.v1.iframe(pdf_to_img(pdf), height=400)
        st.download_button("Download PDF", pdf, f"{cid}.pdf")

# ===========================================
# PAGE – HISTORY
# ===========================================
elif menu=="History":
    st.title("History Log")
    data=fetch_history()
    st.table(data)

# ===========================================
# PAGE – ANALYTICS
# ===========================================
elif menu=="Analytics Dashboard":
    st.title("Analytics Dashboard – Claims Insights")
    conn=sqlite3.connect("claims.db")
    df=None
    try:
        import pandas as pd
        df=pd.read_sql_query("SELECT * FROM claims",conn)
    except: df=None
    conn.close()
    if df is None or df.empty:
        st.info("No history yet.")
    else:
        st.bar_chart(df["severity"].value_counts())
        st.bar_chart(df["fraud"].value_counts())
        st.line_chart(df.groupby("created_at")["id"].count())

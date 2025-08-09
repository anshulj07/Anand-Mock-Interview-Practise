from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from mangum import Mangum
from pydantic import BaseModel
from io import BytesIO
import os
import PyPDF2
import openai
import logging
import time
import pickle
from typing import Dict, List

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# --- Env Vars ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is missing")
openai.api_key = OPENAI_API_KEY

# --- App ---
app = FastAPI()
handler = Mangum(app)  # For Vercel

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Storage ---
SESSION_FILE = "sessions.pkl"
sessions: Dict[str, Dict] = {}

def load_sessions():
    global sessions
    try:
        with open(SESSION_FILE, "rb") as f:
            sessions = pickle.load(f)
        logger.info("Sessions loaded.")
    except FileNotFoundError:
        sessions = {}
    except Exception as e:
        logger.error(f"Failed to load sessions: {e}")
        sessions = {}

def save_sessions():
    try:
        with open(SESSION_FILE, "wb") as f:
            pickle.dump(sessions, f)
        logger.info("Sessions saved.")
    except Exception as e:
        logger.error(f"Failed to save sessions: {e}")

load_sessions()

# --- Utils ---
def extract_text_from_pdf(file_stream: BytesIO) -> str:
    reader = PyPDF2.PdfReader(file_stream)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

def ask_gpt(prompt: str) -> str:
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a friendly HR interviewer."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return resp.choices[0].message["content"].strip()
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        raise HTTPException(status_code=500, detail="LLM API call failed")

# --- Models ---
class UserResponse(BaseModel):
    session_id: str
    answer: str
    is_complete: bool = False

class SessionID(BaseModel):
    session_id: str

# --- Routes ---
@app.get("/")
def home():
    return {"message": "Mock Interview API running"}

@app.post("/upload-resume")
async def upload_resume(resume: UploadFile = File(...)):
    text = extract_text_from_pdf(BytesIO(await resume.read()))
    session_id = str(time.time())
    sessions[session_id] = {"resume": text, "qa_history": [], "ended": False}
    save_sessions()
    return {"session_id": session_id}

@app.post("/start-interview")
async def start_interview(req: SessionID):
    s = sessions.get(req.session_id)
    if not s or s["ended"]:
        raise HTTPException(status_code=404, detail="Session not found")
    question = ask_gpt(f"Ask a short HR question (max 15 words) based on this resume:\n{s['resume']}")
    s["qa_history"].append({"question": question})
    save_sessions()
    return {"question": question}

@app.post("/submit-answer")
async def submit_answer(req: UserResponse):
    s = sessions.get(req.session_id)
    if not s or s["ended"]:
        raise HTTPException(status_code=404, detail="Session not found")
    if not s["qa_history"]:
        raise HTTPException(status_code=400, detail="No question asked yet")
    s["qa_history"][-1]["answer"] = req.answer
    if req.is_complete:
        question = ask_gpt(
            f"Given resume:\n{s['resume']}\n"
            f"History: {s['qa_history']}\n"
            f"Ask next short relevant question (max 15 words)."
        )
        s["qa_history"].append({"question": question})
    save_sessions()
    return {"qa_history": s["qa_history"]}

@app.post("/end-interview")
async def end_interview(req: SessionID):
    s = sessions.get(req.session_id)
    if not s or s["ended"]:
        raise HTTPException(status_code=404, detail="Session not found")
    s["ended"] = True
    feedback = ask_gpt(
        f"Provide concise interview feedback for:\n{s['qa_history']}\n"
        f"Include strengths, areas to improve, clarity & confidence."
    )
    save_sessions()
    return {"feedback": feedback}

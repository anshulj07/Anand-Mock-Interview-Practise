from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from dotenv import load_dotenv
from pydantic import BaseModel
from mangum import Mangum
import logging
import os
import time
import pickle
from typing import Dict, List
import openai
import PyPDF2

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing in environment variables")
openai.api_key = OPENAI_API_KEY

# FastAPI app
app = FastAPI()
handler = Mangum(app)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_FILE = "interview_sessions.pkl"
interview_sessions: Dict[str, Dict] = {}

# -------- Helpers --------
def extract_text_from_pdf(file_stream: BytesIO) -> str:
    reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def load_sessions():
    global interview_sessions
    try:
        with open(SESSION_FILE, "rb") as f:
            interview_sessions = pickle.load(f)
        logger.info("Sessions loaded.")
    except FileNotFoundError:
        interview_sessions = {}
    except Exception as e:
        logger.error(f"Error loading sessions: {e}")
        interview_sessions = {}

def save_sessions():
    try:
        with open(SESSION_FILE, "wb") as f:
            pickle.dump(interview_sessions, f)
        logger.info("Sessions saved.")
    except Exception as e:
        logger.error(f"Error saving sessions: {e}")

load_sessions()

def ask_gpt(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or gpt-4o-mini for faster/cheaper
            messages=[{"role": "system", "content": "You are a friendly HR interviewer."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail="LLM API call failed.")

# -------- Models --------
class UserResponse(BaseModel):
    session_id: str
    answer: str
    is_complete: bool = False

class EndInterviewRequest(BaseModel):
    session_id: str

class StartInterviewRequest(BaseModel):
    session_id: str

# -------- Endpoints --------
@app.get("/")
def home():
    return {"message": "Hello from FastAPI + OpenAI on Vercel"}

@app.post("/upload-resume")
async def upload_resume(resume: UploadFile = File(...)):
    text = extract_text_from_pdf(BytesIO(await resume.read()))
    session_id = str(time.time())
    interview_sessions[session_id] = {
        "resume": text,
        "qa_history": [],
        "phase": "general",
        "ended": False
    }
    save_sessions()
    return {"success": True, "session_id": session_id}

@app.post("/start-interview")
async def start_interview(req: StartInterviewRequest):
    session = interview_sessions.get(req.session_id)
    if not session or session["ended"]:
        raise HTTPException(status_code=404, detail="Session not found or ended")

    prompt = f"""Ask a short, engaging HR question (max 15 words) based on this resume:
    {session['resume']}
    """
    question = ask_gpt(prompt)
    session["qa_history"].append({"question": question})
    save_sessions()
    return {"success": True, "question": question}

@app.post("/submit-answer")
async def submit_answer(req: UserResponse):
    session = interview_sessions.get(req.session_id)
    if not session or session["ended"]:
        raise HTTPException(status_code=404, detail="Session not found or ended")

    if not session["qa_history"]:
        raise HTTPException(status_code=400, detail="No previous question")

    session["qa_history"][-1]["answer"] = req.answer
    save_sessions()

    if req.is_complete:
        prompt = f"""Given this resume and Q&A history:
        Resume: {session['resume']}
        History: {session['qa_history']}
        Ask the next relevant question (max 15 words).
        """
        question = ask_gpt(prompt)
        session["qa_history"].append({"question": question})
        save_sessions()
        return {"success": True, "question": question}

    return {"success": True}

@app.post("/end-interview")
async def end_interview(req: EndInterviewRequest):
    session = interview_sessions.get(req.session_id)
    if not session or session["ended"]:
        raise HTTPException(status_code=404, detail="Session not found or ended")

    session["ended"] = True
    prompt = f"""Provide concise interview feedback for this Q&A:
    {session['qa_history']}
    """
    feedback = ask_gpt(prompt)
    save_sessions()
    return {"success": True, "feedback": feedback}

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from resume_parse import extract_text_from_pdf
from io import BytesIO
import os
import time
import pickle
from typing import Dict
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import logging
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
app = FastAPI()

# Configure CORS to allow the frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set Groq API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LLM and Conversation Chain
llm = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile")
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

SESSION_FILE = "interview_sessions.pkl"
interview_sessions: Dict[str, Dict] = {}

def load_sessions():
    global interview_sessions
    try:
        with open(SESSION_FILE, "rb") as f:
            interview_sessions = pickle.load(f)
        logger.info("Sessions loaded successfully.")
    except FileNotFoundError:
        interview_sessions = {}
        logger.info("No session file found, starting with empty sessions.")
    except Exception as e:
        logger.error(f"Error loading sessions: {e}")
        interview_sessions = {}

def save_sessions():
    try:
        with open(SESSION_FILE, "wb") as f:
            pickle.dump(interview_sessions, f)
        logger.info("Sessions saved successfully.")
    except Exception as e:
        logger.error(f"Error saving sessions: {e}")

load_sessions()

class UserResponse(BaseModel):
    session_id: str
    answer: str
    is_complete: bool = False

class EndInterviewRequest(BaseModel):
    session_id: str

class StartInterviewRequest(BaseModel):
    session_id: str

class AudioRequest(BaseModel):
    session_id: str
    audio_data: str  # Base64-encoded audio

def run_interview(resume_text: str, chat_history: list, category: str = "general") -> str:
    formatted_history = "\n".join([
        f"Interviewer: {turn.get('question', '')}\nCandidate: {turn.get('answer', '')}"
        for turn in chat_history
    ])
    instructions = {
        "general": "Ask a concise behavioral or personality-based question, max 15 words.",
        "technical": "Ask a concise technical skills question from resume, max 15 words.",
        "projects": "Ask a concise project experience question from resume, max 15 words."
    }
    prompt = f"""
You are a friendly HR interviewer. Ask short, engaging questions, max 15 words.

Resume:
\"\"\"{resume_text}\"\"\"

Past Conversation:
{formatted_history}

Ask a single question from category: {category}.
{instructions[category]}
""".strip()
    try:
        response = conversation.predict(input=prompt).strip()
        words = response.split()
        if len(words) > 15:
            response = " ".join(words[:15]) + "?"
        elif not response.endswith("?"):
            response += "?"
        logger.info(f"Generated question for category {category}: {response}")
        return response
    except Exception as e:
        logger.error(f"Error generating question: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate question due to API error.")

def generate_feedback(chat_history: list) -> str:
    if not chat_history:
        return "No interview data available for feedback."
    history = "\n".join([
        f"Interviewer: {turn.get('question', '')}\nCandidate: {turn.get('answer', '')}"
        for turn in chat_history if isinstance(turn, dict) and all(key in turn for key in ['question', 'answer'])
    ])
    prompt = f"""
You are an HR expert giving friendly feedback after a mock interview.

Transcript:
{history}

Provide:
- Strengths
- Areas to improve
- Comments on clarity and confidence
Keep it concise and human-like.
""".strip()
    try:
        feedback = conversation.predict(input=prompt).strip()
        logger.info(f"Generated feedback: {feedback}")
        return feedback
    except Exception as e:
        logger.error(f"Error generating feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate feedback due to API error.")

@app.post("/upload-resume")
async def upload_resume(resume: UploadFile = File(...)):
    contents = await resume.read()
    text = extract_text_from_pdf(BytesIO(contents))
    session_id = str(time.time())
    interview_sessions[session_id] = {
        "resume": text,
        "qa_history": [],
        "start_time": None,
        "question_count": {"general": 0, "technical": 0, "projects": 0},
        "phase": "general",
        "ended": False  # New flag to track session end
    }
    save_sessions()
    logger.info(f"Uploaded resume for session {session_id}")
    return {"success": True, "session_id": session_id}

@app.post("/start-interview")
async def start_interview(request: StartInterviewRequest):
    session = interview_sessions.get(request.session_id)
    if not session or session.get("ended", False):
        raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found or already ended")
    session["start_time"] = time.time()
    category = "general"
    question = run_interview(session["resume"], session["qa_history"], category)
    session["qa_history"].append({"question": question, "type": category})
    session["question_count"][category] += 1
    save_sessions()
    logger.info(f"Started interview for session {request.session_id}")
    return {"success": True, "question": question, "session_id": request.session_id}

@app.post("/submit-answer")
async def submit_answer(request: UserResponse):
    session = interview_sessions.get(request.session_id)
    if not session or session.get("ended", False):
        raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found or already ended")
    elapsed_time = time.time() - session.get("start_time", time.time())
    if elapsed_time > 15 * 60:
        return await end_interview_internal(request.session_id)

    if not session["qa_history"]:
        raise HTTPException(status_code=400, detail="No previous question")
    last_qa = session["qa_history"][-1]
    if "answer" not in last_qa:
        last_qa["answer"] = request.answer
        session["qa_history"][-1] = last_qa
    save_sessions()

    if request.is_complete and request.answer:
        current_type = last_qa["type"]
        if session["phase"] == "general" and session["question_count"]["general"] < 3:
            next_category = "general"
        else:
            if session["phase"] != "technical_projects":
                session["phase"] = "technical_projects"
            next_category = "technical" if last_qa["type"] == "projects" else "projects"

        next_question = run_interview(session["resume"], session["qa_history"], next_category)
        session["qa_history"].append({"question": next_question, "type": next_category})
        session["question_count"][next_category] += 1
        save_sessions()
        logger.info(f"Submitted answer for session {request.session_id}, next question: {next_question}")
        return {"success": True, "question": next_question, "end_interview": False}
    return {"success": True, "question": None, "end_interview": False}

@app.post("/end-interview")
async def end_interview(request: EndInterviewRequest):
    return await end_interview_internal(request.session_id)

async def end_interview_internal(session_id: str):
    session = interview_sessions.get(session_id)
    if not session or session.get("ended", False):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or already ended")
    try:
        feedback = generate_feedback(session["qa_history"])
        result = {"success": True, "feedback": feedback, "end_interview": True}
        session["ended"] = True  # Mark as ended instead of deleting immediately
        save_sessions()
        logger.info(f"Ended interview for session {session_id}, feedback: {feedback}")
        # Schedule session cleanup after response
        asyncio.get_event_loop().call_later(1.0, lambda: cleanup_session(session_id))
        return result
    except Exception as e:
        logger.error(f"Error in end_interview_internal for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def cleanup_session(session_id: str):
    if session_id in interview_sessions and interview_sessions[session_id].get("ended", False):
        del interview_sessions[session_id]
        save_sessions()
        logger.info(f"Cleaned up session {session_id}")

# Install additional dependencies: pip install fastapi uvicorn python-multipart langchain-groq PyPDF2 python-dotenv SpeechRecognition pyaudio pydub
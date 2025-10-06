from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from agents import ResumeAnalysisAgent, JobAgent
import io
import tempfile

app = FastAPI()

# Allow CORS for local dev and frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Resume Analysis Endpoints ---

@app.post("/analyze_resume/")
def analyze_resume(
    api_key: str = Form(...),
    resume: UploadFile = File(...),
    role_requirements: Optional[str] = Form(None),
    custom_jd: Optional[str] = Form(None)
):
    agent = ResumeAnalysisAgent(api_key)
    # Save uploaded resume to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{resume.filename.split('.')[-1]}") as tmp:
        tmp.write(resume.file.read())
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        result = agent.analyze_resume(f, role_requirements=eval(role_requirements) if role_requirements else None, custom_jd=custom_jd)
    agent.cleanup()
    return JSONResponse(result)

@app.post("/resume_qa/")
def resume_qa(api_key: str = Form(...), question: str = Form(...), resume_text: str = Form(...)):
    agent = ResumeAnalysisAgent(api_key)
    agent.resume_text = resume_text
    agent.rag_vectorstore = agent.create_rag_vector_store(resume_text)
    answer = agent.ask_question(question)
    return {"answer": answer}

@app.post("/generate_interview_questions/")
def generate_interview_questions(
    api_key: str = Form(...),
    resume_text: str = Form(...),
    extracted_skills: str = Form(...),
    strengths: str = Form(...),
    missing_skills: str = Form(...),
    question_types: str = Form(...),
    difficulty: str = Form(...),
    num_questions: int = Form(...)
):
    agent = ResumeAnalysisAgent(api_key)
    agent.resume_text = resume_text
    agent.extracted_skills = eval(extracted_skills)
    agent.analysis_result = {"strengths": eval(strengths), "missing_skills": eval(missing_skills)}
    questions = agent.generate_interview_questions(eval(question_types), difficulty, num_questions)
    return {"questions": questions}

@app.post("/resume_improvements/")
def resume_improvements(
    api_key: str = Form(...),
    resume_text: str = Form(...),
    improvement_areas: str = Form(...),
    target_role: str = Form("")
):
    agent = ResumeAnalysisAgent(api_key)
    agent.resume_text = resume_text
    improvements = agent.improve_resume(eval(improvement_areas), target_role)
    return {"improvements": improvements}

@app.post("/improved_resume/")
def improved_resume(
    api_key: str = Form(...),
    resume_text: str = Form(...),
    target_role: str = Form("") ,
    highlight_skills: str = Form("")
):
    agent = ResumeAnalysisAgent(api_key)
    agent.resume_text = resume_text
    improved = agent.get_improved_resume(target_role, highlight_skills)
    return {"improved_resume": improved}

# --- Job Search Endpoint ---

@app.post("/job_search/")
def job_search(
    query: str = Form(...),
    location: str = Form(None),
    platform: str = Form("adzuna"),
    experience: Optional[int] = Form(None),
    num_results: int = Form(10),
    country: str = Form("gb")
):
    agent = JobAgent()
    jobs = agent.search_jobs(query, location, platform, experience, num_results, country)
    return {"jobs": jobs}

# --- Health Check ---

@app.get("/")
def root():
    return {"message": "Resume Tracking & AI Interview Backend is running."}

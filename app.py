import streamlit as st

st.set_page_config(
    page_title = "Recruitment Agent",
    page_icon = "🚀",
    layout  = "wide"
)
from dotenv import load_dotenv
# Load environment variables from a local .env file if present
load_dotenv()

import ui_restored_tmp as ui
from agents import ResumeAnalysisAgent, groq_chat, ollama_chat
from database import (
    init_mysql_db,
    create_user,
    authenticate_user,
    get_user_settings,
    save_user_settings,
    save_user_resume,
    get_user_resume_by_id,
)
import atexit
import os
import io
import json
import time
import wave
 
import requests

try:
    # OpenAI SDK v1 style
    from openai import OpenAI
except Exception:
    OpenAI = None

ROLE_REQUIREMENTS = {
    "AI/ML Engineer": [ 
        "Python", "TensorFlow", "Machine Learning", "Deep Learning", "LangChain",
        "MLOps", "Scikit-learn", "Natural Language Processing (NLP)", "Hugging Face", 
        "SQL", "Git"
        ,"Experiment Tracking (MLflow, DVC)"

        ],
    "Frontend Engineer": [
        "React", "Vue", "Angular", "HTML5", "CSS3", "JavaScript", "TypeScript",
        "Next.js", "Svelte", "Bootstrap", "Tailwind CSS", "GraphQL", "Redux",
        "WebAssembly", "Three.js", "Performance Optimization", "REST APIs",
        "Webpack", "Vite", "Responsive Design", "UI/UX Principles", "Testing (Jest, Cypress)"
],
    "Backend Engineer": [
        "Python", "Java", "Node.js", "Go", "REST APIs", "GraphQL", "gRPC",
        "Spring Boot", "Flask", "FastAPI", "Express.js", "Django",
        "SQL Databases", "NoSQL Databases", "PostgreSQL", "MySQL", "MongoDB",
        "Redis", "RabbitMQ", "Kafka", "Microservices", "Docker", "Kubernetes",
        "Cloud Services (AWS, GCP, Azure)", "CI/CD", "API Security", "Scalability & Performance Optimization"
    ],
    "Full Stack Developer": [
        "HTML5", "CSS3", "JavaScript", "TypeScript", "React", "Vue", "Angular",
        "Next.js", "Node.js", "Express.js", "Python", "Java", "Flask", "FastAPI",
        "Spring Boot", "SQL Databases", "NoSQL Databases", "PostgreSQL", "MySQL", "MongoDB",
        "REST APIs", "GraphQL", "Docker", "Kubernetes", "Microservices",
        "Cloud Services (AWS, GCP, Azure)", "Git", "CI/CD",
        "Responsive Design", "UI/UX Principles", "Testing (Jest, Cypress, PyTest)",
        "Performance Optimization", "API Security", "Version Control"
    ],
    "Data Scientist": [
        "Python",  "SQL", "Machine Learning", "Deep Learning", "Scikit-learn", 
        "TensorFlow/PyTorch", "Natural Language Processing (NLP)/Computer Vision", 
        "Data Visualization (Matplotlib, Seaborn, Plotly)", 
        "Pandas", "NumPy", "Data Preprocessing", "Feature Engineering",   
        "Model Deployment", "Docker", "Git", "Cloud Platforms (AWS, Azure, GCP)", 
        "Model Evaluation", "Experiment Tracking (MLflow, Weights & Biases)", 
        "Power BI/Tableau"
    ],
}

# Initialize session state defaults
if 'resume_agent' not in st.session_state:
    st.session_state.resume_agent = None
if 'resume_analyzed' not in st.session_state:
    st.session_state.resume_analyzed = False
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'interview' not in st.session_state:
    st.session_state.interview = {
        'started': False,
        'completed': False,
        'current': 0,
        'questions': [],
        'answers': [],
        'transcripts': [],
        'per_q_scores': [],
        'summary': None,
        'start_time': None,
        'max_duration_sec': 15 * 60,
        'decision': None,
    }
if 'user' not in st.session_state:
    st.session_state.user = None
if 'user_settings' not in st.session_state:
    st.session_state.user_settings = {}
def generate_followup_question(resume_text: str, last_question: str, transcript: str, api_key: str) -> str:
    """Create a brief, targeted follow-up question based on last question and the candidate answer."""
    provider = st.session_state.get('provider', 'openai')
    if provider == 'openai':
        client = _get_client(api_key)
        sys = (
            "You are a concise technical interviewer. Generate ONE short follow-up question (max 25 words) "
            "to probe deeper based on the prior question and the candidate's answer. If no follow-up is needed, respond with NONE."
        )
        user = (
            f"Resume context (for tailoring):\n{resume_text[:1500]}\n\n"
            f"Previous question: {last_question}\n"
            f"Candidate answer transcript: {transcript}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        content = resp.choices[0].message.content.strip()
        return None if content.upper().startswith("NONE") else content
    else:
        # Groq path
        sys = (
            "You are a concise technical interviewer. Generate ONE short follow-up question (max 25 words) "
            "to probe deeper based on the prior question and the candidate's answer. If no follow-up is needed, respond with NONE."
        )
        user = (
            f"Resume context (for tailoring):\n{resume_text[:1500]}\n\n"
            f"Previous question: {last_question}\n"
            f"Candidate answer transcript: {transcript}"
        )
        try:
            if st.session_state.get('provider') == 'groq':
                content = groq_chat(st.session_state.api_key, messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}])
            elif st.session_state.get('provider') == 'ollama':
                content = ollama_chat(messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}], model=st.session_state.get('ollama_model'), base_url=st.session_state.get('ollama_base_url'))
            else:
                return None
            content = content.strip()
            return None if content.upper().startswith("NONE") else content
        except Exception:
            return None


def start_mock_interview(agent: ResumeAnalysisAgent, question_types, difficulty, num_questions: int = 10):
    with st.spinner("Preparing your personalized interview..."):
        qs = agent.generate_interview_questions(question_types, difficulty, num_questions)
        # Normalize to list of question strings
        questions = []
        for q in qs:
            if isinstance(q, dict):
                questions.append(q.get("question") or q.get("text") or str(q))
            else:
                questions.append(str(q))
        st.session_state.interview = {
            'started': True,
            'completed': False,
            'current': 0,
            'questions': questions[:num_questions],
            'answers': [],
            'transcripts': [],
            'per_q_scores': [],
            'summary': None,
            'start_time': time.time(),
            'max_duration_sec': st.session_state.interview.get('max_duration_sec', 15*60),
            'decision': None,
        }


def process_answer(audio_file):
    """Handle one answer: STT -> scoring -> advance pointer."""
    api_key = st.session_state.api_key
    idx = st.session_state.interview['current']
    question = st.session_state.interview['questions'][idx]

    transcript = transcribe_audio(audio_file, api_key)
    scores = score_answer(question, transcript, api_key)

    st.session_state.interview['answers'].append(audio_file)
    st.session_state.interview['transcripts'].append(transcript)
    st.session_state.interview['per_q_scores'].append(scores)

    st.session_state.interview['current'] += 1
    if st.session_state.interview['current'] >= len(st.session_state.interview['questions']):
        # finalize summary
        perq = st.session_state.interview['per_q_scores']
        if perq:
            comm = sum(p.get('communication', 0) for p in perq) / len(perq)
            tech = sum(p.get('technical_knowledge', 0) for p in perq) / len(perq)
            prob = sum(p.get('problem_solving', 0) for p in perq) / len(perq)
            overall = sum(p.get('overall', 0) for p in perq) / len(perq)
            strengths = []
            weaknesses = []
            for p in perq:
                strengths += p.get('strengths', [])
                weaknesses += p.get('weaknesses', [])
            # keep top 5 unique
            strengths = list(dict.fromkeys(strengths))[:5]
            weaknesses = list(dict.fromkeys(weaknesses))[:5]
            st.session_state.interview['summary'] = {
                'communication': round(comm, 1),
                'technical_knowledge': round(tech, 1),
                'problem_solving': round(prob, 1),
                'overall': round(overall, 1),
                'strengths': strengths,
                'weaknesses': weaknesses,
            }
        st.session_state.interview['completed'] = True


def process_answer_pcm(pcm_bytes: bytes, sample_rate: int):
    """Process one answer captured from WebRTC PCM stream: transcribe, score, advance pointer."""
    api_key = st.session_state.api_key
    idx = st.session_state.interview['current']
    question = st.session_state.interview['questions'][idx]

    transcript = transcribe_pcm_bytes(pcm_bytes, sample_rate, api_key)
    scores = score_answer(question, transcript, api_key)

    st.session_state.interview['answers'].append(None)
    st.session_state.interview['transcripts'].append(transcript)
    st.session_state.interview['per_q_scores'].append(scores)

    # Decide whether to ask a follow-up or move on
    followup = None
    try:
        if scores.get('overall', 0) < 60:
            resume_text = getattr(st.session_state.resume_agent, 'resume_text', '') or ''
            followup = generate_followup_question(resume_text, question, transcript, api_key)
    except Exception:
        followup = None

    st.session_state.interview['current'] += 1
    # Insert follow-up if available and time remains
    if followup and st.session_state.interview['current'] < len(st.session_state.interview['questions']):
        st.session_state.interview['questions'].insert(st.session_state.interview['current'], followup)

    # Finalize if done or timed-out handled by UI caller
    if st.session_state.interview['current'] >= len(st.session_state.interview['questions']):
        perq = st.session_state.interview['per_q_scores']
        if perq:
            comm = sum(p.get('communication', 0) for p in perq) / len(perq)
            tech = sum(p.get('technical_knowledge', 0) for p in perq) / len(perq)
            prob = sum(p.get('problem_solving', 0) for p in perq) / len(perq)
            overall = sum(p.get('overall', 0) for p in perq) / len(perq)
            strengths = []
            weaknesses = []
            for p in perq:
                strengths += p.get('strengths', [])
                weaknesses += p.get('weaknesses', [])
            strengths = list(dict.fromkeys(strengths))[:5]
            weaknesses = list(dict.fromkeys(weaknesses))[:5]
            st.session_state.interview['summary'] = {
                'communication': round(comm, 1),
                'technical_knowledge': round(tech, 1),
                'problem_solving': round(prob, 1),
                'overall': round(overall, 1),
                'strengths': strengths,
                'weaknesses': weaknesses,
            }
        st.session_state.interview['completed'] = True


def end_interview_now():
    """Force-end the interview and compute summary + decision."""
    inter = st.session_state.interview
    if inter['completed']:
        return
    # compute summary if not present
    perq = inter['per_q_scores']
    if perq:
        comm = sum(p.get('communication', 0) for p in perq) / len(perq)
        tech = sum(p.get('technical_knowledge', 0) for p in perq) / len(perq)
        prob = sum(p.get('problem_solving', 0) for p in perq) / len(perq)
        overall = sum(p.get('overall', 0) for p in perq) / len(perq)
        strengths = []
        weaknesses = []
        for p in perq:
            strengths += p.get('strengths', [])
            weaknesses += p.get('weaknesses', [])
        strengths = list(dict.fromkeys(strengths))[:5]
        weaknesses = list(dict.fromkeys(weaknesses))[:5]
        inter['summary'] = {
            'communication': round(comm, 1),
            'technical_knowledge': round(tech, 1),
            'problem_solving': round(prob, 1),
            'overall': round(overall, 1),
            'strengths': strengths,
            'weaknesses': weaknesses,
        }
    inter['completed'] = True
    # decision threshold
    overall = inter.get('summary', {}).get('overall', 0)
    cutoff = getattr(st.session_state.resume_agent, 'cutoff_score', 75)
    inter['decision'] = bool(overall >= cutoff)


def process_empty_answer():
    """Advance to next question treating the answer as empty/no response."""
    api_key = st.session_state.api_key
    idx = st.session_state.interview['current']
    question = st.session_state.interview['questions'][idx]
    transcript = ""
    scores = score_answer(question, transcript, api_key)

    st.session_state.interview['answers'].append(None)
    st.session_state.interview['transcripts'].append(transcript)
    st.session_state.interview['per_q_scores'].append(scores)

    followup = None
    try:
        if scores.get('overall', 0) < 60:
            resume_text = getattr(st.session_state.resume_agent, 'resume_text', '') or ''
            followup = generate_followup_question(resume_text, question, transcript, api_key)
    except Exception:
        followup = None

    st.session_state.interview['current'] += 1
    if followup and st.session_state.interview['current'] < len(st.session_state.interview['questions']):
        st.session_state.interview['questions'].insert(st.session_state.interview['current'], followup)

    if st.session_state.interview['current'] >= len(st.session_state.interview['questions']):
        perq = st.session_state.interview['per_q_scores']
        if perq:
            comm = sum(p.get('communication', 0) for p in perq) / len(perq)
            tech = sum(p.get('technical_knowledge', 0) for p in perq) / len(perq)
            prob = sum(p.get('problem_solving', 0) for p in perq) / len(perq)
            overall = sum(p.get('overall', 0) for p in perq) / len(perq)
            strengths = []
            weaknesses = []
            for p in perq:
                strengths += p.get('strengths', [])
                weaknesses += p.get('weaknesses', [])
            strengths = list(dict.fromkeys(strengths))[:5]
            weaknesses = list(dict.fromkeys(weaknesses))[:5]
            st.session_state.interview['summary'] = {
                'communication': round(comm, 1),
                'technical_knowledge': round(tech, 1),
                'problem_solving': round(prob, 1),
                'overall': round(overall, 1),
                'strengths': strengths,
                'weaknesses': weaknesses,
            }
        st.session_state.interview['completed'] = True


    
    
def login_view() -> bool:
    st.header("🔐 Sign in to continue")
    tabs = st.tabs(["Login", "Register"])
    logged_in = False
    with tabs[0]:
        with st.form("login_form"):
            u = st.text_input("Username", key="login_user")
            p = st.text_input("Password", type="password", key="login_pass")
            submit = st.form_submit_button("Login", type="primary")
        if submit:
            user = authenticate_user(u, p)
            if user:
                st.session_state.user = user
                st.session_state.user_settings = get_user_settings(user['id']) or {}
                logged_in = True
                st.success(f"Welcome back, {user['username']}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    with tabs[1]:
        with st.form("register_form"):
            ru = st.text_input("Choose a username", key="reg_user")
            rp = st.text_input("Choose a password", type="password", key="reg_pass")
            submit_r = st.form_submit_button("Create account", type="primary")
        if submit_r:
            user_id = create_user(ru, rp)
            if user_id:
                st.success("Account created. Please log in.")
            else:
                st.error("Username already exists or invalid input.")
    return logged_in or bool(st.session_state.user)


def ensure_logged_in() -> bool:
    try:
        init_mysql_db()
    except Exception as e:
        st.warning(f"Database init warning: {e}")
    if not st.session_state.user:
        return login_view()
    return True


def setup_agent(config):
    provider = config.get("provider", "groq")
    if provider == 'groq':
        api_key = config.get("api_key") or os.getenv("GROQ_API_KEY")
        selected_model = config.get("model") or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"
        if not api_key:
            st.error("Please provide your GROQ_API_KEY (or set it in .env).")
            st.stop()
    elif provider == 'ollama':
        api_key = None
        selected_model = config.get("model") or os.getenv("OLLAMA_MODEL") or "llama3.1:8b"
        st.session_state.ollama_base_url = config.get("ollama_base_url") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    else:  # openai
        api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        selected_model = None
        if not api_key:
            st.error("Please provide your OPENAI_API_KEY (or set it in .env).")
            st.stop()

    st.session_state.provider = provider
    st.session_state.api_key = api_key
    if provider == 'groq':
        st.session_state.groq_model = selected_model
    elif provider == 'ollama':
        st.session_state.ollama_model = selected_model

    if st.session_state.resume_agent is None:
        st.session_state.resume_agent = ResumeAnalysisAgent(
            api_key=api_key,
            model=(st.session_state.get('ollama_model') if provider == 'ollama' else st.session_state.get('groq_model')),
            provider=provider,
            ollama_base_url=st.session_state.get('ollama_base_url'),
            user_id=(st.session_state.user['id'] if st.session_state.user else None),
            vector_cache_dir=os.getenv('VECTOR_CACHE_DIR') or '.cache/faiss'
        )
    else:
        st.session_state.resume_agent.api_key = api_key
        st.session_state.resume_agent.provider = provider
        st.session_state.resume_agent.model = (st.session_state.get('ollama_model') if provider == 'ollama' else st.session_state.get('groq_model'))
        if provider == 'ollama':
            st.session_state.resume_agent.ollama_base_url = st.session_state.get('ollama_base_url')
        try:
            st.session_state.resume_agent.user_id = (st.session_state.user['id'] if st.session_state.user else None)
        except Exception:
            pass
    # Skip provider sanity pings to avoid extra network calls on reruns
    return st.session_state.resume_agent


def analyze_resume(agent, resume_file, role, custom_jd, quick: bool = False):
    if not resume_file:
        st.error("Please upload a resume or select a saved resume.")
        return
    
    with st.spinner("Quick analysis running..." if quick else "Analyzing resume... This may take a minute."):
        try:
            # Check if using saved resume
            use_saved_resume_id = st.session_state.get('use_saved_resume_id')
            
            print(f"DEBUG: resume_file={resume_file}, use_saved_resume_id={use_saved_resume_id}")
            
            if resume_file == "USE_SAVED_RESUME" and use_saved_resume_id and st.session_state.get('user'):
                # Using saved resume
                print(f"DEBUG: Using saved resume with ID {use_saved_resume_id}")
                ur = get_user_resume_by_id(st.session_state.user['id'], use_saved_resume_id)
                if ur and ur.get('resume_text'):
                    print(f"DEBUG: Found saved resume, text length: {len(ur.get('resume_text', ''))}")
                    result = agent.analyze_resume_text(
                        ur['resume_text'],
                        custom_jd=custom_jd if custom_jd else None,
                        role_requirements=ROLE_REQUIREMENTS.get(role) if not custom_jd else None,
                        quick=quick,
                    )
                else:
                    st.error("Saved resume not found. Please upload a new file.")
                    return
            else:
                # Using newly uploaded resume
                print(f"DEBUG: Using newly uploaded resume")
                result = agent.analyze_resume(
                    resume_file,
                    custom_jd=custom_jd if custom_jd else None,
                    role_requirements=ROLE_REQUIREMENTS.get(role) if not custom_jd else None,
                    quick=quick,
                )
            st.session_state.resume_analyzed = True
            st.session_state.analysis_result = result
            
            # Save newly uploaded resume to database
            try:
                if st.session_state.user and getattr(agent, 'resume_text', None):
                    r_hash = getattr(agent, 'resume_hash', None)
                    # Only save if it's a new upload (not a saved resume)
                    if resume_file != "USE_SAVED_RESUME":
                        save_user_resume(
                            st.session_state.user['id'],
                            getattr(resume_file, 'name', 'uploaded_resume'),
                            r_hash,
                            agent.resume_text,
                        )
            except Exception as e:
                st.info(f"Saved analysis; resume cache note: {e}")
            return result
        except Exception as e:
            st.error(f"Error analyzing resume: {e}")


def ask_question(agent, question):
    try:
        with st.spinner("Generating response..."):
            response = agent.ask_question(question)
            return response
    except Exception as e:
        return f"Error:{e}"


def generate_interview_questions(agent, question_types, difficulty, num_questions):
    try:
        with st.spinner("Generating pesonalized interview questions..."):
            questions = agent.generate_interview_questions(question_types, difficulty, num_questions)
            return questions
    except Exception as e:
        st.error(f"Error Generating questions:{e}")
        return []


def generate_cover_letter(agent: ResumeAnalysisAgent, company: str, role: str, jd: str, tone: str, length: str):
    try:
        with st.spinner("Generating cover letter..."):
            return agent.generate_cover_letter(company=company, role=role, job_description=jd, tone=tone, length=length)
    except Exception as e:
        st.error(f"Error generating cover letter: {e}")
        return "Error generating cover letter."


def improve_resume(agent: ResumeAnalysisAgent, improvement_areas: list, target_role: str):
    """Generate resume improvement suggestions"""
    try:
        result = agent.improve_resume(improvement_areas, target_role)
        # Debug: Check if we got actual suggestions
        if result:
            has_content = any(
                imp.get('specific') and len(imp.get('specific', [])) > 1 
                for imp in result.values() if isinstance(imp, dict)
            )
            if not has_content:
                st.warning("⚠️ Received generic suggestions. The AI may need more context. Try analyzing the resume first or providing a target role.")
        return result
    except Exception as e:
        st.error(f"Error generating improvement suggestions: {e}")
        import traceback
        st.code(traceback.format_exc())
        return {}


def get_improved_resume(agent: ResumeAnalysisAgent, target_role: str, highlight_skills: str):
    """Generate an improved version of the resume"""
    try:
        with st.spinner("Generating improved resume..."):
            return agent.get_improved_resume(target_role, highlight_skills)
    except Exception as e:
        st.error(f"Error generating improved resume: {e}")
        return "Error generating improved resume."


def update_resume_latex(agent: ResumeAnalysisAgent, latex_src: str, jd: str):
    try:
        with st.spinner("Updating LaTeX resume..."):
            return agent.generate_updated_resume_latex(latex_src, jd)
    except Exception as e:
        st.error(f"Error updating LaTeX resume: {e}")
        return "Error updating LaTeX resume."


# Mock interview helpers (OpenAI TTS/STT and scoring for non-OpenAI providers)
def _get_client(api_key: str):
    if st.session_state.get('provider', 'openai') != 'openai':
        raise RuntimeError("OpenAI client requested but provider is not set to OpenAI.")
    if OpenAI is None:
        raise RuntimeError("openai package not found. Install with: pip install openai>=1.0.0")
    return OpenAI(api_key=api_key)


def synthesize_speech(text: str, api_key: str) -> bytes:
    if st.session_state.get('provider', 'openai') != 'openai':
        raise RuntimeError("TTS is only available with the OpenAI provider in this app.")
    client = _get_client(api_key)
    resp = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        response_format="mp3",
    )
    if hasattr(resp, "content") and isinstance(resp.content, (bytes, bytearray)):
        return resp.content
    try:
        return bytes(resp)
    except Exception:
        pass
    try:
        return resp.read()
    except Exception:
        pass
    raise RuntimeError("TTS response not in expected binary format")


def transcribe_audio(audio_file, api_key: str) -> str:
    if st.session_state.get('provider', 'openai') != 'openai':
        raise RuntimeError("Audio transcription is only available with the OpenAI provider in this app.")
    client = _get_client(api_key)
    data = audio_file.read()
    audio_file.seek(0)
    fname = getattr(audio_file, 'name', 'answer.wav')
    mime = getattr(audio_file, 'type', 'audio/wav')
    file_tuple = (fname, io.BytesIO(data), mime)
    tr = client.audio.transcriptions.create(
        model="whisper-1",
        file=file_tuple,
        response_format="json",
    )
    text = getattr(tr, 'text', None)
    if text is None:
        try:
            text = tr["text"]
        except Exception:
            text = ""
    return text or ""


def _wav_bytes_from_pcm16(pcm_bytes: bytes, sample_rate: int = 48000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf.read()


def transcribe_pcm_bytes(pcm_bytes: bytes, sample_rate: int, api_key: str) -> str:
    wav_bytes = _wav_bytes_from_pcm16(pcm_bytes, sample_rate=sample_rate, channels=1)
    if st.session_state.get('provider', 'openai') != 'openai':
        raise RuntimeError("Audio transcription is only available with the OpenAI provider in this app.")
    client = _get_client(api_key)
    fname = 'answer.wav'
    mime = 'audio/wav'
    file_tuple = (fname, io.BytesIO(wav_bytes), mime)
    tr = client.audio.transcriptions.create(
        model="whisper-1",
        file=file_tuple,
        response_format="json",
    )
    text = getattr(tr, 'text', None)
    if text:
        return text
    try:
        return tr.get('text', '')
    except Exception:
        return ""


def score_answer(question: str, transcript: str, api_key: str) -> dict:
    provider = st.session_state.get('provider', 'openai')
    if provider == 'openai':
        _ = _get_client(api_key)  # presence check
    if provider == 'groq':
        sys = (
            "You are an expert technical interviewer. Score the candidate's single answer strictly. "
            "Return ONLY compact JSON with keys: communication (0-10), technical_knowledge (0-10), "
            "problem_solving (0-10), overall (0-100), strengths (array of short phrases), "
            "weaknesses (array of short phrases), feedback (<=60 words)."
        )
        user = f"Question: {question}\n\nCandidate answer (transcript): {transcript}"
        try:
            content = groq_chat(st.session_state.api_key, messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}])
            try:
                return json.loads(content)
            except Exception:
                return {
                    "communication": 5,
                    "technical_knowledge": 5,
                    "problem_solving": 5,
                    "overall": 50,
                    "strengths": [],
                    "weaknesses": [],
                    "feedback": "",
                }
        except Exception as e:
            return {
                "communication": 5,
                "technical_knowledge": 5,
                "problem_solving": 5,
                "overall": 50,
                "strengths": [],
                "weaknesses": [],
                "feedback": f"Error scoring with Groq: {e}",
            }
    elif provider == 'ollama':
        sys = (
            "You are an expert technical interviewer. Score the candidate's single answer strictly. "
            "Return ONLY compact JSON with keys: communication (0-10), technical_knowledge (0-10), "
            "problem_solving (0-10), overall (0-100), strengths (array of short phrases), "
            "weaknesses (array of short phrases), feedback (<=60 words)."
        )
        user = f"Question: {question}\n\nCandidate answer (transcript): {transcript}"
        try:
            content = ollama_chat(messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}], model=st.session_state.get('ollama_model'), base_url=st.session_state.get('ollama_base_url'))
            try:
                return json.loads(content)
            except Exception:
                return {
                    "communication": 5,
                    "technical_knowledge": 5,
                    "problem_solving": 5,
                    "overall": 50,
                    "strengths": [],
                    "weaknesses": [],
                    "feedback": "",
                }
        except Exception as e:
            return {
                "communication": 5,
                "technical_knowledge": 5,
                "problem_solving": 5,
                "overall": 50,
                "strengths": [],
                "weaknesses": [],
                "feedback": f"Error scoring with Ollama: {e}",
            }


def cleanup():
    """clean up resources when the app exits"""
    if st.session_state.resume_agent:
        st.session_state.resume_agent.cleanup()
       
atexit.register(cleanup)        

def main():
    ui.setup_page()
    ui.display_header()
    # Login gate
    if not ensure_logged_in():
        return
    
    config = ui.setup_sidebar() # config = openai api key
    # Merge with user settings
    # Prefer user saved defaults if present
    us = st.session_state.get('user_settings') or {}
    if us:
        # Only set defaults if not provided in the current sidebar selection
        config.setdefault('provider', us.get('provider'))
        config.setdefault('model', us.get('model'))
        if config.get('provider') == 'ollama' and 'ollama_base_url' not in config and us.get('ollama_base_url'):
            config['ollama_base_url'] = us.get('ollama_base_url')
    agent  = setup_agent(config)
    # Save the settings for this user
    try:
        if st.session_state.user:
            to_save = {
                'provider': st.session_state.get('provider'),
                'model': (st.session_state.get('groq_model') if st.session_state.get('provider')=='groq' else st.session_state.get('ollama_model')),
                'ollama_base_url': st.session_state.get('ollama_base_url') if st.session_state.get('provider')=='ollama' else None,
                'jooble_api_key': config.get('jooble_api_key') or (st.session_state.get('user_settings') or {}).get('jooble_api_key'),
            }
            save_user_settings(st.session_state.user['id'], to_save)
            st.session_state.user_settings = to_save
    except Exception as e:
        st.info(f"Could not save settings: {e}")
    
    # Disabled auto-analysis - users should manually click "Analyze Resume" button
    # This prevents unnecessary API calls when just logging in or switching pages
    # try:
    #     if not st.session_state.get('resume_analyzed') and st.session_state.get('user') and (
    #         config.get('selected_resume_id') or st.session_state.get('selected_resume_id')
    #     ):
    #         sel_id = config.get('selected_resume_id') or st.session_state.get('selected_resume_id')
    #         ur = get_user_resume_by_id(st.session_state.user['id'], sel_id)
    #         if ur and ur.get('resume_text'):
    #             # Use a sensible default role if none picked yet
    #             default_role = next(iter(ROLE_REQUIREMENTS.keys())) if ROLE_REQUIREMENTS else None
    #             role_reqs = ROLE_REQUIREMENTS.get(default_role) if default_role else None
    #             with st.spinner("Loading your saved resume and preparing analysis..."):
    #                 # Run in quick mode to avoid rate limits on auto-load
    #                 result = agent.analyze_resume_text(ur['resume_text'], role_requirements=role_reqs, quick=True)
    #             st.session_state.resume_analyzed = True
    #             st.session_state.analysis_result = result
    #             st.session_state.analysis_result['_user'] = st.session_state.user['username'] if st.session_state.user else 'anonymous'
    #             if result and result.get('note') and 'Quick analysis' in result['note']:
    #                 st.info("Using your selected saved resume (quick analysis). Click Analyze for full detailed analysis with your chosen role/JD.")
    #             else:
    #                 st.info("Using your selected saved resume (cached).")
    # except Exception as e:
    #     st.warning(f"Auto-analysis of saved resume skipped: {e}")

    # Status summary to help users know what's configured
    st.caption(f"Provider: {st.session_state.get('provider')} | Model: {st.session_state.get('groq_model') if st.session_state.get('provider')=='groq' else st.session_state.get('ollama_model')} | Analyzed: {st.session_state.get('resume_analyzed')}")
    
    tabs  = ui.create_tabs()
    
    with tabs[0]:
        role , custom_jd = ui.role_selection_section(ROLE_REQUIREMENTS)
        uploaded_resume = ui.resume_upload_section()
        
        # Fast vs full analysis toggle
        quick_mode = st.checkbox(
            "Quick analysis (faster, fewer skills, skips deep weaknesses)",
            value=True,
            help="Quick mode avoids an extra JD skill extraction call and limits skills to speed up analysis."
        )

        col1 , col2 , col3 = st.columns([1 , 1 , 1])
        with col2:
            if st.button("Analyze Resume", type="primary"):
                # Check if we have either a new upload or a saved resume selected
                has_resume = uploaded_resume is not None
                
                if agent and has_resume:
                    result = analyze_resume(agent, uploaded_resume, role, custom_jd, quick=quick_mode)
                    if result:  # Only update if successful
                        st.session_state.analysis_result = result
                        st.session_state.resume_analyzed = True
                        # Tag the analysis with user context (future use)
                        st.session_state.analysis_result['_user'] = st.session_state.user['username'] if st.session_state.user else 'anonymous'
                        st.rerun()
                else:
                    if not agent:
                        st.warning("Please configure your provider/API key in the sidebar.")
                    elif not has_resume:
                        st.warning("Please upload a resume or select a saved resume.")
                    
            
        if st.session_state.analysis_result: 
            ui.display_analysis_results(st.session_state.analysis_result)
            
    
    with tabs[1]:
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.resume_qa_section(
                has_resume  = True,
                ask_question_func = lambda q: ask_question(st.session_state.resume_agent , q)
                
            )
        else:
            st.warning("Please upload and anlyze a resume first in the 'Resume Analysis' tab.")
    
    
    with tabs[2]:
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.interview_questions_section(
                has_resume = True,
                generate_questions_func = lambda types , diff , num:
                    generate_interview_questions(st.session_state.resume_agent , types , diff , num)
            )
        else:
            st.warning("please upload and analyze a resume first in the 'Resume Analysis' tab.")    
            
    
    with tabs[3]:   # Resume Improvement tab
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.resume_improvement_section(
                has_resume=True,
                improve_resume_func=lambda areas, role: improve_resume(st.session_state.resume_agent, areas, role),
                get_improved_resume_func=lambda role, skills: get_improved_resume(st.session_state.resume_agent, role, skills)
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")

    with tabs[4]:   # Cover Letter tab
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.cover_letter_section(
                has_resume=True,
                generate_cover_letter_func=lambda company, role, jd, tone, length: generate_cover_letter(
                    st.session_state.resume_agent, company, role, jd, tone, length
                ),
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")

    with tabs[5]:   # Job Search tab
        ui.job_search_ui() 

    with tabs[6]:   # Mock Interview - Coming Soon
        st.info("🚧 **Mock Interview Feature - Coming Soon!**")
        st.markdown("""
        This feature will include:
        - 🎙️ Voice-based interview practice
        - 📊 Real-time answer evaluation
        - 💬 AI-powered feedback
        - 🎯 Personalized interview coaching
        
        Stay tuned for updates!
        """)
        # Keep old code commented for future use
        # if st.session_state.resume_analyzed and st.session_state.resume_agent:
        #     ui.mock_interview_section(
        #         has_resume=True,
        #         start_interview_func=lambda types, diff, num: start_mock_interview(st.session_state.resume_agent, types, diff, num),
        #         play_tts_func=lambda text: synthesize_speech(text, st.session_state.api_key),
        #         process_audio_answer_func=lambda audio_file: process_answer(audio_file),
        #     )
        # else:
        #     st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")

    with tabs[7]:   # LaTeX Resume Update - Coming Soon
        st.info("🚧 **LaTeX Resume Update Feature - Coming Soon!**")
        st.markdown("""
        This feature will include:
        - 📄 LaTeX resume parsing and updates
        - 🎨 Format-preserving resume enhancements
        - 🔧 Automatic content optimization for job descriptions
        - 📝 Professional LaTeX templates
        
        Stay tuned for updates!
        """)
        # Keep old code commented for future use
        # ui.latex_resume_update_section(
        #     has_resume=True,  # allow usage regardless, since user pastes LaTeX
        #     update_latex_func=lambda latex_src, jd: update_resume_latex(st.session_state.resume_agent, latex_src, jd),
        # ) 

  

if __name__ == "__main__"            :
    main()
    
import streamlit as st

st.set_page_config(
    page_title = "Recruitment Agent",
    page_icon = "🚀",
    layout  = "wide"
)

import ui
from agents import ResumeAnalysisAgent
import atexit
import io
import json

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

if 'resume_agent' not in st.session_state:
    st.session_state.resume_agent = None
if 'resume_analyzed' not in st.session_state:
    st.session_state.resume_analyzed = False
    
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# NEW: store API key for speech funcs
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None
# NEW: mock interview state
if 'interview' not in st.session_state:
    st.session_state.interview = {
        'started': False,
        'completed': False,
        'current': 0,
        'questions': [],
        'answers': [],
        'transcripts': [],
        'per_q_scores': [],  # list of dicts per question
        'summary': None,
    }    
    


def setup_agent(config)    :
    """set up the resume analysis agent with the provided configuration"""
    
    
    if not config["openai_api_key"]:
        st.error("Please enter your OpenAI API Key in the sidebar.")
        st.stop() 
    st.session_state.openai_api_key = config["openai_api_key"]
    if st.session_state.resume_agent is None:
        st.session_state.resume_agent = ResumeAnalysisAgent(api_key = config["openai_api_key"])
    
    else:
        st.session_state.resume_agent.api_key = config["openai_api_key"]
        
    return st.session_state.resume_agent
#     In Streamlit, every time the user clicks something, uploads a file, or changes input, the script reruns from the top.
# Without st.session_state, your ResumeAnalysisAgent would be created again and again — losing any previous data or state.
def analyze_resume(agent, resume_file, role, custom_jd):
    if not resume_file:
        st.error("Please upload a resume.")
        return

    with st.spinner("Analyzing resume... This may take a minute."):
        try:
            result = agent.analyze_resume(
                resume_file,
                custom_jd=custom_jd if custom_jd else None,
                role_requirements=ROLE_REQUIREMENTS.get(role) if not custom_jd else None
            )
            st.session_state.resume_analyzed = True
            st.session_state.analysis_result = result
            return result
        except Exception as e:
            st.error(f"Error analyzing resume: {e}")



def ask_question(agent , question)    :
    """Ask a question about the resume"""
    try:
        with st.spinner("Generating response..."):
            response = agent.ask_question(question)
            return response
        
    except Exception as e:
        return f"Error:{e}"    
    
def generate_interview_questions(agent , question_types , difficulty , num_questions):
        """Generate intervies question based on the resume"""
        
        try:
            with st.spinner("Generating pesonalized interview questions..."):
                questions = agent.generate_interview_questions(question_types , difficulty , num_questions)
                return questions
            
        except Exception as e:
            st.error(f"Error Generating questions:{e}")     
            return []
        


def improve_resume(agent , improvement_areas , target_role):
    """"Generate resume improvement suggestions"""
    
    try:
        with st.spinner("Analyzing and generating improvements..."):
            return agent.improve_resume(improvement_areas , target_role)
        
    except Exception as e:
        st.error(f"Error generating improvements:{e}")    
        return {}
    
def get_improved_resume(agent , target_role , highlight_skills)    :
    """Get an improved version of the resume"""
    try:
        with st.spinner("Creating improved resume..."):
            return agent.get_improved_resume(target_role , highlight_skills)
        
    except Exception as e:
        st.error(f"Error creating resume:{e}")    
        return "Error generating improved resume."


# NEW: Mock Interview helpers
# ==========================

def _get_client(api_key: str):
    if OpenAI is None:
        raise RuntimeError("openai package not found. Install with: pip install openai>=1.0.0")
    return OpenAI(api_key=api_key)


def synthesize_speech(text: str, api_key: str) -> bytes:
    """Use OpenAI TTS to synthesize the given text and return mp3 bytes."""
    client = _get_client(api_key)
    # model names may vary; gpt-4o-mini-tts is widely available
    resp = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        response_format="mp3",
    )
    # SDK returns bytes-like object
    audio_bytes = resp.read() if hasattr(resp, 'read') else resp
    return audio_bytes


def transcribe_audio(audio_file, api_key: str) -> str:
    """Transcribe user audio (UploadedFile) to text using Whisper-1."""
    client = _get_client(api_key)
    # audio_file is a Streamlit UploadedFile; get bytes and name
    data = audio_file.read()
    # reset pointer for any future reads
    audio_file.seek(0)
    # Best-effort mime; Streamlit returns .wav by default for audio_input
    fname = getattr(audio_file, 'name', 'answer.wav')
    mime = getattr(audio_file, 'type', 'audio/wav')
    file_tuple = (fname, io.BytesIO(data), mime)
    tr = client.audio.transcriptions.create(
        model="whisper-1",
        file=file_tuple,
        response_format="json"
    )
    # SDK returns object with .text or dict
    text = getattr(tr, 'text', None)
    if text is None:
        # fallback if dict-like
        try:
            text = tr["text"]
        except Exception:
            text = ""
    return text or ""


def score_answer(question: str, transcript: str, api_key: str) -> dict:
    """Score an answer across categories using an LLM and return a JSON dict."""
    client = _get_client(api_key)
    sys = (
        "You are an expert technical interviewer. Score the candidate's single answer strictly. "
        "Return ONLY compact JSON with keys: communication (0-10), technical_knowledge (0-10), "
        "problem_solving (0-10), overall (0-100), strengths (array of short phrases), "
        "weaknesses (array of short phrases), feedback (<=60 words)."
    )
    user = (
        f"Question: {question}\n\n"
        f"Candidate answer (transcript): {transcript}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        data = {
            "communication": 5,
            "technical_knowledge": 5,
            "problem_solving": 5,
            "overall": 50,
            "strengths": [],
            "weaknesses": [],
            "feedback": "",
        }
    return data


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
        }


def process_answer(audio_file):
    """Handle one answer: STT -> scoring -> advance pointer."""
    api_key = st.session_state.openai_api_key
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


    
    
def cleanup():
    """clean up resources when the app exits"""
    if st.session_state.resume_agent:
        st.session_state.resume_agent.cleanup()
       
atexit.register(cleanup)        

def main():
    ui.setup_page()
    ui.display_header()
    
    config = ui.setup_sidebar() # config = openai api key
    agent  = setup_agent(config)
    
    tabs  = ui.create_tabs()
    
    with tabs[0]:
        role , custom_jd = ui.role_selection_section(ROLE_REQUIREMENTS)
        uploaded_resume = ui.resume_upload_section()
        
        col1 , col2 , col3 = st.columns([1 , 1 , 1])
        with col2:
            if st.button("Analyze Resume", type="primary"):
                if agent and uploaded_resume:
                    result = analyze_resume(agent, uploaded_resume, role, custom_jd)
                    if result:  # Only update if successful
                        st.session_state.analysis_result = result
                        st.session_state.resume_analyzed = True
                        st.rerun()
                    
            
        if st.session_state.analysis_result: 
            ui.display_analysis_results(st.session_state.analysis_result)
            
    
    with tabs[1]        :
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.resume_qa_section(
                has_resume  = True,
                ask_question_func = lambda q: ask_question(st.session_state.resume_agent , q)
                
            )
        else:
            st.warning("Please upload and anlyze a resume first in the 'Resume Analysis' tab.")
    
    
    with tabs[2]         :
        
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.interview_questions_section(
                has_resume = True,
                generate_questions_func = lambda types , diff , num:
                    generate_interview_questions(st.session_state.resume_agent , types , diff , num)
            )
        else:
            st.warning("please upload and analyze a resume first in the 'Resume Analysis' tab.")    
            
    
    with tabs[3]  :
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.resume_improvement_section(
                has_resume = True,
                improve_resume_func = lambda areas , role: improve_resume(st.session_state.resume_agent , areas , role)
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")    
            
    
    
    with tabs[4] :
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.improved_resume_section(
            has_resume=True,
            get_improved_resume_func=lambda role, skills: get_improved_resume(st.session_state.resume_agent, role, skills)
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis tab.")    
    
    
    with tabs[5] :
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.mock_interview_section(
                has_resume=True,
                start_interview_func=lambda types, diff, num: start_mock_interview(st.session_state.resume_agent, types, diff, num),
                play_tts_func=lambda text: synthesize_speech(text, st.session_state.openai_api_key),
                process_audio_answer_func=lambda audio_file: process_answer(audio_file),
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")

    with tabs[6]:   # Job Search tab
        ui.job_search_ui() 

  

if __name__ == "__main__"            :
    main()
    
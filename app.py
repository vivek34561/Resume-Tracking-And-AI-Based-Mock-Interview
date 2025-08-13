import streamlit as st

st.set_page_config(
    page_title = "Recruitment Agent",
    page_icon = "🚀",
    layout  = "wide"
)

import ui
from agents import ResumeAnalysisAgent
import atexit

ROLE_REQUIREMENTS = {
    "AI/ML Engineer": [ 
        "Python", "PyTorch", "TensorFlow", "Machine Learning", "Deep Learning", "LangChain",
        "MLOps", "Scikit-learn", "Natural Language Processing (NLP)", "Computer Vision", "Hugging Face", 
        "Feature Engineering",
        "SQL", "Data Preprocessing", "Model Deployment", "Docker", "Kubernetes",
        "AWS SageMaker", "Azure ML", "Git", "Data Analysis",
        "Transfer Learning", "Hyperparameter Tuning",
        "Model Monitoring", "Experiment Tracking (MLflow, DVC)"

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
]
}

if 'resume_agent' not in st.session_state:
    st.session_state.resume_agent = None
if 'resume_analyzed' not in st.session_state:
    st.session_state.resume_analyzed = False
    
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
    


def setup_agent(config)    :
    """set up the resume analysis agent with the provided configuration"""
    
    
    if not config["openai_api_key"]:
        st.error("Please enter your OpenAI API Key in the sidebar.")
        st.stop() 
    
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
    
    
def cleanup()    :
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
            
    
    with tabs[3]        :
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.resume_improvement_section(
                has_resume = True,
                improve_resume_func = lambda areas , role: improve_resume(st.session_state.resume_agent , areas , role)
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")    
            
    
    
    with tabs[4]        :
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.improved_resume_section(
            has_resume=True,
            get_improved_resume_func=lambda role, skills: get_improved_resume(st.session_state.resume_agent, role, skills)
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis tab.")    
            

if __name__ == "__main__"            :
    main()
    
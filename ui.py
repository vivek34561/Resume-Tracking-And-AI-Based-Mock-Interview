import streamlit as st
import json
from typing import List, Tuple, Dict, Callable, Optional

def setup_page():
    """Set up the page layout and styles"""
    st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .stTextArea textarea {
            min-height: 150px;
        }
        .section {
            margin-bottom: 2rem;
        }
        .skill-score {
            padding: 0.5rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .good-score {
            background-color: #e6f7e6;
        }
        .medium-score {
            background-color: #fff8e6;
        }
        .low-score {
            background-color: #ffebeb;
        }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display the application header"""
    st.title("🚀AI-Powered Interview prep tool") 
    st.markdown("""
    **AI-powered resume analysis and interview preparation tool**
    
    Upload a resume, select a target role or paste job description, and get personalized feedback to improve your job application.
    """)
    st.divider()

def setup_sidebar() -> Dict:
    """Set up the sidebar configuration options"""
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter Your OpenAI API Key", type="password", help="Required for all AI functionality")
        
        st.markdown("---")
        st.markdown("""
        **How to use:**
        1. Enter your OpenAI API key
        2. Upload your resume
        3. Select target role or paste job description
        4. Click 'Analyze Resume'
        5. Explore the different tabs for insights
        """)
        
        st.markdown("---")
        st.markdown("""
        **Note:** This tool uses AI to analyze resumes and generate suggestions. 
        Always review the outputs carefully before using them.
        """)
        
    return {
        "openai_api_key": api_key
    }

def create_tabs() -> List:
    """Create the main navigation tabs"""
    return st.tabs([
        "📝 Resume Analysis",
        "❓ Resume Q&A",
        "💡 Interview Questions",
        "🔧 Resume Improvements",
        "✨ Improved Resume",
        "🎤 Mock Interview",
        "🔍 Job Search" 
    ])

def role_selection_section(role_requirements: Dict) -> Tuple[str, Optional[str]]:
    """Create the role selection section"""
    st.subheader("Target Role Selection")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        role = st.selectbox(
            "Select Target Role",
            options=list(role_requirements.keys()),
            index=0
        )
    
    with col2:
        st.markdown("**OR**")
        custom_jd = st.text_area(
            "Paste Job Description (overrides role selection)",
            height=200,
            placeholder="Paste the full job description here if you want analysis against a specific job..."
        )
    
    return role, custom_jd if custom_jd else None

def resume_upload_section():
    st.header("📄 Upload Your Resume")

    # Ensure label is a string to prevent TypeError
    uploaded_file = st.file_uploader(
        label="Upload your resume (PDF or TXT)",
        type=["pdf", "txt"]
    )

    if uploaded_file is not None:
        file_details = {
            "filename": uploaded_file.name,
            "filetype": uploaded_file.type,
            "filesize": uploaded_file.size
        }
        st.write(file_details)
        return uploaded_file
    else:
        return None

def display_analysis_results(analysis_result: Dict):
    """Display the resume analysis results"""
    st.subheader("Analysis Results")
    
    # Overall score
    score = analysis_result.get("overall_score", 0)
    score_color = "green" if score >= 75 else "orange" if score >= 50 else "red"
    
    st.metric("Overall Match Score", f"{score}%", help="How well your resume matches the target role requirements")
    st.progress(score / 100)
    
    # Strengths and weaknesses
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Strengths")
        if analysis_result.get("strengths"):
            for strength in analysis_result.get("strengths", []):
                st.markdown(f"- {strength}")
        else:
            st.info("No notable strengths identified")
    
    with col2:
        st.subheader("⚠️ Areas for Improvement")
        if analysis_result.get("missing_skills"):
            for skill in analysis_result.get("missing_skills", []):
                st.markdown(f"- {skill}")
        else:
            st.info("No major improvement areas identified")
    
    # Detailed skill scores
    st.subheader("Detailed Skill Analysis")
    if analysis_result.get("skill_scores"):
        for skill, score in analysis_result.get("skill_scores", {}).items():
            reasoning = analysis_result.get("skill_reasoning", {}).get(skill, "")
            
            if score >= 7:
                score_class = "good-score"
                emoji = "✅"
            elif score >= 4:
                score_class = "medium-score"
                emoji = "⚠️"
            else:
                score_class = "low-score"
                emoji = "❌"
            
            with st.expander(f"{emoji} {skill} - Score: {score}/10"):
                st.markdown(f"**Assessment:** {reasoning}")
                st.markdown(f"""<div class="skill-score {score_class}">
                    Score: {score}/10 - {get_score_description(score)}
                </div>""", unsafe_allow_html=True)
    else:
        st.info("No detailed skill analysis available")

def get_score_description(score: int) -> str:
    """Get descriptive text for a skill score"""
    if score >= 9:
        return "Excellent demonstration of this skill"
    elif score >= 7:
        return "Good demonstration of this skill"
    elif score >= 5:
        return "Moderate demonstration - could be improved"
    elif score >= 3:
        return "Weak demonstration - needs improvement"
    else:
        return "Very weak or missing - significant improvement needed"

def resume_qa_section(has_resume: bool, ask_question_func: Callable):
    """Create the resume Q&A section"""
    st.subheader("Ask Questions About Your Resume")
    
    if has_resume:
        question = st.text_input(
            "Ask any question about your resume content",
            placeholder="What technologies has this candidate worked with?"
        )
        
        if question:
            response = ask_question_func(question)
            st.markdown("**Response:**")
            st.write(response)
    else:
        st.warning("Please upload and analyze a resume first")

def interview_questions_section(has_resume: bool, generate_questions_func: Callable):
    """Create the interview questions section"""
    st.subheader("Generate Personalized Interview Questions")
    
    if has_resume:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            question_types = st.multiselect(
                "Question Types",
                options=["Technical", "Behavioral", "Situational", "System Design", "Problem Solving"],
                default=["Technical", "Behavioral"]
            )
        
        with col2:
            difficulty = st.selectbox(
                "Difficulty Level",
                options=["Easy", "Medium", "Hard", "Mixed"],
                index=1
            )
        
        with col3:
            num_questions = st.slider(
                "Number of Questions",
                min_value=5,
                max_value=20,
                value=10,
                step=1
            )
        
        if st.button("Generate Questions", type="primary"):
            questions = generate_questions_func(question_types, difficulty, num_questions)
            
            if questions:
                st.markdown("### Generated Questions")
                for i, q in enumerate(questions, 1): # FIX: Iterate over dictionaries, not tuples
                    q_type = q.get("type", "N/A")
                    question = q.get("question", "N/A")
                    with st.expander(f"{i}. {question}"):
                        st.markdown(f"**Type:** {q_type}")
                        st.markdown("**Suggested Approach:**")
                        st.info("Think of specific examples from your experience that demonstrate this skill or situation")
            else:
                st.warning("No questions were generated. Please try again with different parameters.")
    else:
        st.warning("Please upload and analyze a resume first")

def resume_improvement_section(has_resume: bool, improve_resume_func: Callable):
    """Create the resume improvement suggestions section"""
    st.subheader("Get Resume Improvement Suggestions")
    
    if has_resume:
        target_role = st.text_input(
            "Target Role (optional, helps tailor suggestions)",
            placeholder="e.g. Senior AI Engineer"
        )
        
        improvement_areas = st.multiselect(
            "Select areas to improve",
            options=["Skills Highlighting", "Work Experience", "Education", "Projects", "Achievements", "Overall Structure"],
            default=["Skills Highlighting", "Work Experience"]
        )
        
        if st.button("Get Improvement Suggestions", type="primary"):
            improvements = improve_resume_func(improvement_areas, target_role)
            
            if improvements:
                for area, details in improvements.items():
                    with st.expander(f"Improvements for {area}"):
                        st.markdown(f"**{details.get('description', '')}**")
                        
                        st.markdown("**Specific Suggestions:**")
                        for suggestion in details.get("specific", []):
                            st.markdown(f"- {suggestion}")
                        
                        if "before_after" in details:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Current Version:**")
                                st.code(details["before_after"]["before"], language="text")
                            
                            with col2:
                                st.markdown("**Suggested Improvement:**")
                                st.code(details["before_after"]["after"], language="text")
            else:
                st.warning("No improvement suggestions were generated. Please try again.")
    else:
        st.warning("Please upload and analyze a resume first")

def improved_resume_section(has_resume: bool, get_improved_resume_func: Callable):
    """Create the improved resume generation section"""
    st.subheader("Generate Improved Resume Version")
    
    if has_resume:
        target_role = st.text_input(
            "Target Role for Resume (optional)",
            placeholder="e.g. Machine Learning Engineer",
            key="improved_role"
        )
        
        highlight_skills = st.text_area(
            "Skills to Highlight (comma separated, optional)",
            placeholder="python, machine learning, data analysis",
            height=100,
            key="highlight_skills"
        )
        
        if st.button("Generate Improved Resume", type="primary"):
            improved_resume = get_improved_resume_func(target_role, highlight_skills)
            
            if improved_resume and not improved_resume.startswith("Error"):
                st.markdown("### Improved Resume")
                st.text_area(
                    "Improved Resume Content",
                    improved_resume,
                    height=600,
                    key="improved_resume_content"
                )
                
                st.download_button(
                    label="Download Improved Resume",
                    data=improved_resume,
                    file_name="improved_resume.txt",
                    mime="text/plain"
                )
            else:
                st.error(improved_resume)
    else:
        st.warning("Please upload and analyze a resume first")
        


def mock_interview_section(
    has_resume: bool,
    start_interview_func: Callable,
    play_tts_func: Callable[[str], bytes],
    process_audio_answer_func: Callable,
):
    st.subheader("Interactive Mock Interview (10 Questions)")
    st.caption("We speak each question. You record your answer, we transcribe and score it across Communication, Technical Knowledge, and Problem Solving. At the end you get a summary with strengths and weaknesses.")

    if not has_resume:
        st.warning("Please upload and analyze a resume first")
        return

    inter = st.session_state.interview

    if not inter['started'] and not inter['completed']:
        c1, c2, c3 = st.columns(3)
        with c1:
            q_types = st.multiselect(
                "Question Types",
                options=["Technical", "Behavioral", "Situational", "System Design", "Problem Solving"],
                default=["Technical", "Behavioral", "Problem Solving"],
            )
        with c2:
            diff = st.selectbox("Difficulty", ["Easy", "Medium", "Hard", "Mixed"], index=2)
        with c3:
            num = st.number_input("Questions", min_value=5, max_value=15, value=10, step=1)
        if st.button("Start Mock Interview", type="primary"):
            start_interview_func(q_types, diff, int(num))
            st.rerun()
        return

    # Interview running
    if inter['started'] and not inter['completed']:
        q_idx = inter['current']
        total = len(inter['questions'])
        st.progress(q_idx / total if total else 0.0, text=f"Question {q_idx+1} of {total}")
        question = inter['questions'][q_idx]
        st.markdown(f"### Q{q_idx+1}. {question}")

        # TTS playback
        if st.button("▶️ Play question"):
            try:
                audio_bytes = play_tts_func(question)
                st.audio(audio_bytes, format='audio/mp3')
            except Exception as e:
                st.error(f"TTS error: {e}")

        # Record answer (requires Streamlit >= 1.30 with st.audio_input)
        audio = st.audio_input("Record your answer")
        submit_col, skip_col = st.columns([3,1])
        with submit_col:
            if st.button("Submit Answer", disabled=(audio is None), type="primary"):
                if audio is None:
                    st.warning("Please record your answer first.")
                else:
                    process_audio_answer_func(audio)
                    st.rerun()
        with skip_col:
            if st.button("Skip"):
                # Append blanks to keep alignment
                inter['answers'].append(None)
                inter['transcripts'].append("")
                inter['per_q_scores'].append({"communication":0,"technical_knowledge":0,"problem_solving":0,"overall":0,"strengths":[],"weaknesses":[],"feedback":""})
                inter['current'] += 1
                if inter['current'] >= len(inter['questions']):
                    inter['completed'] = True
                st.rerun()

        # If we already have a score for previous question, show it quickly
        if inter['per_q_scores'] and q_idx > 0:
            last = inter['per_q_scores'][q_idx-1]
            with st.expander("Previous question feedback"):
                cols = st.columns(4)
                cols[0].metric("Communication", last.get('communication',0))
                cols[1].metric("Technical", last.get('technical_knowledge',0))
                cols[2].metric("Problem Solving", last.get('problem_solving',0))
                cols[3].metric("Overall", last.get('overall',0))
                st.markdown(f"**Feedback:** {last.get('feedback','')}")
                if last.get('strengths'):
                    st.markdown("**Strengths:** " + ", ".join(last['strengths']))
                if last.get('weaknesses'):
                    st.markdown("**Weaknesses:** " + ", ".join(last['weaknesses']))
        return

    # Completed
    if inter['completed']:
        st.success("Mock interview completed!")
        summary = inter.get('summary')
        if summary:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall", summary['overall'])
            c2.metric("Communication", summary['communication'])
            c3.metric("Technical", summary['technical_knowledge'])
            c4.metric("Problem Solving", summary['problem_solving'])
            st.progress(summary['overall'] / 100.0)

            colA, colB = st.columns(2)
            with colA:
                st.subheader("Top Strengths")
                if summary['strengths']:
                    for s in summary['strengths']:
                        st.markdown(f"- {s}")
                else:
                    st.write("—")
            with colB:
                st.subheader("Key Weaknesses")
                if summary['weaknesses']:
                    for w in summary['weaknesses']:
                        st.markdown(f"- {w}")
                else:
                    st.write("—")

        # Detailed per-question review
        st.markdown("### Per-Question Review")
        for i, (q, sc, tr) in enumerate(zip(inter['questions'], inter['per_q_scores'], inter['transcripts'])):
            with st.expander(f"Q{i+1}: {q}"):
                st.markdown(f"**Your answer (transcript):** {tr if tr else '—'}")
                cols = st.columns(4)
                cols[0].metric("Communication", sc.get('communication',0))
                cols[1].metric("Technical", sc.get('technical_knowledge',0))
                cols[2].metric("Problem Solving", sc.get('problem_solving',0))
                cols[3].metric("Overall", sc.get('overall',0))
                st.markdown(f"**Feedback:** {sc.get('feedback','')}")
                if sc.get('strengths'):
                    st.caption("Strengths: " + ", ".join(sc['strengths']))
                if sc.get('weaknesses'):
                    st.caption("Weaknesses: " + ", ".join(sc['weaknesses']))

        if st.button("Restart Interview"):
            inter.update({
                'started': False,
                'completed': False,
                'current': 0,
                'questions': [],
                'answers': [],
                'transcripts': [],
                'per_q_scores': [],
                'summary': None,
            })
            st.rerun()



import streamlit as st
from agents import JobAgent

def job_search_ui():
    st.title("🌍 Recruitment Agent - Job Search")

    # Initialize JobAgent
    agent = JobAgent()

    # Select platform (future proof, currently Adzuna only)
    platform = st.selectbox("Select Job Platform", ["Adzuna" , "Indeed"])

    # Job role input
    query = st.selectbox(
        "Select Role",
        [
            "Data Analyst", "Data Scientist", "Software Engineer", "ML Engineer",
            "Backend Developer", "Frontend Developer", "Full Stack Developer",
            "AI Engineer", "Business Analyst", "DevOps Engineer", "Cloud Engineer",
            "Cybersecurity Specialist", "Product Manager", "QA Engineer"
        ]
    )

    # Location input
    location = st.selectbox(
    "Enter Location (optional)", 
    ["India" , "Delhi", "Bengaluru", "Mumbai", "Hyderabad", "Chennai", "Pune", "Kolkata"])


    # Country selection
    country = st.selectbox("Select Country", ["in", "us", "gb", "ca", "au"])

    # Experience filter
    experience = st.slider("Years of Experience (optional)", 0, 15, 0)

    # Number of results
    num_results = st.slider("Number of Results", 5, 30, 10)

    # Search button
    if st.button("🔍 Search Jobs"):
        with st.spinner("Fetching jobs..."):
            jobs = agent.search_jobs(
                query=query,
                location=location,
                platform=platform.lower(),
                experience=experience if experience > 0 else None,
                num_results=num_results,
                country=country
            )

        # Display results
        if jobs and "error" not in jobs[0]:
            for job in jobs:
                st.markdown(f"### {job['title']}")
                st.write(f"**Company:** {job['company']}")
                st.write(f"**Location:** {job['location']}")
                st.markdown(f"[Apply Here]({job['link']})", unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.error(jobs[0].get("error", "No jobs found."))

import streamlit as st
import os
import json
from typing import List, Tuple, Dict, Callable, Optional
from database import get_user_resumes

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
        # Logged-in user info and logout
        _user = st.session_state.get('user')
        if _user:
            st.markdown(f"Logged in as: **{_user.get('username','')}**")
            if st.button("Logout"):
                st.session_state.user = None
                st.session_state.user_settings = {}
                st.rerun()
        # Settings from user profile
        _user_settings = st.session_state.get('user_settings') or {}
        # Saved resumes for quick reuse (per user)
        saved_resume_id = None
        saved_resumes = []
        if _user:
            try:
                saved_resumes = get_user_resumes(_user['id']) or []
                if saved_resumes:
                    st.markdown("---")
                    st.subheader("Saved Resumes")
                    labels = [f"{r.get('filename') or 'resume'} · {str(r.get('created_at'))[:19]}" for r in saved_resumes]
                    ids = [r['id'] for r in saved_resumes]
                    # Remember last selection if present
                    default_idx = 0
                    if 'selected_resume_id' in st.session_state:
                        try:
                            default_idx = ids.index(st.session_state['selected_resume_id'])
                        except ValueError:
                            default_idx = 0
                    sel = st.selectbox("Use a previously uploaded resume", options=list(range(len(ids))), format_func=lambda i: labels[i], index=default_idx)
                    saved_resume_id = ids[sel]
                    st.session_state['selected_resume_id'] = saved_resume_id
                    st.caption("Tip: Click Analyze in the main page to analyze this saved resume without re-embedding.")
            except Exception as e:
                st.info(f"Could not load saved resumes: {e}")
        # Provider is fixed to Groq
        st.subheader("AI Provider")
        st.caption("Using Groq for all AI features")
        provider = "groq"

        # Groq API key - user input takes priority over .env
        # Don't pre-fill the input field to distinguish user input from .env
        api_key = st.text_input(
            "Enter Your Groq API Key (optional if set in .env)", 
            value="", 
            type="password", 
            help="Leave empty to use GROQ_API_KEY from .env file, or enter your own key here",
            placeholder="gsk_..."
        )
        
        # If user didn't provide a key, fall back to .env
        if not api_key or api_key.strip() == "":
            api_key = os.getenv("GROQ_API_KEY", "")
            if api_key:
                st.caption("✅ Using API key from .env file")
            else:
                st.warning("⚠️ No API key provided. Please enter your Groq API key or set GROQ_API_KEY in .env file")
        else:
            st.caption("✅ Using your provided API key")

        model = None
        # Groq models only
        groq_models = [
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "llama-3.1-8b-instant",
            "Custom...",
        ]
        env_model = _user_settings.get('model') or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"
        default_model_index = groq_models.index(env_model) if env_model in groq_models else groq_models.index("Custom...")
        preset = st.selectbox(
            "Groq preset model",
            options=groq_models,
            index=default_model_index,
            help="Choose a known model or select Custom to provide a model id"
        )
        if preset == "Custom...":
            model = st.text_input(
                "Custom Groq model id",
                value=env_model if env_model not in groq_models else "",
                help="Enter an exact model id available to your Groq account"
            )
        else:
            model = preset
        
        st.markdown("---")
        # Job APIs section (per-user)
        st.subheader("Job APIs")
        existing_jooble = (_user_settings.get('jooble_api_key') if _user_settings else None) or os.getenv("JOOBLE_API_KEY", "")
        jooble_api_key = st.text_input("Jooble API Key (per user)", value=existing_jooble, type="password", help="Used for Jooble job search. Stored in your user settings.")

        st.markdown("---")
        st.markdown("""
        **How to use:**
        1. Enter your API key (auto-filled from .env if present)
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
        
    config = {
        "provider": provider,
        "api_key": api_key,
        "model": model,
        "jooble_api_key": jooble_api_key,
    }
    # Provider is fixed to Groq; no Ollama config
    if saved_resume_id:
        config["selected_resume_id"] = saved_resume_id
    return config

def create_tabs() -> List:
    """Create the main navigation tabs"""
    return st.tabs([
        "📝 Resume Analysis",
        "💬 Resume Chatbot",
        "💡 Interview Questions",
        "✨ Resume Improvement",
        "✉️ Cover Letter",
        "🔍 Job Search",
        "🎤 Mock Interview",
        "🧩 LaTeX Resume Update"
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
    st.header("📄 Resume Selection")
    
    # Check if user has saved resumes
    _user = st.session_state.get('user')
    has_saved_resumes = False
    saved_resumes = []
    
    if _user:
        try:
            saved_resumes = get_user_resumes(_user['id']) or []
            has_saved_resumes = len(saved_resumes) > 0
        except Exception:
            pass
    
    # Option selector
    if has_saved_resumes:
        resume_source = st.radio(
            "Choose resume source:",
            ["📤 Upload New Resume", "💾 Use Saved Resume"],
            horizontal=True,
            help="Select whether to upload a new resume or use a previously saved one"
        )
    else:
        resume_source = "📤 Upload New Resume"
        if _user:
            st.info("💡 Tip: After analyzing a resume, it will be saved for quick reuse!")
    
    uploaded_file = None
    
    if resume_source == "📤 Upload New Resume":
        # Ensure label is a string to prevent TypeError
        uploaded_file = st.file_uploader(
            label="Upload your resume (PDF or TXT)",
            type=["pdf", "txt"],
            key="new_resume_upload"
        )

        if uploaded_file is not None:
            file_details = {
                "filename": uploaded_file.name,
                "filetype": uploaded_file.type,
                "filesize": uploaded_file.size
            }
            st.write(file_details)
            # Clear any previously selected saved resume
            if 'use_saved_resume_id' in st.session_state:
                del st.session_state['use_saved_resume_id']
            return uploaded_file
    
    else:  # Use Saved Resume
        st.markdown("**Select a saved resume:**")
        
        # Create a more readable list
        resume_options = []
        resume_ids = []
        for r in saved_resumes:
            filename = r.get('filename') or 'Resume'
            created = str(r.get('created_at', ''))[:19]
            resume_options.append(f"{filename} (saved on {created})")
            resume_ids.append(r['id'])
        
        selected_idx = st.selectbox(
            "Saved Resumes",
            options=range(len(resume_options)),
            format_func=lambda i: resume_options[i],
            key="saved_resume_selector"
        )
        
        if selected_idx is not None and selected_idx < len(resume_ids):
            selected_resume_id = resume_ids[selected_idx]
            st.session_state['use_saved_resume_id'] = selected_resume_id
            
            # Show resume details
            selected_resume = saved_resumes[selected_idx]
            st.success(f"✅ Using saved resume: **{selected_resume.get('filename', 'Resume')}**")
            
            # Show a preview if available
            resume_text = selected_resume.get('resume_text', '')
            if resume_text:
                with st.expander("📄 Preview Resume Content"):
                    st.text_area("Resume Preview", resume_text[:500] + "..." if len(resume_text) > 500 else resume_text, height=150, disabled=True)
            
            # Return a marker indicating saved resume is being used
            return "USE_SAVED_RESUME"
    
    return None

def display_analysis_results(analysis_result: Dict):
    """Display the resume analysis results"""
    st.subheader("Analysis Results")
    
    # Overall score
    score = analysis_result.get("overall_score", 0)
    score_color = "green" if score >= 75 else "orange" if score >= 50 else "red"
    
    st.metric("Overall Match Score", f"{score}%", help="How well your resume matches the target role requirements")
    st.progress(score / 100)
    
    # Application Status based on score
    st.markdown("---")
    if score >= 80:
        st.success(f"""
        ### ✅ **Highly Qualified - Strong Match!**
        
        **Recommendation:** You are an excellent candidate for this role.
        - Your profile strongly aligns with the job requirements
        - You exceed most of the required qualifications
        - **Apply with confidence!** You have a high chance of success
        - Highlight your top matching skills in your application
        """)
    elif score >= 70:
        st.success(f"""
        ### ✅ **Well Qualified - Good Match**
        
        **Recommendation:** You are a strong candidate for this position.
        - Your qualifications align well with most requirements
        - You meet the core competencies for this role
        - **You should apply!** Good chance of getting shortlisted
        - Emphasize your relevant experience and skills
        """)
    elif score >= 50:
        st.info(f"""
        ### 📋 **Qualified - You Can Apply**
        
        **Recommendation:** You meet the minimum requirements.
        - You satisfy the basic qualifications for this role
        - Review the "Areas for Improvement" section below
        - **You can apply**, but consider:
          - Tailoring your resume to highlight relevant skills
          - Addressing skill gaps mentioned below
          - Showcasing transferable skills and achievements
        - 💪 Work on improving the missing skills to strengthen your application
        """)
    else:
        st.warning(f"""
        ### ⚠️ **Below Threshold - Consider Upskilling**
        
        **Recommendation:** Your profile needs improvement for this specific role.
        -  Significant skill gaps exist between your profile and job requirements
        -  **Consider upskilling** in the areas mentioned below before applying
        -  You may want to:
          - Take relevant courses or certifications
          - Gain practical experience in missing skills
          - Look for entry-level or related positions first
        - 💡 Focus on the "Areas for Improvement" to increase your chances
        """)
    st.markdown("---")
    
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
    """Create the RAG-based chatbot section for resume questions"""
    st.subheader("💬 Resume AI Chatbot")
    st.markdown("Ask me anything about the resume! I have complete knowledge of the candidate's experience, skills, and background.")
    
    if has_resume:
        # Initialize chat history in session state if not exists
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Suggested questions
        with st.expander("💡 Suggested Questions", expanded=False):
            st.markdown("""
            **Experience & Skills:**
            - What technologies has this candidate worked with?
            - What are the candidate's strongest skills?
            - Tell me about their most recent work experience
            
            **Projects & Achievements:**
            - What notable projects has the candidate completed?
            - What achievements are highlighted in the resume?
            - Has the candidate worked with [specific technology]?
            
            **General:**
            - Summarize this candidate's background
            - What makes this candidate unique?
            - Is this candidate suitable for a [role name] position?
            """)
        
        # Chat container with fixed height and scrolling
        chat_container = st.container(height=500)
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(message['content'])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(message['content'])
        
        # Chat input
        user_question = st.chat_input("Ask a question about the resume...")
        
        if user_question:
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_question
            })
            
            # Get bot response with chat history context
            with st.spinner("🤔 Thinking..."):
                # Pass chat history (excluding the just-added user message for context)
                response = ask_question_func(user_question, st.session_state.chat_history[:-1])
            
            # Add bot response to history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
            
            # Rerun to update chat display
            st.rerun()
        
        # Clear chat button
        if st.session_state.chat_history:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🗑️ Clear Chat History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
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
                options=["Technical", "Projects" , "Behavioral", "Situational", "System Design", "Problem Solving"],
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
                for i, q in enumerate(questions, 1):
                    question = q.get("question", "N/A")
                    solution = q.get("solution") or ""
                    with st.expander(f"{i}. {question}"):
                        if solution:
                            st.markdown("**Solution:**")
                            st.write(solution)
                        else:
                            st.info("No solution generated for this question.")
            else:
                st.warning("No questions were generated. Please try again with different parameters.")
    else:
        st.warning("Please upload and analyze a resume first")

# Removed Resume Improvements and Improved Resume sections per request
        
def cover_letter_section(has_resume: bool, generate_cover_letter_func: Callable):
    """Create the cover letter generation section"""
    st.subheader("Generate a Tailored Cover Letter")

    if has_resume:
        col1, col2 = st.columns(2)
        with col1:
            company = st.text_input("Company Name", placeholder="e.g. Acme Corp")
            role = st.text_input("Role / Position", placeholder="e.g. Machine Learning Engineer")
            tone = st.selectbox("Tone", ["professional", "enthusiastic", "confident", "concise"], index=0)
        with col2:
            length = st.selectbox("Length", ["short (~250-300 words)", "one-page (~400-500 words)"], index=0)
        jd = st.text_area(
            "Paste Job Description (optional)",
            height=200,
            placeholder="Paste the full JD to tailor the letter more precisely"
        )

        if st.button("Generate Cover Letter", type="primary"):
            if not company or not role:
                st.warning("Please provide both company and role.")
            else:
                with st.spinner("Writing your letter..."):
                    letter = generate_cover_letter_func(company, role, jd, tone, "one-page" if length.startswith("one-page") else "short")
                if letter and not letter.startswith("Error"):
                    st.markdown("### Cover Letter")
                    st.text_area("Letter", letter, height=500, key="cover_letter_output")
                    st.download_button(
                        label="Download Cover Letter",
                        data=letter,
                        file_name="cover_letter.txt",
                        mime="text/plain"
                    )
                else:
                    st.error(letter or "No letter generated.")
    else:
        st.warning("Please upload and analyze a resume first")

def latex_resume_update_section(has_resume: bool, update_latex_func: Callable):
    """Section to update LaTeX resume content based on the job description while preserving LaTeX format"""
    st.subheader("Update Your LaTeX Resume for a Specific JD")

    st.caption("Paste your current LaTeX resume code and the target JD. We'll tailor the content while keeping your LaTeX format unchanged.")

    col1, col2 = st.columns(2)
    with col1:
        latex_src = st.text_area(
            "Your LaTeX Resume Source",
            height=350,
            placeholder="\\documentclass{resume}\n% ... preamble ...\n\\begin{document}\n% ... your resume ...\n\\end{document}"
        )
    with col2:
        jd = st.text_area(
            "Job Description",
            height=350,
            placeholder="Paste the full job description here"
        )

    if st.button("Update LaTeX Resume", type="primary"):
        if not latex_src or not latex_src.strip():
            st.warning("Please paste your LaTeX resume source.")
        elif not jd or not jd.strip():
            st.warning("Please paste the job description.")
        else:
            with st.spinner("Updating LaTeX resume..."):
                updated = update_latex_func(latex_src, jd)
            if updated and not updated.startswith("Error"):
                st.markdown("### Updated LaTeX Resume")
                st.text_area("LaTeX Output", updated, height=500, key="updated_latex_output")
                st.download_button(
                    label="Download Updated LaTeX",
                    data=updated,
                    file_name="updated_resume.tex",
                    mime="text/plain"
                )
            else:
                st.error(updated or "No updated LaTeX generated.")
        

def mock_interview_section(
    has_resume: bool,
    start_interview_func: Callable,
    play_tts_func: Callable[[str], bytes],
    process_audio_answer_func: Callable,
):
    """
    Interactive Mock Interview with:
    - Configurable interview duration (5, 10, 15, 20 min)
    - Real-time timer display
    - Speech-to-text for answers (browser-based)
    - Text-to-speech for questions (browser-based)
    - Immediate AI feedback after each answer
    - End interview button to stop anytime
    - Final summary with scores and suggestions
    """
    st.subheader("🎤 Interactive Mock Interview")
    st.caption("Practice with AI-powered interview simulation. Choose your duration, answer via text or speech, and get instant feedback!")

    if not has_resume:
        st.warning("⚠️ Please upload and analyze a resume first")
        return

    inter = st.session_state.interview

    # Interview Setup Phase
    if not inter['started'] and not inter['completed']:
        st.markdown("### Configure Your Interview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Interview Settings")
            duration = st.selectbox(
                "Interview Duration",
                options=[5, 10, 15, 20, 30],
                index=1,
                help="Select how long you want the interview to be"
            )
            
            q_types = st.multiselect(
                "Question Types",
                options=["Technical", "Behavioral", "Situational", "System Design", "Problem Solving"],
                default=["Technical", "Behavioral", "Problem Solving"],
                help="Select the types of questions you want to practice"
            )
            
            diff = st.selectbox(
                "Difficulty Level",
                ["Easy", "Medium", "Hard", "Mixed"],
                index=1,
                help="Choose the difficulty level of questions"
            )
        
        with col2:
            st.markdown("#### How It Works")
            st.info("""
            **Interview Flow:**
            1. 🎯 AI generates personalized questions
            2. 🔊 Questions are read aloud (optional)
            3. 💬 You answer via text or speech
            4. 🤖 AI provides instant feedback
            5. 📊 Get final summary with scores
            
            **Features:**
            - ⏱️ Timer tracks your progress
            - 🛑 End interview anytime
            - 📈 Real-time scoring
            - 💡 Actionable suggestions
            """)
        
        st.markdown("---")
        
        # Calculate number of questions based on duration
        questions_per_minute = 0.5  # Roughly 2 minutes per question
        estimated_questions = max(3, int(duration * questions_per_minute))
        
        st.info(f"ℹ️ Based on {duration} minutes, you'll get approximately **{estimated_questions} questions**")
        
        if st.button("🚀 Start Mock Interview", type="primary", use_container_width=True):
            if not q_types:
                st.error("Please select at least one question type")
            else:
                # Store duration in session state
                st.session_state.interview['max_duration_sec'] = duration * 60
                st.session_state.interview['estimated_questions'] = estimated_questions
                start_interview_func(q_types, diff, estimated_questions)
                st.rerun()
        return

    # Interview In Progress
    if inter['started'] and not inter['completed']:
        import time
        from datetime import timedelta
        
        # Calculate time elapsed and remaining
        if inter.get('start_time'):
            elapsed_sec = time.time() - inter['start_time']
            remaining_sec = max(0, inter['max_duration_sec'] - elapsed_sec)
            elapsed_str = str(timedelta(seconds=int(elapsed_sec)))
            remaining_str = str(timedelta(seconds=int(remaining_sec)))
        else:
            # Initialize start time if not set
            inter['start_time'] = time.time()
            elapsed_sec = 0
            remaining_sec = inter['max_duration_sec']
            elapsed_str = "0:00:00"
            remaining_str = str(timedelta(seconds=int(remaining_sec)))
        
        # Check if time is up
        if remaining_sec <= 0 and not inter['completed']:
            st.warning("⏰ Time's up! Ending interview...")
            # Force end the interview
            from app import end_interview_now
            end_interview_now()
            st.rerun()
        
        # Top Bar with Timer and Progress
        st.markdown("### Interview in Progress")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            q_idx = inter['current']
            total = len(inter['questions'])
            progress_pct = q_idx / total if total else 0.0
            st.metric("Progress", f"{q_idx}/{total} questions")
            st.progress(progress_pct)
        
        with col2:
            st.metric("Time Elapsed", elapsed_str)
            st.metric("Time Remaining", remaining_str, delta=None)
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🛑 End Interview", type="secondary", use_container_width=True):
                if st.session_state.get('confirm_end', False):
                    from app import end_interview_now
                    end_interview_now()
                    st.rerun()
                else:
                    st.session_state['confirm_end'] = True
                    st.warning("Click again to confirm")
                    time.sleep(1)
                    st.session_state['confirm_end'] = False
        
        st.markdown("---")
        
        # Current Question
        if q_idx < total:
            question_data = inter['questions'][q_idx]
            if isinstance(question_data, dict):
                question = question_data.get('question', str(question_data))
                q_type = question_data.get('type', 'General')
            else:
                question = str(question_data)
                q_type = 'General'
            
            st.markdown(f"### Question {q_idx + 1}")
            st.info(f"**Type:** {q_type}")
            st.markdown(f"### {question}")
            
            # TTS - Read Question Aloud using streamlit-js-eval
            col_tts1, col_tts2 = st.columns([1, 4])
            with col_tts1:
                if st.button("🔊 Read Aloud", use_container_width=True, key=f"tts_{q_idx}"):
                    try:
                        # Clean question text for JavaScript
                        clean_question = question.replace('"', "'").replace('\n', ' ').replace('\r', '')
                        
                        # Use streamlit-js-eval to execute TTS
                        from streamlit_js_eval import streamlit_js_eval
                        tts_script = f"""
                        (function() {{
                            const text = "{clean_question}";
                            const utterance = new SpeechSynthesisUtterance(text);
                            utterance.rate = 0.9;
                            utterance.pitch = 1.0;
                            utterance.volume = 1.0;
                            window.speechSynthesis.cancel(); // Clear any previous speech
                            window.speechSynthesis.speak(utterance);
                            return "Speaking...";
                        }})();
                        """
                        result = streamlit_js_eval(js_expressions=tts_script, key=f"tts_exec_{q_idx}")
                        st.success("🎵 Reading question aloud...")
                    except Exception as e:
                        st.error(f"TTS error: {e}. Please check your browser supports speech synthesis.")
            
            st.markdown("---")
            
            # Answer Input Section
            st.markdown("### Your Answer")
            
            # Tabs for Text vs Speech input
            input_tab1, input_tab2 = st.tabs(["💬 Type Answer", "🎤 Voice Answer"])
            
            user_answer = None
            
            with input_tab1:
                st.caption("Type your answer below:")
                text_answer = st.text_area(
                    "Your response:",
                    height=200,
                    key=f"text_answer_{q_idx}",
                    placeholder="Use the STAR method: Situation, Task, Action, Result..."
                )
                
                if st.button("Submit Text Answer", type="primary", key=f"submit_text_{q_idx}"):
                    if text_answer.strip():
                        user_answer = text_answer
                    else:
                        st.error("Please provide an answer")
            
            with input_tab2:
                st.caption("🎤 Voice Recording - Click 'Start' to begin recording, then 'Stop' when done.")
                
                # Use unique ID for this question
                stt_id = f"stt_{q_idx}"
                
                # Improved Speech recognition HTML component with unique IDs
                speech_html = f"""
                <div id="voice-container-{stt_id}" style="padding: 20px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #f9f9f9; margin-bottom: 10px;">
                    <h4 style="color: #333; margin-top: 0;">🎤 Voice Recording</h4>
                    <button id="startBtn-{stt_id}" onclick="startRecording_{stt_id}()" 
                            style="padding: 12px 24px; margin: 5px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; font-weight: bold;">
                        ▶️ Start Recording
                    </button>
                    <button id="stopBtn-{stt_id}" onclick="stopRecording_{stt_id}()" disabled
                            style="padding: 12px 24px; margin: 5px; background-color: #f44336; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; font-weight: bold; opacity: 0.5;">
                        ⏹️ Stop Recording
                    </button>
                    <p id="status-{stt_id}" style="margin-top: 15px; font-weight: bold; font-size: 16px; color: #666;">Ready to record</p>
                    <textarea id="transcript-{stt_id}" readonly 
                              style="width: 100%; height: 150px; margin-top: 10px; padding: 10px; border: 2px solid #ddd; border-radius: 5px; font-size: 14px; font-family: Arial, sans-serif; resize: vertical;"></textarea>
                    <p style="margin-top: 5px; font-size: 12px; color: #666;">💡 Tip: Speak clearly and at a moderate pace for best results</p>
                </div>
                
                <script>
                (function() {{
                    let recognition_{stt_id};
                    let finalTranscript_{stt_id} = '';
                    let isRecording_{stt_id} = false;
                    
                    const statusEl = document.getElementById('status-{stt_id}');
                    const transcriptEl = document.getElementById('transcript-{stt_id}');
                    const startBtn = document.getElementById('startBtn-{stt_id}');
                    const stopBtn = document.getElementById('stopBtn-{stt_id}');
                    
                    // Check if speech recognition is supported
                    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {{
                        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                        recognition_{stt_id} = new SpeechRecognition();
                        recognition_{stt_id}.continuous = true;
                        recognition_{stt_id}.interimResults = true;
                        recognition_{stt_id}.lang = 'en-US';
                        recognition_{stt_id}.maxAlternatives = 1;
                        
                        recognition_{stt_id}.onstart = function() {{
                            isRecording_{stt_id} = true;
                            statusEl.textContent = '🔴 Recording in progress... Speak now!';
                            statusEl.style.color = '#dc3545';
                            startBtn.disabled = true;
                            startBtn.style.opacity = '0.5';
                            stopBtn.disabled = false;
                            stopBtn.style.opacity = '1.0';
                        }};
                        
                        recognition_{stt_id}.onresult = function(event) {{
                            let interimTranscript = '';
                            for (let i = event.resultIndex; i < event.results.length; i++) {{
                                const transcript = event.results[i][0].transcript;
                                if (event.results[i].isFinal) {{
                                    finalTranscript_{stt_id} += transcript + ' ';
                                }} else {{
                                    interimTranscript += transcript;
                                }}
                            }}
                            transcriptEl.value = finalTranscript_{stt_id} + interimTranscript;
                        }};
                        
                        recognition_{stt_id}.onerror = function(event) {{
                            console.error('Speech recognition error:', event.error);
                            if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {{
                                statusEl.textContent = '❌ Microphone access denied. Please allow microphone permissions.';
                            }} else if (event.error === 'no-speech') {{
                                statusEl.textContent = '⚠️ No speech detected. Please try again.';
                            }} else {{
                                statusEl.textContent = '❌ Error: ' + event.error;
                            }}
                            statusEl.style.color = '#dc3545';
                            isRecording_{stt_id} = false;
                            startBtn.disabled = false;
                            startBtn.style.opacity = '1.0';
                            stopBtn.disabled = true;
                            stopBtn.style.opacity = '0.5';
                        }};
                        
                        recognition_{stt_id}.onend = function() {{
                            if (isRecording_{stt_id}) {{
                                statusEl.textContent = '✅ Recording stopped. Transcript captured above.';
                                statusEl.style.color = '#28a745';
                            }}
                            isRecording_{stt_id} = false;
                            startBtn.disabled = false;
                            startBtn.style.opacity = '1.0';
                            stopBtn.disabled = true;
                            stopBtn.style.opacity = '0.5';
                        }};
                        
                        statusEl.textContent = '✅ Ready to record (microphone access available)';
                        statusEl.style.color = '#28a745';
                    }} else {{
                        statusEl.textContent = '❌ Speech recognition not supported in this browser. Please use Chrome or Edge.';
                        statusEl.style.color = '#dc3545';
                        startBtn.disabled = true;
                        startBtn.style.opacity = '0.5';
                    }}
                    
                    // Global functions for this instance
                    window.startRecording_{stt_id} = function() {{
                        if (recognition_{stt_id} && !isRecording_{stt_id}) {{
                            try {{
                                finalTranscript_{stt_id} = '';
                                transcriptEl.value = '';
                                recognition_{stt_id}.start();
                                console.log('Speech recognition started');
                            }} catch (e) {{
                                console.error('Error starting recognition:', e);
                                statusEl.textContent = '❌ Error starting recording: ' + e.message;
                                statusEl.style.color = '#dc3545';
                            }}
                        }}
                    }};
                    
                    window.stopRecording_{stt_id} = function() {{
                        if (recognition_{stt_id} && isRecording_{stt_id}) {{
                            try {{
                                recognition_{stt_id}.stop();
                                console.log('Speech recognition stopped');
                            }} catch (e) {{
                                console.error('Error stopping recognition:', e);
                            }}
                        }}
                    }};
                }})();
                </script>
                """
                
                st.components.v1.html(speech_html, height=350)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Manual input for speech transcript (as fallback or for copying from above)
                speech_answer = st.text_area(
                    "📝 Copy the transcript from above and paste here, or type your answer:",
                    height=120,
                    key=f"speech_answer_{q_idx}",
                    help="Copy the transcript from the voice recording box above, or type your answer manually",
                    placeholder="Paste your transcript here or type your answer..."
                )
                
                if st.button("Submit Voice Answer", type="primary", key=f"submit_speech_{q_idx}"):
                    if speech_answer.strip():
                        user_answer = speech_answer
                    else:
                        st.error("Please provide a transcript")
            
            # Process the answer if submitted
            if user_answer:
                with st.spinner("🤖 Analyzing your answer..."):
                    # Get the agent and evaluate
                    agent = st.session_state.resume_agent
                    if agent and hasattr(agent, 'evaluate_interview_answer'):
                        feedback = agent.evaluate_interview_answer(question, user_answer)
                        
                        # Store the answer and feedback
                        inter['answers'].append(user_answer)
                        inter['transcripts'].append(user_answer)
                        inter['per_q_scores'].append(feedback)
                        
                        # Display immediate feedback
                        st.markdown("---")
                        st.markdown("### 📊 Immediate Feedback")
                        
                        col_score1, col_score2 = st.columns([1, 3])
                        
                        with col_score1:
                            score = feedback.get('score', 0)
                            st.metric("Score", f"{score}/10")
                            
                            # Visual score indicator
                            if score >= 8:
                                st.success("Excellent! 🌟")
                            elif score >= 6:
                                st.info("Good job! 👍")
                            elif score >= 4:
                                st.warning("Needs work 📝")
                            else:
                                st.error("Keep practicing 💪")
                        
                        with col_score2:
                            st.markdown("**Feedback:**")
                            st.write(feedback.get('feedback', 'No feedback available'))
                            
                            st.markdown("**Suggestion for Improvement:**")
                            st.info(feedback.get('suggestion', 'No specific suggestion'))
                        
                        # Move to next question
                        inter['current'] += 1
                        
                        if inter['current'] >= total:
                            # Interview completed
                            from app import end_interview_now
                            end_interview_now()
                        
                        st.success("✅ Answer recorded! Moving to next question...")
                        time.sleep(2)
                        st.rerun()
        else:
            # Should not reach here, but just in case
            from app import end_interview_now
            end_interview_now()
            st.rerun()
        
        return

    # Interview Completed - Show Summary
    if inter['completed']:
        st.markdown("### 🎉 Interview Complete!")
        st.balloons()
        
        # Overall Statistics
        st.markdown("---")
        st.markdown("### 📊 Performance Summary")
        
        scores = [s.get('score', 0) for s in inter.get('per_q_scores', [])]
        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Score", f"{avg_score:.1f}/10")
            with col2:
                st.metric("Best Answer", f"{max_score}/10")
            with col3:
                st.metric("Questions", len(scores))
            with col4:
                decision = inter.get('decision', False)
                if decision:
                    st.success("✅ PASS")
                else:
                    st.error("❌ NEEDS IMPROVEMENT")
        
        st.markdown("---")
        
        # Detailed Question-by-Question Breakdown
        st.markdown("### 📝 Detailed Breakdown")
        
        for i, (q, ans, feedback) in enumerate(zip(
            inter.get('questions', []),
            inter.get('transcripts', []),
            inter.get('per_q_scores', [])
        )):
            with st.expander(f"Question {i+1}: {q if isinstance(q, str) else q.get('question', 'N/A')}", expanded=False):
                st.markdown(f"**Your Answer:**")
                st.write(ans)
                
                col_a, col_b = st.columns([1, 3])
                
                with col_a:
                    score = feedback.get('score', 0)
                    st.metric("Score", f"{score}/10")
                
                with col_b:
                    st.markdown("**Feedback:**")
                    st.info(feedback.get('feedback', 'N/A'))
                    
                    st.markdown("**Improvement Suggestion:**")
                    st.warning(feedback.get('suggestion', 'N/A'))
        
        st.markdown("---")
        
        # Summary insights
        if inter.get('summary'):
            st.markdown("### 💡 Overall Insights")
            summary = inter['summary']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Strengths:**")
                strengths = summary.get('strengths', 'Good communication')
                st.success(strengths)
            
            with col2:
                st.markdown("**Areas for Improvement:**")
                weaknesses = summary.get('weaknesses', 'Focus on quantifying results')
                st.warning(weaknesses)
        
        st.markdown("---")
        
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔄 Start New Interview", type="primary", use_container_width=True):
                # Reset interview state
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
                st.rerun()
        
        with col_btn2:
            # Export results
            if st.button("📥 Download Report", use_container_width=True):
                import json
                from datetime import datetime
                
                report = {
                    'date': datetime.now().isoformat(),
                    'questions': inter.get('questions', []),
                    'answers': inter.get('transcripts', []),
                    'scores': inter.get('per_q_scores', []),
                    'summary': inter.get('summary', {}),
                    'decision': inter.get('decision', False)
                }
                
                report_json = json.dumps(report, indent=2)
                st.download_button(
                    label="Download JSON Report",
                    data=report_json,
                    file_name=f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )


from agents import JobAgent

def resume_improvement_section(has_resume: bool, improve_resume_func: Callable, get_improved_resume_func: Callable):
    """Create the resume improvement suggestions section"""
    st.subheader("✨ Resume Improvement Suggestions")
    st.markdown("Get AI-powered suggestions to enhance your resume and make it stand out!")
    
    if has_resume:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📋 Improvement Areas")
            improvement_areas = st.multiselect(
                "Select areas to improve",
                options=[
                    "Skills Highlighting",
                    "Work Experience",
                    "Achievements & Impact",
                    "Keywords & ATS Optimization",
                    "Professional Summary",
                    "Education & Certifications",
                    "Formatting & Structure"
                ],
                default=["Skills Highlighting", "Work Experience", "Achievements & Impact"],
                help="Choose specific areas where you want improvement suggestions"
            )
        
        with col2:
            st.markdown("#### 🎯 Target Role (Optional)")
            target_role = st.text_input(
                "Target Role",
                placeholder="e.g., Senior Software Engineer, Data Scientist",
                help="Specify a target role for more focused suggestions"
            )
        
        # Generate Suggestions Button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("🔍 Generate Improvement Suggestions", type="primary", use_container_width=True):
                if improvement_areas:
                    with st.spinner("Analyzing your resume and generating personalized suggestions..."):
                        improvements = improve_resume_func(improvement_areas, target_role or "")
                        st.session_state['improvement_suggestions'] = improvements
                        st.success("✅ Improvement suggestions generated!")
                        st.rerun()
                else:
                    st.warning("Please select at least one improvement area.")
        
        # Display Suggestions if available
        if 'improvement_suggestions' in st.session_state and st.session_state['improvement_suggestions']:
            st.markdown("---")
            st.markdown("### 💡 Your Personalized Improvement Suggestions")
            
            improvements = st.session_state['improvement_suggestions']
            
            for area, details in improvements.items():
                with st.expander(f"**{area}**", expanded=True):
                    if isinstance(details, dict):
                        # Description
                        if 'description' in details and details['description']:
                            st.markdown(f"**Overview:** {details['description']}")
                            st.markdown("")
                        
                        # Specific suggestions
                        if 'specific' in details and details['specific']:
                            st.markdown("**Actionable Suggestions:**")
                            for idx, suggestion in enumerate(details['specific'], 1):
                                st.markdown(f"{idx}. {suggestion}")
                            st.markdown("")
                        
                        # Before/After example
                        if 'before_after' in details and details['before_after']:
                            st.markdown("**Example:**")
                            ba = details['before_after']
                            
                            col_before, col_after = st.columns(2)
                            with col_before:
                                st.markdown("**Before:**")
                                st.info(ba.get('before', 'N/A'))
                            
                            with col_after:
                                st.markdown("**After:**")
                                st.success(ba.get('after', 'N/A'))
                    else:
                        st.markdown(str(details))
            
            # Option to generate improved resume
            st.markdown("---")
            st.markdown("### 📄 Generate Improved Resume")
            st.markdown("Want an AI-rewritten version of your resume incorporating these suggestions?")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                highlight_skills = st.text_area(
                    "Skills to highlight (comma-separated) or paste Job Description",
                    placeholder="Python, Machine Learning, AWS, Docker... OR paste full job description",
                    help="Enter specific skills to emphasize or paste a full job description for targeted optimization"
                )
            
            with col2:
                st.markdown("") # spacing
                st.markdown("") # spacing
                if st.button("✨ Generate Improved Resume", use_container_width=True):
                    with st.spinner("Generating your improved resume..."):
                        improved_resume = get_improved_resume_func(target_role or "", highlight_skills)
                        st.session_state['improved_resume_text'] = improved_resume
                        st.success("✅ Improved resume generated!")
                        st.rerun()
            
            # Display improved resume
            if 'improved_resume_text' in st.session_state and st.session_state['improved_resume_text']:
                st.markdown("---")
                st.markdown("### 📝 Your Improved Resume")
                
                # Display in a nice text area
                st.text_area(
                    "Improved Resume Content",
                    value=st.session_state['improved_resume_text'],
                    height=400,
                    help="Copy this improved version or download it below"
                )
                
                # Download button
                st.download_button(
                    label="📥 Download Improved Resume",
                    data=st.session_state['improved_resume_text'],
                    file_name="improved_resume.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    else:
        st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")


def job_search_ui():
    st.title("🌍 Recruitment Agent - Job Search")

    # Initialize JobAgent
    # Use saved per-user Jooble API key if available
    _us = st.session_state.get('user_settings') or {}
    saved_jooble_key = _us.get('jooble_api_key') or os.getenv("JOOBLE_API_KEY")
    agent = JobAgent(jooble_api_key=saved_jooble_key)

    # Cache wrapper to avoid refetching the same queries repeatedly during reruns
    @st.cache_data(ttl=600, show_spinner=False)
    def _cached_job_search(platform: str, query: str, location: str | None, num_results: int, country: str, experience: int | None, jooble_key: str | None):
        ja = JobAgent(jooble_api_key=jooble_key)
        return ja.search_jobs(
            query=query,
            location=location,
            platform=platform.lower(),
            experience=experience,
            num_results=num_results,
            country=country,
            jooble_api_key=jooble_key,
        )

    # Select platform (future proof, currently Adzuna only)
    platform = st.selectbox("Select Job Platform", ["Adzuna", "Jooble"])

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


    # Country fixed to India
    country = "in"
    # Experience removed; default to 0/None
    experience = 0

    # Number of results
    num_results = st.slider("Number of Results", 5, 30, 10)

    # Search button
    if st.button("🔍 Search Jobs"):
        with st.spinner("Fetching jobs..."):
            # Use cached results keyed by arguments to reduce repeated API calls on rerun
            jobs = _cached_job_search(platform, query, location, num_results, country, None, saved_jooble_key)

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

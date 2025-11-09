"""Agent modules for Resume Tracking and AI Mock Interview system."""

from .resume_analyzer import ResumeAnalyzer
from .interview_agent import InterviewAgent
from .resume_improver import ResumeImprover
from .job_search_agent import JobAgent

# Import LLM functions for backward compatibility
from utils.llm_providers import groq_chat, ollama_chat, SESSION
from utils.text_utils import clamp_text as _clamp_text


class ResumeAnalysisAgent(ResumeAnalyzer):
    """Backward compatible wrapper combining all agent functionality."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize sub-agents
        self._interview_agent = None
        self._improver_agent = None
    
    @property
    def interview_agent(self):
        """Lazy-load interview agent."""
        if self._interview_agent is None:
            self._interview_agent = InterviewAgent(self)
        return self._interview_agent
    
    @property
    def improver_agent(self):
        """Lazy-load improver agent."""
        if self._improver_agent is None:
            self._improver_agent = ResumeImprover(self)
        return self._improver_agent
    
    # Delegate methods to sub-agents for backward compatibility
    def ask_question(self, question):
        """Delegate to interview agent."""
        return self.interview_agent.ask_question(question)
    
    def answer_interview_question(self, question: str) -> str:
        """Delegate to interview agent."""
        return self.interview_agent.answer_interview_question(question)
    
    def generate_interview_questions(self, question_types, difficulty, num_questions):
        """Delegate to interview agent."""
        return self.interview_agent.generate_interview_questions(question_types, difficulty, num_questions)
    
    def improve_resume(self, improvement_areas, target_role=""):
        """Delegate to improver agent."""
        return self.improver_agent.improve_resume(improvement_areas, target_role)
    
    def get_improved_resume(self, target_role="", highlight_skills=""):
        """Delegate to improver agent."""
        return self.improver_agent.get_improved_resume(target_role, highlight_skills)
    
    def generate_cover_letter(self, company: str, role: str, job_description: str = "", 
                            tone: str = "professional", length: str = "one-page") -> str:
        """Delegate to improver agent."""
        return self.improver_agent.generate_cover_letter(company, role, job_description, tone, length)
    
    def generate_updated_resume_latex(self, latex_source: str, job_description: str) -> str:
        """Delegate to improver agent."""
        return self.improver_agent.generate_updated_resume_latex(latex_source, job_description)
    
    def cleanup(self):
        """Delegate to improver agent."""
        return self.improver_agent.cleanup()


__all__ = [
    'ResumeAnalyzer',
    'InterviewAgent', 
    'ResumeImprover',
    'JobAgent',
    'ResumeAnalysisAgent',  # Backward compatibility
    # LLM functions for backward compatibility
    'groq_chat',
    'ollama_chat',
    'SESSION',
    '_clamp_text',
]

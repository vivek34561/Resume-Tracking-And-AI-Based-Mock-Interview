"""Backward compatibility wrapper for agents module.

DEPRECATED: This file is maintained for backward compatibility only.
Please import from the new modular structure instead:
- agents.resume_analyzer.ResumeAnalyzer
- agents.interview_agent.InterviewAgent  
- agents.resume_improver.ResumeImprover
- agents.job_search_agent.JobAgent
"""

# Import from new structure
from utils.llm_providers import groq_chat, ollama_chat, SESSION
from utils.text_utils import clamp_text as _clamp_text
from agents import ResumeAnalysisAgent, JobAgent

# Re-export
__all__ = ['groq_chat', 'ollama_chat', 'SESSION', '_clamp_text', 'ResumeAnalysisAgent', 'JobAgent']

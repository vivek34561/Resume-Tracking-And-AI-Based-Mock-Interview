"""Interview Agent - Q&A and interview question generation."""

import re
import json
from utils.text_utils import clamp_text


class InterviewAgent:
    """Handles resume Q&A and interview question generation."""
    
    def __init__(self, resume_analyzer):
        """Initialize with a ResumeAnalyzer instance."""
        self.analyzer = resume_analyzer
    
    def ask_question(self, question, chat_history=None):
        """Answer questions about the resume using RAG with chat history context.
        
        Args:
            question: The user's current question
            chat_history: List of previous messages [{'role': 'user'/'assistant', 'content': '...'}]
        """
        if not self.analyzer.resume_text:
            return "Please analyze a resume first."
        
        chat_history = chat_history or []
        
        # Lazily build RAG store on first use
        if not self.analyzer.rag_vectorstore:
            try:
                self.analyzer.rag_vectorstore = self.analyzer.create_rag_vector_store(self.analyzer.resume_text)
            except Exception as e:
                # If RAG fails, use full resume text
                pass
        
        # Try to get relevant context from RAG
        context = ""
        if self.analyzer.rag_vectorstore:
            retriever = self.analyzer.rag_vectorstore.as_retriever(search_kwargs={"k": 5})
            try:
                docs = retriever.get_relevant_documents(question)
                context = "\n\n".join([getattr(d, 'page_content', str(d)) for d in docs])
            except Exception:
                pass
        
        # If no context from RAG or context is too short, use full resume (clamped)
        if not context or len(context) < 100:
            context = clamp_text(self.analyzer.resume_text, 2500)
        else:
            context = clamp_text(context, 2500)
        
        # Check if asking about weaknesses/analysis results
        if any(word in question.lower() for word in ['weakness', 'weak', 'missing', 'lack', 'improve', 'gap', 'need to add']):
            weakness_info = ""
            if hasattr(self.analyzer, 'resume_weaknesses') and self.analyzer.resume_weaknesses:
                weakness_info = "\n\nIdentified Weaknesses:\n"
                for w in self.analyzer.resume_weaknesses[:5]:
                    weakness_info += f"- {w.get('skill', 'Unknown')}: {w.get('detail', '')}\n"
            
            if hasattr(self.analyzer, 'analysis_result') and self.analyzer.analysis_result:
                missing = self.analyzer.analysis_result.get('missing_skills', [])
                if missing:
                    weakness_info += f"\nMissing Skills: {', '.join(missing[:10])}\n"
            
            context += weakness_info
        
        # Check if asking about strengths/skills
        if any(word in question.lower() for word in ['strength', 'strong', 'skill', 'technology', 'experience', 'good at']):
            strength_info = ""
            if hasattr(self.analyzer, 'analysis_result') and self.analyzer.analysis_result:
                strengths = self.analyzer.analysis_result.get('strengths', [])
                if strengths:
                    strength_info = f"\n\nKey Strengths: {', '.join(strengths)}\n"
                
                skill_scores = self.analyzer.analysis_result.get('skill_scores', {})
                if skill_scores:
                    top_skills = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                    strength_info += f"\nSkill Ratings:\n"
                    for skill, score in top_skills:
                        strength_info += f"- {skill}: {score}/10\n"
            
            context += strength_info
        
        # Build conversation context from chat history
        conversation_context = ""
        if chat_history and len(chat_history) > 0:
            # Include last 4 exchanges (8 messages) for context
            recent_history = chat_history[-8:] if len(chat_history) > 8 else chat_history
            conversation_context = "\n\nPrevious Conversation:\n"
            for msg in recent_history:
                role = "User" if msg['role'] == 'user' else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
            conversation_context += "\n"
        
        prompt = (
            "You are a helpful AI assistant analyzing a resume. Answer the user's question based on the resume content and conversation history provided.\n"
            "Be conversational, friendly, and helpful. Provide specific details from the resume.\n"
            "Use the conversation history to understand context and give relevant follow-up answers.\n"
            "If referring to something mentioned earlier, acknowledge it naturally.\n"
            "If you greet the user (hi/hello), respond warmly and ask how you can help with the resume.\n\n"
            f"Resume Content:\n{context}\n"
            f"{conversation_context}"
            f"Current Question: {question}\n\n"
            "Answer:"
        )
        return self.analyzer.llm_chat(messages=[{"role": "user", "content": prompt}], max_tokens=2000).strip()

    def answer_interview_question(self, question: str) -> str:
        """Generate a best-fit model answer to an interview question."""
        if not self.analyzer.resume_text:
            return ""
        
        jd_context = self.analyzer.jd_text or ""
        prompt = (
            "You are a senior candidate crafting a concise, strong answer.\n"
            "Use only the candidate's resume context (and JD if present).\n"
            "Keep it specific, with impact/metrics where possible, 4-7 sentences max.\n\n"
            f"Resume context (may be partial):\n{clamp_text(self.analyzer.resume_text, 900)}\n\n"
            + (f"Job description (optional):\n{clamp_text(jd_context, 600)}\n\n" if jd_context else "") +
            f"Question: {question}\n\nAnswer:"
        )
        try:
            return self.analyzer.llm_chat(messages=[{"role": "user", "content": prompt}]).strip()
        except Exception:
            return ""

    def generate_interview_questions(self, question_types, difficulty, num_questions):
        """Generate interview questions based on the resume."""
        if not self.analyzer.resume_text or not self.analyzer.extracted_skills:
            return []

        try:
            context = f"""
    Resume Content:
    {clamp_text(self.analyzer.resume_text, 1200)}...

    Skills to focus on: {', '.join(self.analyzer.extracted_skills)}
    Strengths: {', '.join(self.analyzer.analysis_result.get('strengths', []))}
    Areas for improvement: {', '.join(self.analyzer.analysis_result.get('missing_skills', []))}
    """

            prompt = f"""
        Generate exactly {num_questions} personalized {difficulty.lower()} level interview questions
        for this candidate based on their resume and skills.

        Only include question types from this list: {', '.join(question_types)}.

        Return ONLY valid JSON in this exact format (no backticks, no prefixes/suffixes):
        [
            {{
                "type": "<One type from the list above>",
                "question": "<A real interview question>",
                "solution": "<A best-fit, strong answer tailored to the resume in 4-7 sentences>"
            }}
        ]

        Requirements:
        - Output MUST contain exactly {num_questions} items.
        - "type" must be one of the allowed types exactly.
        - "question" must be a complete interview question.
        - "solution" must be a best-fit answer using the resume context.
        - Do not include any extra commentary.
        {context}
        """

            raw_response = self.analyzer.llm_chat(messages=[{"role": "user", "content": prompt}]).strip()

            # Try parsing JSON
            try:
                parsed_questions = json.loads(raw_response)
            except json.JSONDecodeError:
                pattern = r'"type"\s*:\s*"([^"]+)"\s*,\s*"question"\s*:\s*"([^"]+)"(?:\s*,\s*"solution"\s*:\s*"([^"]+)")?'
                matches = re.findall(pattern, raw_response, re.DOTALL)
                parsed_questions = [{"type": m[0], "question": m[1], "solution": (m[2] if len(m) > 2 else "")} for m in matches]

            # Clean & validate
            cleaned_questions = []
            for q in parsed_questions:
                q_type = q.get("type", "").strip()
                q_text = q.get("question", "").strip()
                q_sol = q.get("solution", "").strip()

                if q_type and q_text and q_type.lower() in [t.lower() for t in question_types]:
                    if not q_sol:
                        q_sol = self.answer_interview_question(q_text)
                    cleaned_questions.append({"type": q_type, "question": q_text, "solution": q_sol})

            # Deduplicate
            seen = set()
            deduped = []
            for item in cleaned_questions:
                qt = item.get("question", "").strip()
                if qt and qt.lower() not in seen:
                    deduped.append(item)
                    seen.add(qt.lower())
            cleaned_questions = deduped

            # If too few, generate more
            if len(cleaned_questions) < num_questions:
                remaining = num_questions - len(cleaned_questions)
                fill_prompt = f"""
            Generate exactly {remaining} additional interview questions that are DIFFERENT from:
            {json.dumps([q.get('question','') for q in cleaned_questions])}

            Only include question types from this list: {', '.join(question_types)}.
            Return ONLY valid JSON list in the same format (type, question, solution).
            {context}
            """
                fill_raw = self.analyzer.llm_chat(messages=[{"role": "user", "content": fill_prompt}]).strip()
                try:
                    fill_parsed = json.loads(fill_raw)
                except json.JSONDecodeError:
                    pattern = r'"type"\s*:\s*"([^"]+)"\s*,\s*"question"\s*:\s*"([^"]+)"(?:\s*,\s*"solution"\s*:\s*"([^"]+)")?'
                    matches = re.findall(pattern, fill_raw, re.DOTALL)
                    fill_parsed = [{"type": m[0], "question": m[1], "solution": (m[2] if len(m) > 2 else "")} for m in matches]

                for q in fill_parsed:
                    q_type = q.get("type", "").strip()
                    q_text = q.get("question", "").strip()
                    q_sol = q.get("solution", "").strip()
                    if q_type and q_text and q_type.lower() in [t.lower() for t in question_types]:
                        if not q_sol:
                            q_sol = self.answer_interview_question(q_text)
                        if q_text.lower() not in seen:
                            cleaned_questions.append({"type": q_type, "question": q_text, "solution": q_sol})
                            seen.add(q_text.lower())

            # Fallback if still empty
            if not cleaned_questions:
                for t in question_types:
                    q_text = f"Tell me about your experience with {t}."
                    q_sol = self.answer_interview_question(q_text)
                    cleaned_questions.append({"type": t, "question": q_text, "solution": q_sol})

            return cleaned_questions[:num_questions]

        except Exception as e:
            print(f"Error generating interview questions: {e}")
            return []

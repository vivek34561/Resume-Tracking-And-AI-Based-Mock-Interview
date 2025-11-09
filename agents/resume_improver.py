"""Resume Improver Agent - Resume enhancement and document generation."""

import os
import re
import json
import tempfile
from utils.text_utils import clamp_text


class ResumeImprover:
    """Handles resume improvement, cover letter, and LaTeX resume generation."""
    
    def __init__(self, resume_analyzer):
        """Initialize with a ResumeAnalyzer instance."""
        self.analyzer = resume_analyzer
        self.improved_resume_path = None
    
    def improve_resume(self, improvement_areas, target_role=""):
        """Generate suggestions to improve the resume."""
        if not self.analyzer.resume_text:
            print("ERROR: No resume text found in analyzer")
            return {}

        print(f"DEBUG: Processing {len(improvement_areas)} improvement areas")
        print(f"DEBUG: Resume text length: {len(self.analyzer.resume_text)}")
        print(f"DEBUG: Weaknesses count: {len(self.analyzer.resume_weaknesses or [])}")

        try:
            improvements = {}

            for area in improvement_areas:
                if area == 'Skills Highlighting' and self.analyzer.resume_weaknesses:
                    skill_improvements = {
                        "description": "Your resume needs to better highlight key skills that are important for the role.",
                        "specific": []
                    }
                    before_after_examples = {}

                    for weakness in self.analyzer.resume_weaknesses:
                        skill_name = weakness.get("skill", "")
                        if "suggestions" in weakness and weakness["suggestions"]:
                            for suggestion in weakness["suggestions"]:
                                skill_improvements["specific"].append(f"**{skill_name}**: {suggestion}")

                        if "example" in weakness and weakness["example"]:
                            resume_chunks = self.analyzer.resume_text.split('\n\n')
                            relevant_chunk = ""

                            for chunk in resume_chunks:
                                if skill_name.lower() in chunk.lower() or "experience" in chunk.lower():
                                    relevant_chunk = chunk
                                    break
                            if relevant_chunk:
                                before_after_examples = {
                                    "before": relevant_chunk.strip(),
                                    "after": relevant_chunk.strip() + "\n" + weakness["example"]
                                }

                    if before_after_examples:
                        skill_improvements["before_after"] = before_after_examples

                    improvements["Skills Highlighting"] = skill_improvements

            remaining_areas = [area for area in improvement_areas if area not in improvements]

            if remaining_areas:
                weaknesses_text = ""
                if self.analyzer.resume_weaknesses:
                    weaknesses_text = "Resume Weaknesses:\n"
                    for i, weakness in enumerate(self.analyzer.resume_weaknesses):
                        weaknesses_text += f"{i + 1}. {weakness['skill']}: {weakness['detail']}\n"
                        if "suggestions" in weakness:
                            for j, sugg in enumerate(weakness["suggestions"]):
                                weaknesses_text += f"   - {sugg}\n"

                # Get strengths and other analysis data
                strengths_list = self.analyzer.analysis_result.get('strengths', []) if self.analyzer.analysis_result else []
                
                context = f"""
Resume Content (first 2000 chars):
{clamp_text(self.analyzer.resume_text, 2000)}

Extracted Skills: {', '.join(self.analyzer.extracted_skills or [])}

Strengths: {', '.join(strengths_list)}

{weaknesses_text}

Target role: {target_role if target_role else "Not specified"}
"""

                prompt = f"""You are an expert resume consultant. Analyze the resume and provide detailed, actionable improvement suggestions for these areas: {', '.join(remaining_areas)}.

{context}

For EACH improvement area listed above, provide:
1. A clear description explaining what needs improvement (2-3 sentences)
2. 3-5 specific, actionable suggestions with concrete examples
3. If applicable, a before/after example showing the improvement

Return ONLY a valid JSON object with this exact structure:
{{
  "Area Name": {{
    "description": "What needs improvement and why",
    "specific": [
      "First actionable suggestion with specific examples",
      "Second actionable suggestion with specific examples",
      "Third actionable suggestion with specific examples"
    ],
    "before_after": {{
      "before": "Original text example",
      "after": "Improved text example"
    }}
  }}
}}

Be specific and practical. Reference actual content from the resume in your suggestions."""

                print(f"DEBUG: Sending prompt to LLM for {len(remaining_areas)} areas")
                response = self.analyzer.llm_chat(
                    messages=[{"role": "user", "content": prompt}], 
                    temperature=0.3,
                    max_tokens=2000  # Increased for detailed suggestions
                )
                print(f"DEBUG: LLM response length: {len(response)}")
                print(f"DEBUG: LLM response preview: {response[:200]}")
                
                # Try to extract JSON
                ai_improvements = {}
                
                # Try direct JSON parse first
                try:
                    ai_improvements = json.loads(response)
                    improvements.update(ai_improvements)
                except json.JSONDecodeError:
                    # Try to find JSON in code blocks
                    json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', response)
                    if json_match:
                        try:
                            ai_improvements = json.loads(json_match.group(1))
                            improvements.update(ai_improvements)
                        except json.JSONDecodeError:
                            pass
                
                # If still no improvements, try markdown parsing
                if not ai_improvements:
                    # Parse markdown-style response
                    current_area = None
                    current_desc = []
                    current_suggestions = []
                    
                    lines = response.split('\n')
                    for line in lines:
                        # Check for area headers
                        if any(area in line for area in remaining_areas):
                            # Save previous area
                            if current_area and (current_desc or current_suggestions):
                                improvements[current_area] = {
                                    "description": ' '.join(current_desc).strip(),
                                    "specific": current_suggestions
                                }
                            # Start new area
                            for area in remaining_areas:
                                if area in line:
                                    current_area = area
                                    current_desc = []
                                    current_suggestions = []
                                    break
                        elif current_area:
                            # Collect description and suggestions
                            stripped = line.strip()
                            if stripped.startswith(('- ', '* ', '• ', '1.', '2.', '3.', '4.', '5.')):
                                # This is a suggestion
                                cleaned = re.sub(r'^[-*•\d.]\s*', '', stripped)
                                if cleaned:
                                    current_suggestions.append(cleaned)
                            elif stripped and not stripped.startswith('#'):
                                # This is description text
                                current_desc.append(stripped)
                    
                    # Save last area
                    if current_area and (current_desc or current_suggestions):
                        improvements[current_area] = {
                            "description": ' '.join(current_desc).strip(),
                            "specific": current_suggestions
                        }

            # Only add fallback for areas that truly have no content
            for area in improvement_areas:
                if area not in improvements or (not improvements[area].get('specific') and not improvements[area].get('description')):
                    improvements[area] = {
                        "description": f"Enhance your {area.lower()} to make your resume more competitive and aligned with industry standards.",
                        "specific": [
                            f"Review and strengthen the {area.lower()} section",
                            "Add specific metrics and quantifiable achievements where possible",
                            "Ensure alignment with target role requirements and industry best practices"
                        ]
                    }

            return improvements

        except Exception as e:
            print(f"Error generating resume improvements: {e}")
            return {area: {"description": "Error generating suggestions", "specific": []} for area in improvement_areas}

    def get_improved_resume(self, target_role="", highlight_skills=""):
        """Generate an improved version of the resume."""
        if not self.analyzer.resume_text:
            return "Please upload and analyze a resume first."

        try:
            skills_to_highlight = []

            if highlight_skills:
                if len(highlight_skills) > 100:
                    self.analyzer.jd_text = highlight_skills
                    try:
                        parsed_skills = self.analyzer.extract_skills_from_jd(highlight_skills)
                        skills_to_highlight = parsed_skills if parsed_skills else [s.strip() for s in highlight_skills.split(",") if s.strip()]
                    except:
                        skills_to_highlight = [s.strip() for s in highlight_skills.split(",") if s.strip()]
                else:
                    skills_to_highlight = [s.strip() for s in highlight_skills.split(",") if s.strip()]

            if not skills_to_highlight and self.analyzer.analysis_result:
                skills_to_highlight = self.analyzer.analysis_result.get("missing_skills", [])
                skills_to_highlight.extend([s for s in self.analyzer.analysis_result.get("strengths", []) if s not in skills_to_highlight])
                if self.analyzer.extracted_skills:
                    skills_to_highlight.extend([s for s in self.analyzer.extracted_skills if s not in skills_to_highlight])

            weakness_context = ""
            improvement_examples = ""

            if self.analyzer.resume_weaknesses:
                weakness_context = "Address these specific weaknesses:\n"
                for weakness in self.analyzer.resume_weaknesses:
                    skill_name = weakness.get('skill', '')
                    weakness_context += f"- {skill_name}: {weakness.get('detail', '')}\n"
                    if 'suggestions' in weakness:
                        for suggestion in weakness['suggestions']:
                            weakness_context += f" * {suggestion}\n"
                    if 'example' in weakness and weakness['example']:
                        improvement_examples += f"For {skill_name}: {weakness['example']}\n\n"

            jd_context = ""
            if self.analyzer.jd_text:
                jd_context = f"Job Description:\n{self.analyzer.jd_text}\n\n"
            elif target_role:
                jd_context = f"Target Role: {target_role}\n\n"

            prompt = f"""
Rewrite and improve this resume to make it highly optimized for the target job.
{jd_context}
Original Resume:
{clamp_text(self.analyzer.resume_text, 3000)}

Skills to highlight (in order of priority): {', '.join(skills_to_highlight)}

{weakness_context}

Here are specific examples of content to add:
{improvement_examples}

Please improve the resume by:
1. Adding strong, quantifiable achievements
2. Highlighting the specified skills strategically for ATS scanning
3. Addressing all the weakness areas identified with the specific suggestions provided
4. Incorporating the example improvements provided above
5. Structuring information in a clear, professional format
6. Using industry-standard terminology
7. Ensuring all relevant experience is properly emphasized
8. Adding measurable outcomes and achievements

Return only the improved resume text without any additional explanations.
Format the resume in a modern, clean style with clear section headings.
Make sure to include ALL sections from the original resume (contact info, summary, experience, education, skills, etc.).
"""

            print(f"DEBUG: Generating improved resume with target_role='{target_role}', skills_count={len(skills_to_highlight)}")
            # Use much higher max_tokens for full resume generation (Groq allows up to 8000)
            improved_resume = self.analyzer.llm_chat(
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.3,
                max_tokens=4000  # Increased from default 600 to 4000 for complete resume
            ).strip()
            print(f"DEBUG: Generated improved resume length: {len(improved_resume)} characters")

            with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as tmp:
                tmp.write(improved_resume)
                self.improved_resume_path = tmp.name

            return improved_resume

        except Exception as e:
            print(f"Error generating improved resume: {e}")
            return "Error generating improved resume. Please try again."

    def generate_cover_letter(self, company: str, role: str, job_description: str = "", 
                            tone: str = "professional", length: str = "one-page") -> str:
        """Generate a tailored cover letter."""
        if not self.analyzer.resume_text:
            return "Please upload and analyze a resume first."

        try:
            jd_clean = self.analyzer.clean_job_description(job_description) if job_description else ""
            skills_focus = ", ".join(self.analyzer.extracted_skills or [])
            strengths = ", ".join(self.analyzer.analysis_result.get('strengths', [])) if self.analyzer.analysis_result else ""
            weaknesses = ", ".join(self.analyzer.analysis_result.get('missing_skills', [])) if self.analyzer.analysis_result else ""
            jd_section = f" - Job Description (cleaned):\n{jd_clean}" if jd_clean else ""

            prompt = f"""
You are an expert career writer. Draft a tailored cover letter.

Context:
- Company: {company}
- Role: {role}
- Writing tone: {tone}
- Desired length: {length}
- Resume (excerpts, may be partial):\n{clamp_text(self.analyzer.resume_text, 1000)}
- Skills to emphasize: {skills_focus}
- Strengths from analysis: {strengths}
- Potential gaps to address carefully: {weaknesses}
{jd_section}

Requirements:
- Start with a compelling intro aligned to the company and role.
- Highlight 3-4 most relevant achievements with measurable impact.
- Weave in the listed skills naturally.
- Address potential gaps briefly and positively if relevant.
- End with a confident, polite closing and call to action.
- Keep formatting as plain text paragraphs (no markdown), with a professional sign-off.

Output:
Return ONLY the letter body, no extra commentary.
"""
            letter = self.analyzer.llm_chat(messages=[{"role": "user", "content": prompt}]).strip()
            
            if len(letter) < 200:
                prompt2 = prompt + "\nEnsure the letter is at least 250 words and no more than 600 words."
                letter = self.analyzer.llm_chat(messages=[{"role": "user", "content": prompt2}]).strip()
            
            return letter
        except Exception as e:
            print(f"Error generating cover letter: {e}")
            return "Error generating cover letter. Please try again."

    def generate_updated_resume_latex(self, latex_source: str, job_description: str) -> str:
        """Update a LaTeX resume to match job description."""
        if not latex_source or latex_source.strip() == "":
            return "Please paste your current LaTeX resume code."
        
        jd_clean = self.analyzer.clean_job_description(job_description) if job_description else ""
        
        try:
            context = f"Analyzed resume excerpts (for content ideas):\n{clamp_text((self.analyzer.resume_text or ''), 1500)}"
            prompt = (
                "You are an expert resume editor and LaTeX practitioner. Update the LaTeX resume below to match the given job description, "
                "STRICTLY preserving the LaTeX format (documentclass, preamble, macros, environments). Only modify textual content (section text, bullets, achievements).\n\n"
                f"Job Description (cleaned):\n{jd_clean}\n\n{context}\n\n"
                "Return ONLY the updated LaTeX source with no explanations. Ensure it compiles."
            )
            user_content = (
                "----- BEGIN ORIGINAL LATEX -----\n" + latex_source + "\n----- END ORIGINAL LATEX -----\n"
            )
            updated = self.analyzer.llm_chat(messages=[
                {"role": "system", "content": "Follow rules strictly; preserve LaTeX preamble and macros; output only LaTeX."},
                {"role": "user", "content": prompt},
                {"role": "user", "content": user_content},
            ]).strip()
            
            # Sanity check
            if "\\documentclass" not in updated and "\\begin{document}" not in updated:
                repair_prompt = (
                    "Output must be the FULL LaTeX document. Include the original preamble unchanged and all sections."
                )
                updated = self.analyzer.llm_chat(messages=[
                    {"role": "system", "content": "Output only full LaTeX source; preserve preamble and macros exactly."},
                    {"role": "user", "content": repair_prompt},
                    {"role": "user", "content": user_content},
                ]).strip()
            
            return updated
        except Exception as e:
            print(f"Error updating LaTeX resume: {e}")
            return "Error generating updated LaTeX resume. Please try again."

    def cleanup(self):
        """Clean up temporary files."""
        try:
            if hasattr(self, 'improved_resume_path') and self.improved_resume_path and os.path.exists(self.improved_resume_path):
                os.unlink(self.improved_resume_path)
            if hasattr(self.analyzer, 'resume_file_path') and os.path.exists(self.analyzer.resume_file_path):
                os.unlink(self.analyzer.resume_file_path)
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")

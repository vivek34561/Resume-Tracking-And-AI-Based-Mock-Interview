"""Resume Analysis Agent - Core analysis functionality."""

import os
import re
import json
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from utils.llm_providers import groq_chat, ollama_chat
from utils.text_utils import clamp_text, compute_hash
from utils.file_handlers import extract_text_from_pdf, extract_text_from_txt, extract_text_from_file


class ResumeAnalyzer:
    """Handles resume analysis, skill extraction, and job description processing."""
    
    def __init__(self, api_key, cutoff_score=75, model=None, provider: str = 'groq', 
                 ollama_base_url: str | None = None, user_id: int | None = None, 
                 vector_cache_dir: str | None = None):
        self.api_key = api_key
        self.cutoff_score = cutoff_score
        self.provider = provider
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        
        # Prefer provider-specific default
        if provider == 'ollama':
            self.model = model or os.getenv("OLLAMA_MODEL") or "llama3.1:8b"
        else:
            self.model = model or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"
        
        self.resume_text = None
        self.resume_hash = None
        self.analysis_result = None
        self.rag_vectorstore = None
        self.jd_text = None
        self.extracted_skills = None
        self.resume_weaknesses = []
        self.resume_strengths = []
        self.improvement_suggestions = {}
        
        # Caching settings
        self.user_id = user_id
        self.vector_cache_dir = vector_cache_dir or os.getenv("VECTOR_CACHE_DIR") or ".cache/faiss"
        
        # Lazy embeddings cache
        self._embeddings = None

    def _get_embeddings(self):
        """Lazy load embeddings."""
        if self._embeddings is None:
            self._embeddings = FastEmbedEmbeddings()
        return self._embeddings

    def _compute_resume_hash(self, text: str) -> str:
        """Compute hash for resume text."""
        return compute_hash(text)

    def _compute_jd_hash(self, jd_text: str | None, skills: list | None) -> str:
        """Compute hash for job description."""
        base = (jd_text or "").strip()
        if not base and skills:
            base = ",".join(sorted([str(s).strip().lower() for s in skills if s]))
        if not base:
            base = "no-jd"
        return compute_hash(base)

    def llm_chat(self, messages: list, temperature: float = 0.2, max_tokens: int = 600) -> str:
        """Provider-aware chat helper."""
        if self.provider == 'ollama':
            return ollama_chat(messages, model=self.model, base_url=self.ollama_base_url, temperature=temperature)
        return groq_chat(self.api_key, messages=messages, model=self.model, temperature=temperature, max_tokens=max_tokens)

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file."""
        return extract_text_from_pdf(pdf_file)

    def extract_text_from_txt(self, txt_file):
        """Extract text from TXT file."""
        return extract_text_from_txt(txt_file)

    def extract_text_from_file(self, file):
        """Extract text from file (PDF or TXT)."""
        return extract_text_from_file(file)

    def create_rag_vector_store(self, text):
        """Create or load a cached FAISS vector store for RAG using FastEmbed."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_text(text)
        embeddings = self._get_embeddings()
        
        # Determine cache path
        r_hash = self.resume_hash or self._compute_resume_hash(text)
        user_part = str(self.user_id or "anon")
        cache_path = os.path.join(self.vector_cache_dir, user_part, r_hash, "rag")
        
        try:
            if os.path.isdir(cache_path) and os.listdir(cache_path):
                vs = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
                return vs
        except Exception:
            pass
        
        # Build fresh and cache
        vectorstore = FAISS.from_texts(chunks, embeddings)
        try:
            os.makedirs(cache_path, exist_ok=True)
            vectorstore.save_local(cache_path)
        except Exception:
            pass
        return vectorstore

    def create_vector_store(self, text):
        """Create or load a cached single-shot FAISS store for whole-resume queries."""
        embeddings = self._get_embeddings()
        r_hash = self.resume_hash or self._compute_resume_hash(text)
        user_part = str(self.user_id or "anon")
        cache_path = os.path.join(self.vector_cache_dir, user_part, r_hash, "single")
        
        try:
            if os.path.isdir(cache_path) and os.listdir(cache_path):
                vs = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
                return vs
        except Exception:
            pass
        
        # Build and cache
        vectorstore = FAISS.from_texts([text], embeddings)
        try:
            os.makedirs(cache_path, exist_ok=True)
            vectorstore.save_local(cache_path)
        except Exception:
            pass
        return vectorstore

    def clean_job_description(self, raw_text: str) -> str:
        """Clean job description text."""
        patterns = [
            r'Apply', r'Save', r'Show more options', r'How your profile.*',
            r'Get AI-powered advice.*', r'Tailor my resume.*', r'Did you apply.*',
            r'Yes', r'No'
        ]
        for pat in patterns:
            raw_text = re.sub(pat, '', raw_text, flags=re.IGNORECASE)
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        return "\n".join(lines)

    def extract_skills_from_jd(self, jd_text):
        """Extract skills from job description using LLM."""
        try:
            jd_snippet = clamp_text(jd_text, 1500)
            prompt = f"""
            Extract a comprehensive list of technical skills, technologies, and competencies required from this job description.
            Return ONLY a plain comma-separated list of skills.

            Job Description:
            {jd_snippet}
            """
            skills_text = self.llm_chat(messages=[{"role": "user", "content": prompt}]).strip()
            skills = [s.strip() for s in re.split(r',|\n|-|\*', skills_text) if s.strip()]
            return list(dict.fromkeys(skills))
        except Exception as e:
            print(f"Error extracting skills from job description: {e}")
            return []

    def fast_extract_skills_from_jd(self, jd_text: str) -> list:
        """Heuristic skill extraction without LLM for quick mode."""
        vocab = {
            # Programming Languages
            "python","java","javascript","typescript","c","c++","c#","go","golang","rust","kotlin","swift","ruby","php","scala","r","matlab","perl","dart","lua",
            # Web/Frontend
            "react","next.js","nextjs","angular","vue","svelte","html","css","html5","css3","sass","scss","less","tailwind","bootstrap","redux","graphql","webpack","vite","parcel","gulp","npm","yarn","pnpm",
            # Backend/Frameworks
            "node","node.js","express","django","flask","fastapi","spring","spring boot",".net","dotnet","asp.net","grpc","rest","restful","microservices","soap","laravel","rails","ruby on rails",
            # Databases
            "sql","mysql","postgresql","postgres","mongodb","redis","elasticsearch","cassandra","dynamodb","mariadb","oracle","sqlite","neo4j","couchdb","firestore",
            # Message Queue/Streaming
            "kafka","rabbitmq","redis","activemq","zeromq","nats","pulsar","kinesis",
            # Big Data/Analytics
            "spark","hadoop","hive","airflow","databricks","etl","data warehouse","snowflake","bigquery","redshift","presto","flink",
            # Machine Learning/AI
            "machine learning","deep learning","ml","dl","nlp","natural language processing","computer vision","cv","pandas","numpy","scikit-learn","sklearn","tensorflow","pytorch","keras","transformers","hugging face","bert","gpt","llm","generative ai","langchain","llama","rag",
            # DevOps/Cloud
            "docker","kubernetes","k8s","terraform","ansible","puppet","chef","jenkins","gitlab","github actions","ci/cd","cicd","git","github","gitlab","bitbucket","linux","unix","bash","shell","aws","azure","gcp","google cloud","cloud","heroku","vercel","netlify","cloudflare",
            # AWS Services
            "ec2","s3","lambda","rds","cloudformation","ecs","eks","sqs","sns","cloudwatch","iam","vpc","route53","api gateway",
            # Azure Services  
            "azure functions","azure sql","blob storage","cosmos db","aks","azure devops",
            # GCP Services
            "compute engine","cloud storage","cloud functions","cloud run","gke","pub/sub",
            # Mobile
            "android","ios","react native","flutter","swiftui","kotlin","swift","xamarin","ionic",
            # Testing/QA
            "pytest","unittest","selenium","cypress","playwright","junit","jest","mocha","jasmine","testng","postman","jmeter","loadrunner",
            # Methodologies
            "agile","scrum","kanban","waterfall","devops","tdd","test driven development","bdd","behavior driven development",
            # Soft Skills
            "leadership","communication","teamwork","problem solving","analytical","critical thinking","project management","time management",
            # Tools
            "jira","confluence","slack","trello","asana","figma","sketch","postman","insomnia","datadog","new relic","splunk","prometheus","grafana","tableau","power bi","excel","jupyter","vscode","intellij","eclipse","vim",
            # Security
            "oauth","jwt","ssl","tls","encryption","authentication","authorization","security","cybersecurity","penetration testing","owasp",
            # Other Tech
            "api","json","xml","yaml","websocket","graphql","grpc","protobuf","openapi","swagger","nginx","apache","tomcat","iis","elasticsearch","solr","memcached"
        }
        
        text = jd_text.lower()
        found = set()
        for term in vocab:
            if term in text:
                found.add(term)
        
        # Capture capitalized acronyms
        caps = set(re.findall(r"\b([A-Z]{2,5})\b", jd_text))
        for c in caps:
            if c.lower() not in found:
                found.add(c.lower())
        
        # Normalize for display
        norm = []
        for f in found:
            if len(f) <= 5 and f.isupper():
                norm.append(f)
            elif " " in f:
                norm.append(" ".join(w.capitalize() for w in f.split()))
            elif f in {"sql","aws","gcp","nlp","cv","ci/cd","cicd"}:
                norm.append(f.upper())
            elif f in {"c#","c++"}:
                norm.append(f.upper())
            else:
                norm.append(f.capitalize())
        
        # Keep original order of appearance
        order = []
        for m in re.finditer(r"[A-Za-z\+#\.]{2,}(?:\s[A-Za-z\+\.#]{2,})*", jd_text):
            token = m.group(0).strip().lower()
            for n in norm:
                if n.lower() in token and n not in order:
                    order.append(n)
        return order or sorted(set(norm), key=lambda s: s.lower())

    def analyze_skill(self, retriever, resume_text, skill):
        """Analyze a single skill."""
        try:
            docs = retriever.get_relevant_documents(skill)
        except Exception:
            docs = []
        
        context = "\n\n".join([getattr(d, 'page_content', str(d)) for d in docs][:3]) or clamp_text(resume_text, 1200)
        context = clamp_text(context, 1500)
        
        user = (
            f"Context from resume (may be partial):\n{context}\n\n"
            f"Task: On a scale of 0-10, how clearly does the candidate mention proficiency in '{skill}'? "
            f"First output ONLY a number (0-10), then a short reasoning sentence."
        )
        text = self.llm_chat(messages=[{"role": "user", "content": user}])
        match = re.search(r"\b(\d{1,2})\b", text)
        score = int(match.group(1)) if match else 0
        reasoning = text
        if match:
            idx = text.find(match.group(1))
            if idx >= 0:
                reasoning = text[idx + len(match.group(1)):].strip(" -:;\n")
        return skill, min(score, 10), reasoning

    def semantic_skill_analysis(self, resume_text, skills):
        """Batch skill scoring in a single LLM call."""
        vectorstore = self.create_vector_store(resume_text)
        skill_scores, skill_reasoning, missing_skills, total_score = {}, {}, [], 0
        
        if not skills:
            return {
                "overall_score": 0,
                "skill_scores": {},
                "skill_reasoning": {},
                "selected": False,
                "reasoning": "No skills provided.",
                "missing_skills": [],
                "strengths": [],
                "improvement_areas": []
            }
        
        # Use larger resume snippet for better skill detection (increased from 900 to 2000)
        resume_snippet = clamp_text(resume_text, 2000)
        skills_list = ", ".join(skills)
        prompt = (
            "Rate each skill (0-10) based ONLY on this resume text. Return strict JSON: {\"skill_scores\":{skill:score}, \"skill_reasoning\":{skill:short_reason}}.\n"
            f"Resume:\n{resume_snippet}\n\nSkills: {skills_list}\n"
        )
        
        parsed_ok = False
        try:
            resp = self.llm_chat(messages=[{"role": "user", "content": prompt}], temperature=0.1)
            data = json.loads(resp)
            ss = data.get("skill_scores", {})
            sr = data.get("skill_reasoning", {})
            
            if isinstance(ss, dict) and ss:
                for k, v in ss.items():
                    try:
                        v_int = int(v)
                    except Exception:
                        m = re.search(r"\b(\d{1,2})\b", str(v))
                        v_int = int(m.group(1)) if m else 0
                    v_int = max(0, min(10, v_int))
                    skill_scores[k] = v_int
                    total_score += v_int
                    if v_int <= 5:
                        missing_skills.append(k)
                    skill_reasoning[k] = (sr.get(k) or "").strip()
                parsed_ok = True
        except Exception:
            parsed_ok = False
        
        if not parsed_ok:
            # Fallback to per-skill analysis
            retriever = vectorstore.as_retriever()
            for s in skills:
                skill, score, reasoning = self.analyze_skill(retriever, resume_text, s)
                skill_scores[skill] = score
                skill_reasoning[skill] = reasoning
                total_score += score
                if score <= 5:
                    missing_skills.append(skill)
        
        overall_score = int((total_score / (10 * len(skills))) * 100) if skills else 0
        selected = overall_score >= self.cutoff_score
        strengths = [skill for skill, score in skill_scores.items() if score >= 7]
        self.resume_strengths = strengths
        
        return {
            "overall_score": overall_score,
            "skill_scores": skill_scores,
            "skill_reasoning": skill_reasoning,
            "selected": selected,
            "reasoning": "Batch skill analysis.",
            "missing_skills": missing_skills,
            "strengths": strengths,
            "improvement_areas": missing_skills if not selected else []
        }

    def analyze_resume(self, resume_file, role_requirements=None, custom_jd=None, quick: bool = False):
        """Analyze resume from file."""
        self.resume_text = self.extract_text_from_file(resume_file)
        self.resume_hash = self._compute_resume_hash(self.resume_text)
        
        # Cache check
        try:
            from database import get_cached_analysis
        except Exception:
            get_cached_analysis = None
        
        # Process JD
        if custom_jd:
            raw_jd_text = self.extract_text_from_file(custom_jd) if hasattr(custom_jd, 'read') else str(custom_jd)
            self.jd_text = self.clean_job_description(raw_jd_text)
            jd_skills = self.fast_extract_skills_from_jd(self.jd_text) if quick else self.extract_skills_from_jd(self.jd_text)
        else:
            jd_skills = role_requirements or []
        
        if not jd_skills:
            jd_skills = ["teamwork"]
        
        # In quick mode, limit to 10 skills instead of 5 for better coverage
        if quick and len(jd_skills) > 10:
            jd_skills = jd_skills[:10]
        
        self.extracted_skills = jd_skills
        
        # Cache hit?
        if get_cached_analysis and (self.user_id and self.resume_hash):
            jd_hash = self._compute_jd_hash(self.jd_text, jd_skills)
            prov = getattr(self, 'provider', '')
            mdl = getattr(self, 'model', '')
            intensity = 'quick' if quick else 'full'
            cached = get_cached_analysis(self.user_id, self.resume_hash, jd_hash, prov, mdl, intensity)
            if cached:
                self.analysis_result = cached
                self.resume_weaknesses = cached.get("detailed_weaknesses", [])
                return self.analysis_result
        
        self.analysis_result = self.semantic_skill_analysis(self.resume_text, jd_skills)
        
        if not quick:
            self.analyze_resume_weaknesses()
        
        self.analysis_result["detailed_weaknesses"] = getattr(self, "resume_weaknesses", [])
        
        if quick:
            self.analysis_result["note"] = "Quick analysis completed. Click Analyze to run full detailed analysis."
        
        # Save cache
        try:
            from database import save_cached_analysis
            if self.user_id and self.resume_hash:
                jd_hash = self._compute_jd_hash(self.jd_text, jd_skills)
                prov = getattr(self, 'provider', '')
                mdl = getattr(self, 'model', '')
                intensity = 'quick' if quick else 'full'
                save_cached_analysis(self.user_id, self.resume_hash, jd_hash, prov, mdl, intensity, self.analysis_result)
        except Exception:
            pass
        
        return self.analysis_result

    def analyze_resume_text(self, resume_text: str, role_requirements=None, custom_jd=None, quick: bool = False):
        """Analyze resume from text string."""
        self.resume_text = resume_text or ""
        self.resume_hash = self._compute_resume_hash(self.resume_text)
        
        try:
            from database import get_cached_analysis, save_cached_analysis
        except Exception:
            get_cached_analysis = None
            save_cached_analysis = None
        
        # Process JD/skills
        if custom_jd:
            self.jd_text = self.clean_job_description(custom_jd)
            jd_skills = self.fast_extract_skills_from_jd(self.jd_text) if quick else self.extract_skills_from_jd(self.jd_text)
        else:
            jd_skills = role_requirements or []
        
        if not jd_skills:
            jd_skills = ["teamwork"]
        
        # In quick mode, limit to 10 skills instead of 5 for better coverage
        if quick and len(jd_skills) > 10:
            jd_skills = jd_skills[:10]
        
        self.extracted_skills = jd_skills
        
        # Cache hit?
        if get_cached_analysis and (self.user_id and self.resume_hash):
            jd_hash = self._compute_jd_hash(self.jd_text, jd_skills)
            prov = getattr(self, 'provider', '')
            mdl = getattr(self, 'model', '')
            intensity = 'quick' if quick else 'full'
            cached = get_cached_analysis(self.user_id, self.resume_hash, jd_hash, prov, mdl, intensity)
            if cached:
                self.analysis_result = cached
                self.resume_weaknesses = cached.get("detailed_weaknesses", [])
                return self.analysis_result
        
        self.analysis_result = self.semantic_skill_analysis(self.resume_text, jd_skills)
        
        if not quick:
            self.analyze_resume_weaknesses()
        
        self.analysis_result["detailed_weaknesses"] = getattr(self, "resume_weaknesses", [])
        
        if quick:
            self.analysis_result["note"] = "Quick analysis completed. Click Analyze to run full detailed analysis."
        
        # Save cache
        try:
            if save_cached_analysis and self.user_id and self.resume_hash:
                jd_hash = self._compute_jd_hash(self.jd_text, jd_skills)
                prov = getattr(self, 'provider', '')
                mdl = getattr(self, 'model', '')
                intensity = 'quick' if quick else 'full'
                save_cached_analysis(self.user_id, self.resume_hash, jd_hash, prov, mdl, intensity, self.analysis_result)
        except Exception:
            pass
        
        return self.analysis_result

    def analyze_resume_weaknesses(self):
        """Analyze weaknesses in resume."""
        weaknesses = []
        
        if not self.resume_text or not self.extracted_skills or not self.analysis_result:
            return weaknesses
        
        missing = list(self.analysis_result.get("missing_skills", []))
        if not missing:
            self.resume_weaknesses = []
            return []
        
        try:
            # Increased from 900 to 1500 characters for better context
            resume_snip = clamp_text(self.resume_text, 1500)
            skills_csv = ", ".join(missing)
            prompt = (
                "For each of these skills, analyze why the resume appears weak or missing, and provide 2-3 actionable suggestions and one example bullet. "
                "Return STRICT JSON of the form {skill:{detail:str, suggestions:[str], example:str}} with only these keys.\n\n"
                f"Resume (excerpt):\n{resume_snip}\n\nSkills: {skills_csv}\n"
            )
            resp = self.llm_chat(messages=[{"role": "user", "content": prompt}], temperature=0.2)
            data = {}
            try:
                data = json.loads(resp)
            except Exception:
                data = {}
            
            for sk in missing:
                entry = data.get(sk) or {}
                weaknesses.append({
                    "skill": sk,
                    "detail": (entry.get("detail") or "Not clearly demonstrated."),
                    "suggestions": (entry.get("suggestions") or [])[:3],
                    "example": (entry.get("example") or ""),
                })
        except Exception:
            # Fallback to per-skill analysis
            for skill in missing:
                try:
                    prompt = f"Briefly state why '{skill}' seems weak in this resume and give 2 short fixes. Resume: {resume_snip}"
                    response = self.llm_chat(messages=[{"role": "user", "content": prompt}], temperature=0.2)
                    weaknesses.append({"skill": skill, "detail": response[:200]})
                except Exception:
                    weaknesses.append({"skill": skill, "detail": "Error generating weakness"})
        
        self.resume_weaknesses = weaknesses
        return weaknesses

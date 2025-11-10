# ResuMate - Your AI Career Companion 🤖

> AI-powered resume analysis and interview preparation tool to help students and job seekers land their dream jobs.

## 🌟 Features

- **🔐 Google OAuth Login** - Secure "Continue with Google" authentication
- **Resume Analysis** - AI-powered analysis with ATS scoring
- **Smart Chatbot** - Ask questions about your resume with context memory
- **Interview Preparation** - Generate personalized interview questions
- **Resume Improvement** - Get AI-driven suggestions and enhanced resume
- **Cover Letter Generator** - Create tailored cover letters
- **Job Search** - Search multiple job boards at once
- **Multi-user Support** - Personal data storage with Google account integration

---

## 🚀 Quick Start

### Prerequisites

1. **Google OAuth Credentials** (Required)
   - See [GOOGLE_AUTH_SETUP.md](GOOGLE_AUTH_SETUP.md) for detailed setup instructions
   - Get credentials from [Google Cloud Console](https://console.cloud.google.com/)

2. **Groq API Key** (Required for AI features)
   - Get free API key from [Groq Console](https://console.groq.com/keys)
   - Watch: [How to Get Groq API Key](https://www.youtube.com/watch?v=nt1PJu47nTk)

3. **MySQL Database** (Required)
   - Local: MySQL 8.0+
   - Heroku: ClearDB or JawsDB add-on

### Local Setup

1. Clone the repository:
   ```powershell
   git clone https://github.com/vivek34561/Resume-Tracking-And-AI-Based-Mock-Interview.git
   cd Resume-Tracking-And-AI-Based-Mock-Interview
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Set up environment variables (copy `.env.example` to `.env`):
   ```powershell
   cp .env.example .env
   ```
   
4. Configure your `.env` file with:
   - Google OAuth credentials
   - Groq API key
   - MySQL database credentials

5. Set up the database:
   ```powershell
   python migrate_google_auth.py
   ```

6. Run the app:
   ```powershell
   streamlit run app.py
   ```

---

## Deploy to Heroku

This app is a Streamlit-based web app. The repo now includes everything needed to run on Heroku: `Procfile`, `runtime.txt`, and `Aptfile` (for `ffmpeg`). Follow these steps from Windows PowerShell.

### 1) Prerequisites

- Git installed
- Heroku account and Heroku CLI installed
- Python not required on Heroku, but helpful locally

### 2) One-time setup in this repo

The repo contains:

- `Procfile` — tells Heroku how to start Streamlit
- `runtime.txt` — pins Python 3.11.9
- `Aptfile` — installs `ffmpeg` for `streamlit-webrtc`/`av`
- `requirements.txt` — Python deps

Commit any local changes before deploying.

### 3) Login and create the app

```powershell
heroku login
heroku create your-app-name
```

### 4) Add buildpacks (Python + Apt)

Order matters — Python first, then Apt to install `ffmpeg`.

```powershell
heroku buildpacks:add -i 1 heroku/python
heroku buildpacks:add -i 2 heroku-community/apt
```

### 5) Set config vars

```powershell
heroku config:set STREAMLIT_SERVER_HEADLESS=true
heroku config:set PYTHONUNBUFFERED=true

# Optional but recommended if you want a default for API usage
# (the UI also lets you paste the key at runtime)
heroku config:set OPENAI_API_KEY=sk-... 
heroku config:set JOOBLE_API_KEY=your_jooble_key  # if you use Jooble
```

### 6) Deploy

If your current branch is `main`:

```powershell
git push heroku main
```

If your working branch is `frontend_deploy` (as in this repo):

```powershell
git push heroku frontend_deploy:main
```

Heroku will detect Python, install dependencies from `requirements.txt`, install `ffmpeg` from `Aptfile`, and launch the web dyno using the `Procfile`:

```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### 7) Open and monitor

```powershell
heroku open
heroku logs --tail
```

## Notes & Troubleshooting

- API Keys: The app UI asks for your OpenAI API key in the sidebar. You can also set `OPENAI_API_KEY` in Heroku config vars, but the current UI still expects a key pasted by the user at runtime.
- FFmpeg: `Aptfile` installs `ffmpeg` required by `streamlit-webrtc`/`av`. If you don’t use voice/video, you can remove `Aptfile` and the apt buildpack.
- Build errors on `faiss-cpu`/`av`: Ensure Heroku-22/24 stack and Python 3.11. If issues persist, try pinning versions in `requirements.txt`.
- Memory/timeouts: Streamlit apps are single-process; keep models lightweight. Prefer hosted APIs (OpenAI, Pinecone) to heavy local compute.
- Region: If you need EU region, create the app with `--region eu`.

## Local run (optional)

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Then open the printed local URL in your browser.

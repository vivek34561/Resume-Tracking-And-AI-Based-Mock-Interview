# Resume-Tracking-And-AI-Based-Mock-Interview

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

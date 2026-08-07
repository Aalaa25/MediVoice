# HealthVoice — Arabic/English Insurance Voice Assistant

Prototype voice AI demo for a healthcare insurance use case, using
OpenAI's Realtime API (WebRTC) with a FastAPI backend as the tool layer.

## Project structure

```
healthvoice/
  api/
    index.py           # FastAPI app (all backend routes, under /api/*)
    tools.py           # Business logic (patient lookup, coverage, etc.)
    tool_schemas.py     # Tool definitions passed to the Realtime API
    data/                # Mock JSON "database"
  public/
    index.html           # Frontend (WebRTC voice UI)
  vercel.json             # Vercel routing config
  requirements.txt        # Python dependencies
  .env.example             # Template for local environment variables
```

## Deploying to Vercel (global deployment)

### 1. Push this project to GitHub

```bash
cd healthvoice
git init
git add .
git commit -m "Initial commit: HealthVoice voice AI prototype"
```

Then create a new repo on GitHub (via github.com → New repository), and:

```bash
git remote add origin https://github.com/YOUR_USERNAME/healthvoice.git
git branch -M main
git push -u origin main
```

### 2. Import the project in Vercel

1. Go to vercel.com → **Add New... → Project**
2. Select your `healthvoice` GitHub repo
3. Vercel will auto-detect the `vercel.json` config — no need to change
   build settings

### 3. Add your environment variable

Your real OpenAI API key must **never** be committed to GitHub. Instead:

1. In the Vercel project → **Settings → Environment Variables**
2. Add a variable:
   - Key: `OPENAI_API_KEY`
   - Value: your real key (starts with `sk-...`)
3. Redeploy (Vercel usually does this automatically after adding env vars)

### 4. Done

Vercel will give you a URL like `https://healthvoice-yourname.vercel.app`.
Since the frontend (`public/index.html`) calls `/api/...` as a **relative
path**, everything works automatically on the same domain — no extra
configuration needed.

Open that URL on your phone or any browser worldwide, allow microphone
access, and press "Start Conversation".

## Local development (optional, for testing before deploying)

```bash
pip install -r requirements.txt fastapi uvicorn
cd api
cp ../.env.example ../.env   # then edit .env with your real key
uvicorn index:app --reload --port 8000
```

Note: locally, `public/index.html`'s `BACKEND_URL = "/api"` won't work
unless you also serve it through a proxy/dev server that maps `/api` to
port 8000. The simplest local test is via Vercel's own dev server:

```bash
npm install -g vercel
vercel dev
```

This runs the exact same routing locally as production.

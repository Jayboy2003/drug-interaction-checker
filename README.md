# 💊 Drug Interaction Checker Agent

> **CiphezNexus Hackathon project** — an autonomous multi-agent system that
> takes a patient's medication list, checks it for dangerous combinations, and
> returns a clear, prioritised **safety report**.

Built with the **OpenAI Agents SDK** (runtime/orchestration) and **CrewAI**
(agent roles & collaboration), served through a **FastAPI** backend and a small
web UI. Interaction data is a **bundled, offline knowledge base** so the demo
works with zero external API calls.

---

## ⚠️ Not medical advice
This tool is a **demonstration only**. It is **NOT medical advice** and does not
replace consultation with a qualified healthcare professional or a verified
clinical database (e.g. NIH RxNav, Lexicomp, Micromedex). Always confirm with
your doctor or pharmacist before starting, stopping, or changing any medication.

---

## Architecture

```
Medication list (UI / API)
        │
        ▼
┌───────────────────────────────┐
│  OpenAI Agents SDK (runtime)  │  guarded "report writer" agent
└───────────────┬───────────────┘
                │  calls
                ▼
┌───────────────────────────────┐
│   CrewAI crew (agent roles)   │  Researcher → Analyst → Writer → Reviewer
└───────────────┬───────────────┘
                │  uses
                ▼
┌───────────────────────────────┐
│  Drug Interaction Tool        │  queries data/interactions.json (offline)
└───────────────────────────────┘
                │
                ▼
        Safety Report (JSON + UI)
```

### Agent roles (CrewAI)
1. **Interaction Researcher** — retrieves candidate interactions from the tool.
2. **Clinical Interaction Analyst** — ranks risks by severity for *this* patient.
3. **Safety Report Writer** — drafts a clear, patient-friendly report.
4. **Safety Reviewer** — enforces the disclaimer & removes advisory language.

### Resilience
If no `OPENAI_API_KEY` is set (or the SDK is unavailable), the system silently
falls back to a **deterministic offline report** computed directly from the
knowledge base — so the demo never breaks.

---

## Quick start (local)

```bash
# 1. Clone / open the project
cd drug-interaction-checker

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI key
cp .env.example .env
#   then edit .env and paste your key

# 5. Run the server
uvicorn app.api.main:app --reload

# 6. Open the app
#   http://localhost:8000
```

### Try it from the terminal
```bash
python -m app.tools.interaction_tool "warfarin, aspirin, lisinopril, spironolactone"
```

### API usage
```bash
curl -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -d '{"medications": ["warfarin", "aspirin", "ibuprofen"]}'
```

---

## Deploy for the live link (hackathon submission)

### Option A — Render (one-click Blueprint) ⭐ recommended
1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New** → **Blueprint**.
3. Connect the GitHub repo. Render reads `render.yaml` automatically.
4. When prompted, paste your `OPENAI_API_KEY` (marked `sync: false`).
5. Deploy → use the generated `https://...onrender.com` as your live link.

   *(Manual alternative: New → Web Service → Docker → start command
   `uvicorn app.api.main:app --host 0.0.0.0 --port $PORT`, add `OPENAI_API_KEY`.)*

### Option B — Hugging Face Spaces
1. Create a Space (Docker SDK).
2. Upload the repo (the included `Dockerfile` works as-is) and set
   `OPENAI_API_KEY` in the Space's Secrets.

---

## Project structure
```
drug-interaction-checker/
├── app/
│   ├── agents_sdk.py          # OpenAI Agents SDK orchestration + fallback
│   ├── api/main.py            # FastAPI server + /check endpoint
│   ├── crew/agents.py         # CrewAI agents, roles, offline report
│   └── tools/interaction_tool.py  # offline lookup tool
├── data/interactions.json     # curated interaction knowledge base
├── frontend/index.html        # web UI
├── requirements.txt
├── Dockerfile                  # container image (for Render/HF)
├── render.yaml                 # one-click Render Blueprint
├── .env.example
└── README.md
```

---

## Submission checklist
- [ ] GitHub repo pushed
- [ ] `.env` with `OPENAI_API_KEY` (never commit the real key)
- [ ] Deployed live link (Render/HF)
- [ ] 2–3 min demo video (show: enter meds → agent runs → report + disclaimer)
- [ ] Submit the form with repo + link + video before **Thu 11:59 PM**

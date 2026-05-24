# 🧠 RecruitIQ — Multi-Intent AI Recruitment Agent

A production-grade AI agent that screens resumes, generates interview questions, compares candidates, and summarises applicant pipelines — all from natural language queries.

**Built with:** FastAPI · LangChain · ChromaDB · Sentence Transformers · Groq (LLaMA 3.3) · RAG

---

## 🎯 What Makes This an Agent (not just a chatbot)

A chatbot answers. An agent **plans and reasons**:

1. **Perceives** intent from natural language — no keyword matching
2. **Decides** which tool to call based on detected intent
3. **Retrieves** only relevant document chunks via semantic search (RAG)
4. **Reasons** over retrieved context using LLaMA 3.3 via Groq
5. **Returns** structured, actionable output with confidence scores

---

## 🏗️ Architecture

```
User Query (natural language)
        │
        ▼
┌─────────────────────┐
│   Intent Detector   │  ← LLaMA 3.3 classifies intent with confidence score
│                     │    screen / questions / compare / summarise / qa
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────────────────────────┐
│   Agent Router      │────▶│  RAG Engine                          │
└─────────────────────┘     │  1. Extract text (PDF / DOCX / TXT)  │
         │                  │  2. Chunk (600 chars, 100 overlap)    │
         │                  │  3. Embed locally (MiniLM-L6-v2)      │
         │                  │  4. Store in ChromaDB                 │
         │                  │  5. Retrieve top-K chunks by query    │
         │                  └──────────────────────────────────────┘
         ▼
┌──────────────────────────────────────┐
│  Agent Tools                         │
│  • screen_resume()                   │
│  • generate_interview_questions()    │
│  • compare_candidates()              │
│  • summarise_all_applicants()        │
│  • general_qa()                      │
└──────────────────────────────────────┘
         │
         ▼
  Structured JSON response
  rendered as rich UI cards
```

---

## ⚡ Quick Setup (5 minutes)

### 1. Clone the repository
```bash
git clone https://github.com/P1Rohini/recruitment-agent.git
cd recruitment-agent
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\Activate.ps1       # Windows PowerShell
```

### 3. Install dependencies
```bash
pip install fastapi==0.111.0 uvicorn==0.30.1 python-multipart==0.0.9 langchain==0.2.6 langchain-google-genai==1.0.6 langchain-community==0.2.6 langchain-chroma==0.1.2 chromadb==0.5.3 pymupdf==1.24.5 python-docx==1.1.2 pydantic==2.7.4 python-dotenv==1.0.1 "google-generativeai>=0.5.2,<0.6.0" sentence-transformers==3.0.1
```

### 4. Get your free Groq API key
- Go to **https://console.groq.com**
- Sign up → API Keys → Create API Key
- Copy the key starting with `gsk_...`

### 5. Create your .env file
```bash
cp .env.example .env
```
Open `.env` and add:
```
GROQ_API_KEY=gsk_your_key_here
```

### 6. Run the server
```bash
py main.py        # Windows
python main.py    # Linux/Mac
```

Visit: **http://localhost:8000**

On first run, it downloads the embedding model (~90MB). Subsequent runs are instant.

---

## 🎯 Intent Detection — 5 Supported Intents

| What you say | Intent detected | What happens |
|---|---|---|
| "Screen this resume against the JD" | `screen_resume` | Score 1-10 + strengths + gaps + Shortlist/Maybe/Reject |
| "Generate interview questions for this candidate" | `generate_questions` | 5 targeted questions with what to listen for |
| "Compare candidate A and B" | `compare_candidates` | Side-by-side across 4 dimensions + winner |
| "Give me a summary of all applicants" | `summarise_all` | Overview of every uploaded resume |
| "What are her key skills?" | `general_qa` | RAG-powered answer with confidence score |

---

## 📁 Project Structure

```
recruitment-agent/
├── main.py                    # FastAPI app + all API endpoints
├── backend/
│   ├── rag_engine.py          # Chunking, local embeddings, ChromaDB
│   ├── gemini_http.py         # Groq LLM calls (pure HTTP, no SDK)
│   ├── intent_detector.py     # LLaMA-powered intent classification
│   ├── agent_tools.py         # 5 agent tools
│   └── agent.py               # Orchestrator — intent → tool → response
├── frontend/
│   └── index.html             # Dark UI (served by FastAPI)
├── uploads/                   # Uploaded documents (auto-created)
├── vectorstore/               # ChromaDB persistence (auto-created)
├── requirements.txt
└── .env.example
```

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload PDF/DOCX/TXT document |
| `GET` | `/api/documents` | List all ingested documents |
| `DELETE` | `/api/documents/{id}` | Remove a document |
| `POST` | `/api/agent` | Query the agent (main endpoint) |
| `GET` | `/api/health` | Health check |

### Agent request format
```json
{
  "message": "Screen this resume against the job description",
  "resume_doc_id": "abc12345",
  "jd_doc_id": "xyz67890",
  "resume_doc_id_2": null
}
```

### Agent response format
```json
{
  "intent": "screen_resume",
  "confidence": 0.9,
  "intent_reasoning": "User wants to evaluate a resume",
  "data": {
    "overall_score": 9,
    "summary": "Strong candidate...",
    "strengths": ["..."],
    "gaps": ["..."],
    "recommendation": "Shortlist",
    "recommendation_reason": "..."
  }
}
```

---

## 🧠 Chunking Strategy

| Parameter | Value | Why |
|---|---|---|
| Chunk size | 600 characters | Large enough for semantic context, small enough for precision |
| Overlap | 100 characters | Prevents losing meaning at chunk boundaries |
| Splitter | RecursiveCharacterTextSplitter | Tries paragraph → sentence → word, never cuts mid-thought |
| Embedding | all-MiniLM-L6-v2 (local) | Free, fast on CPU, 384-dim vectors, no API needed |

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Backend | FastAPI | Async, fast, you already know it |
| LLM | LLaMA 3.3 70B via Groq | Free tier, faster than Gemini, no gRPC issues |
| Embeddings | all-MiniLM-L6-v2 (local) | Runs on CPU, no API key needed, production-grade |
| Vector Store | ChromaDB | Free, runs locally, persists to disk |
| Document parsing | PyMuPDF + python-docx | Best PDF/Word text extraction |
| Orchestration | LangChain | Standard RAG tooling |
| Frontend | Vanilla HTML/CSS/JS | Zero dependencies, served by FastAPI |

---

## ✅ Live Demo Results

All 5 intents tested and working:

- **Screen resume** → 9/10 score for MCA candidate vs AI/ML JD
- **Interview questions** → 5 targeted questions (Technical, Behavioral, Situational)
- **General Q&A** → Accurate answers about skills, projects, education
- **Summarise all** → Pipeline overview with top skills extracted
- **Compare candidates** → Side-by-side winner recommendation

---
## ✅ Expected Output on Startup

When you run `py main.py`, you should see:
[RAG] Loading local embedding model (all-MiniLM-L6-v2)...

[RAG] First run will download ~90MB — please wait...

[RAG] Embedding model ready ✓

[LLM] Using Groq — model: llama-3.3-70b-versatile

INFO: Uvicorn running on http://0.0.0.0:8000

INFO: Application startup complete.

Then open **http://localhost:8000** in your browser.

## 🔮 Future Improvements

- Batch processing for 100+ resumes at once
- Email integration to auto-import resumes
- Candidate ranking leaderboard
- Export screening reports as PDF
- Multi-language resume support

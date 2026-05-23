"""
FastAPI Application — Recruitment Intelligence Agent
All endpoints for document upload, management, and agent queries.
"""

import traceback
import os
import uuid
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.rag_engine import ingest_document, list_documents, delete_document
from backend.agent import run_agent

# ── App Setup ──────────────────────────────────────────
app = FastAPI(
    title="Recruitment Intelligence Agent",
    description="Multi-intent RAG agent for screening, comparing, and analysing candidates.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


# ── Document Endpoints ─────────────────────────────────

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form(...),  # "resume" or "job_description"
):
    """Upload and ingest a document into the vector store."""
    allowed_types = [".pdf", ".docx", ".doc", ".txt"]
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_types:
        raise HTTPException(400, f"Unsupported file type: {ext}. Use PDF, DOCX, or TXT.")

    doc_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"

    # Save file to disk
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingest_document(str(save_path), doc_type, doc_id)
        return {
            "success": True,
            "doc_id": doc_id,
            "file_name": file.filename,
            "doc_type": doc_type,
            "chunk_count": result["chunk_count"],
            "message": f"Successfully ingested {result['chunk_count']} chunks from {file.filename}",
        }
    except Exception as e:
        traceback.print_exc()   # <-- add this line
        save_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Ingestion failed: {str(e)}")


@app.get("/api/documents")
async def get_documents():
    """List all ingested documents."""
    docs = list_documents()
    return {"documents": docs, "count": len(docs)}


@app.delete("/api/documents/{doc_id}")
async def remove_document(doc_id: str):
    """Remove a document from the vector store."""
    success = delete_document(doc_id)
    if not success:
        raise HTTPException(404, f"Document {doc_id} not found.")
    return {"success": True, "message": f"Document {doc_id} removed."}


# ── Agent Endpoint ─────────────────────────────────────

class AgentRequest(BaseModel):
    message: str
    resume_doc_id: Optional[str] = None
    resume_doc_id_2: Optional[str] = None
    jd_doc_id: Optional[str] = None


@app.post("/api/agent")
async def query_agent(req: AgentRequest):
    """
    Main agent endpoint.
    Detects intent from message and routes to the appropriate tool.
    """
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty.")

    try:
        result = run_agent(
            message=req.message,
            resume_doc_id=req.resume_doc_id,
            resume_doc_id_2=req.resume_doc_id_2,
            jd_doc_id=req.jd_doc_id,
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Agent error: {str(e)}")


# ── Health Check ───────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "Recruitment Intelligence Agent"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

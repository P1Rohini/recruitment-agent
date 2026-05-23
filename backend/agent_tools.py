"""
Agent Tools — The 4 core capabilities of the Recruitment Intelligence Agent.
Each tool retrieves relevant chunks via RAG, then calls Gemini to reason over them.
All Gemini calls use pure HTTPS — no gRPC, no SDK.
"""

import json
import re
from typing import List, Dict, Any, Optional
from backend.rag_engine import retrieve_chunks, list_documents
from backend.gemini_http import call_gemini


def _format_chunks(chunks) -> str:
    """Turn retrieved LangChain Documents into a readable context block."""
    parts = []
    for i, doc in enumerate(chunks):
        fname = doc.metadata.get("file_name", "unknown")
        parts.append(f"[Source {i+1} — {fname}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── TOOL 1: Screen Resume ──────────────────────────────
def screen_resume(resume_doc_id: str, jd_doc_id: str) -> Dict[str, Any]:
    resume_chunks = retrieve_chunks("skills experience education projects", doc_id=resume_doc_id, k=6)
    jd_chunks = retrieve_chunks("required skills qualifications responsibilities", doc_id=jd_doc_id, k=4)

    resume_context = _format_chunks(resume_chunks)
    jd_context = _format_chunks(jd_chunks)

    prompt = f"""
You are an expert technical recruiter. Analyse the candidate's resume against the job description below.

=== JOB DESCRIPTION ===
{jd_context}

=== CANDIDATE RESUME ===
{resume_context}

Provide a structured evaluation in EXACTLY this JSON format (no extra text, no markdown):
{{
  "overall_score": <integer 1-10>,
  "summary": "<2-3 sentence candidate summary>",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "gaps": ["<gap 1>", "<gap 2>"],
  "recommendation": "Shortlist" or "Maybe" or "Reject",
  "recommendation_reason": "<one clear sentence>"
}}
"""
    raw = call_gemini(prompt)
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"error": "Failed to parse screening result", "raw": raw}


# ── TOOL 2: Generate Interview Questions ───────────────
def generate_interview_questions(
    resume_doc_id: str,
    jd_doc_id: Optional[str] = None,
    question_count: int = 5,
) -> Dict[str, Any]:
    resume_chunks = retrieve_chunks("skills experience projects achievements", doc_id=resume_doc_id, k=6)
    resume_context = _format_chunks(resume_chunks)

    jd_context = ""
    if jd_doc_id:
        jd_chunks = retrieve_chunks("responsibilities required skills", doc_id=jd_doc_id, k=3)
        jd_context = f"\n=== JOB DESCRIPTION ===\n{_format_chunks(jd_chunks)}"

    prompt = f"""
You are a senior technical interviewer. Based on the candidate's resume, generate {question_count} targeted interview questions.

=== CANDIDATE RESUME ===
{resume_context}
{jd_context}

Return ONLY a JSON object (no markdown, no extra text):
{{
  "candidate_name": "<name if found, else Candidate>",
  "questions": [
    {{
      "category": "Technical" or "Behavioral" or "Situational" or "Gap",
      "question": "<the interview question>",
      "what_to_listen_for": "<what a good answer should contain>"
    }}
  ]
}}
"""
    raw = call_gemini(prompt, temperature=0.5)
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"error": "Failed to parse questions", "raw": raw}


# ── TOOL 3: Compare Candidates ─────────────────────────
def compare_candidates(
    resume_doc_id_1: str,
    resume_doc_id_2: str,
    jd_doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    query = "skills experience education projects achievements"
    chunks_1 = retrieve_chunks(query, doc_id=resume_doc_id_1, k=5)
    chunks_2 = retrieve_chunks(query, doc_id=resume_doc_id_2, k=5)

    context_1 = _format_chunks(chunks_1)
    context_2 = _format_chunks(chunks_2)

    jd_context = ""
    if jd_doc_id:
        jd_chunks = retrieve_chunks("required skills responsibilities", doc_id=jd_doc_id, k=3)
        jd_context = f"\n=== JOB DESCRIPTION ===\n{_format_chunks(jd_chunks)}"

    prompt = f"""
You are a senior recruiter comparing two candidates.

=== CANDIDATE A ===
{context_1}

=== CANDIDATE B ===
{context_2}
{jd_context}

Return ONLY a JSON object (no markdown):
{{
  "candidate_a_name": "<name or Candidate A>",
  "candidate_b_name": "<name or Candidate B>",
  "comparison": {{
    "technical_skills": {{"winner": "A" or "B" or "Tie", "reason": "<reason>"}},
    "experience_depth": {{"winner": "A" or "B" or "Tie", "reason": "<reason>"}},
    "education": {{"winner": "A" or "B" or "Tie", "reason": "<reason>"}},
    "overall_fit": {{"winner": "A" or "B" or "Tie", "reason": "<reason>"}}
  }},
  "final_recommendation": "A" or "B" or "Either",
  "recommendation_reason": "<clear reasoning>"
}}
"""
    raw = call_gemini(prompt)
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"error": "Failed to parse comparison", "raw": raw}


# ── TOOL 4: Summarise All Applicants ──────────────────
def summarise_all_applicants(jd_doc_id: Optional[str] = None) -> Dict[str, Any]:
    docs = list_documents()
    resumes = [d for d in docs if d["doc_type"] == "resume"]

    if not resumes:
        return {"error": "No resumes uploaded yet."}

    summaries = []
    for resume in resumes:
        chunks = retrieve_chunks("skills experience education summary", doc_id=resume["doc_id"], k=4)
        context = _format_chunks(chunks)

        prompt = f"""
Summarise this candidate in 2-3 sentences. Focus on their role, key skills, and standout experience.
Return ONLY a JSON object: {{"name": "<name or file>", "summary": "<summary>", "top_skills": ["skill1", "skill2", "skill3"]}}

Resume content:
{context}
"""
        raw = call_gemini(prompt, temperature=0.2)
        raw = re.sub(r"```json|```", "", raw).strip()
        try:
            summaries.append(json.loads(raw))
        except Exception:
            summaries.append({"name": resume["file_name"], "summary": raw, "top_skills": []})

    return {"total_applicants": len(resumes), "applicants": summaries}


# ── TOOL 5: General Q&A ────────────────────────────────
def general_qa(question: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
    chunks = retrieve_chunks(question, doc_id=doc_id, k=5)
    context = _format_chunks(chunks)

    prompt = f"""
Answer the following question based only on the provided document context.
If the answer cannot be found, say so clearly.

Question: {question}

Context:
{context}

Return a JSON object:
{{"answer": "<detailed answer>", "confidence": "High" or "Medium" or "Low", "sources": ["<file1>"]}}
"""
    raw = call_gemini(prompt)
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"answer": raw, "confidence": "Low", "sources": []}
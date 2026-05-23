"""
Agent Orchestrator — The brain of the system.
Detects intent → routes to the right tool → returns structured result.
This is what makes the system an AGENT.
"""

from typing import Optional, Dict, Any
from backend.intent_detector import detect_intent
from backend.agent_tools import (
    screen_resume,
    generate_interview_questions,
    compare_candidates,
    summarise_all_applicants,
    general_qa,
)


def run_agent(
    message: str,
    resume_doc_id: Optional[str] = None,
    resume_doc_id_2: Optional[str] = None,
    jd_doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main agent entry point.

    Flow:
    1. Detect intent from user message
    2. Route to the appropriate tool
    3. Return structured result with intent metadata

    Args:
        message: Natural language query from the user
        resume_doc_id: Primary resume to work with
        resume_doc_id_2: Second resume (for comparison)
        jd_doc_id: Job description doc ID
    """

    # Step 1: Understand what the user wants
    intent_result = detect_intent(message)
    intent = intent_result.get("intent", "unknown")
    confidence = intent_result.get("confidence", 0)

    # Step 2: Route to the right tool
    result = {}

    if intent == "screen_resume":
        if not resume_doc_id or not jd_doc_id:
            result = {
                "error": "Please select both a resume and a job description to screen against."
            }
        else:
            result = screen_resume(resume_doc_id, jd_doc_id)

    elif intent == "generate_questions":
        if not resume_doc_id:
            result = {"error": "Please select a resume to generate questions for."}
        else:
            result = generate_interview_questions(resume_doc_id, jd_doc_id)

    elif intent == "compare_candidates":
        if not resume_doc_id or not resume_doc_id_2:
            result = {"error": "Please select two resumes to compare."}
        else:
            result = compare_candidates(resume_doc_id, resume_doc_id_2, jd_doc_id)

    elif intent == "summarise_all":
        result = summarise_all_applicants(jd_doc_id)

    elif intent == "general_qa":
        result = general_qa(message, doc_id=resume_doc_id)

    else:
        # Fallback: treat as general Q&A
        result = general_qa(message, doc_id=resume_doc_id)
        intent = "general_qa"

    # Step 3: Return with metadata so the frontend can render intelligently
    return {
        "intent": intent,
        "confidence": confidence,
        "intent_reasoning": intent_result.get("reasoning", ""),
        "data": result,
    }

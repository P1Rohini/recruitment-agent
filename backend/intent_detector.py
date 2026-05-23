"""
Intent Detection Layer
Classifies user queries into one of 4 agent intents before routing to the right tool.
"""

import json
import re
from typing import Literal
from backend.gemini_http import call_gemini

IntentType = Literal[
    "screen_resume",
    "generate_questions",
    "compare_candidates",
    "summarise_all",
    "general_qa",
    "unknown",
]

INTENT_PROMPT = """
You are an intent classifier for an AI recruitment assistant.
Given a user message, classify it into EXACTLY ONE of these intents:

1. screen_resume       — User wants to evaluate/score a resume against a job description
2. generate_questions  — User wants interview questions for a candidate
3. compare_candidates  — User wants to compare two or more candidates
4. summarise_all       — User wants a summary/overview of all uploaded resumes
5. general_qa          — User asks a general question about a document or candidate
6. unknown             — Cannot determine intent

Rules:
- Return ONLY a valid JSON object, no extra text, no markdown.
- Format: {{"intent": "<intent_name>", "confidence": <0.0-1.0>, "reasoning": "<one line>"}}

User message: "{message}"
"""


def detect_intent(message: str) -> dict:
    """Uses Gemini to classify user intent via pure HTTP."""
    prompt = INTENT_PROMPT.format(message=message)
    try:
        raw = call_gemini(prompt, temperature=0.1)
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        assert "intent" in result and "confidence" in result
        return result
    except Exception as e:
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "reasoning": f"Classification failed: {str(e)}",
        }
"""
LLM Helper — uses Groq API (free, fast, no gRPC issues).
Model: llama3-8b-8192 (free tier, very capable)
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

print(f"[LLM] Using Groq — model: {MODEL}")


def call_gemini(prompt: str, temperature: float = 0.3) -> str:
    """
    Call Groq LLM. Named call_gemini for drop-in compatibility.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 2048,
    }
    resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise ValueError(f"Groq API error {resp.status_code}: {resp.text}")
    return resp.json()["choices"][0]["message"]["content"].strip()
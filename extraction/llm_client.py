"""LLM client abstraction. Supports Ollama (default) and OpenAI."""
import json
import logging
import os

logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def extract_structured(prompt: str) -> dict:
    """Single entry point -- returns parsed dict from LLM response."""
    if LLM_PROVIDER == "openai":
        return _extract_openai(prompt)
    return _extract_ollama(prompt)


def _extract_ollama(prompt: str) -> dict:
    """Call Ollama API. Lazy-imports httpx."""
    import httpx

    response = httpx.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
        timeout=120,
    )
    response.raise_for_status()
    return json.loads(response.json()["response"])


def _extract_openai(prompt: str) -> dict:
    """Call OpenAI API. Lazy-imports openai."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    completion = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(completion.choices[0].message.content)

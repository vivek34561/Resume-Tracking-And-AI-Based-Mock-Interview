"""LLM Provider utilities for Groq and Ollama."""

import os
import re
import json
import time
import requests

# Reuse a single HTTP session for all outbound requests
SESSION = requests.Session()


def groq_chat(api_key: str, messages: list, model: str = None, temperature: float = 0.2, max_tokens: int = 600) -> str:
    """Minimal Groq chat-completions helper returning assistant content as text.

    Token-optimized: enforce a single model (llama-3.1-8b-instant) with no fallbacks.
    """
    if not api_key:
        raise RuntimeError("Groq API key missing")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    model = (model or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant")
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    # Simple retry/backoff for rate limits and transient errors
    attempts = 0
    last_err = None
    while attempts < 2:
        resp = SESSION.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 429:
            # Parse suggested wait from message, else sleep a small backoff
            wait_s = 2.5
            try:
                msg = resp.text or ""
                m = re.search(r"try again in\s([\d\.]+)s", msg)
                if m:
                    wait_s = min(max(float(m.group(1)) * 1.2, 2.0), 8.0)
            except Exception:
                pass
            time.sleep(wait_s)
            attempts += 1
            last_err = resp
            continue
        if 500 <= resp.status_code < 600:
            # transient server error
            time.sleep(1.5 * (attempts + 1))
            attempts += 1
            last_err = resp
            continue
        if resp.status_code >= 400:
            raise requests.HTTPError(f"{resp.status_code} {resp.reason}: {resp.text}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(data)
    # if we exhausted retries
    if last_err is not None:
        raise requests.HTTPError(f"{last_err.status_code} {last_err.reason}: {last_err.text}")
    raise RuntimeError("Groq request failed after retries")


def ollama_chat(messages: list, model: str, base_url: str = None, temperature: float = 0.2) -> str:
    """Call local Ollama chat; if /api/chat is not available (404), fallback to /api/generate."""
    url_base = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    headers = {"Content-Type": "application/json"}

    def _messages_to_prompt(msgs: list) -> str:
        parts = []
        for m in msgs:
            role = (m.get("role") or "user").lower()
            if role == "system": label = "System"
            elif role == "assistant": label = "Assistant"
            else: label = "User"
            parts.append(f"{label}: {m.get('content','')}")
        parts.append("Assistant:")
        return "\n".join(parts)

    chat_url = f"{url_base}/api/chat"
    chat_payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature}
    }
    resp = SESSION.post(chat_url, headers=headers, json=chat_payload, timeout=120)
    if resp.status_code == 404:
        def _try_generate(base):
            gen_url = f"{base}/api/generate"
            prompt = _messages_to_prompt(messages)
            gen_payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature}
            }
            gr = SESSION.post(gen_url, headers=headers, json=gen_payload, timeout=120)
            # If model isn't pulled, Ollama returns 404 Not Found with a helpful message
            if gr.status_code == 404:
                try:
                    err_txt = gr.text
                except Exception:
                    err_txt = ""
                raise requests.HTTPError(
                    f"Ollama model not found: '{model}'. Pull it first: ollama pull {model}. Server said: {err_txt}"
                )
            gr.raise_for_status()
            gd = gr.json()
            return gd.get("response", json.dumps(gd))
        try:
            return _try_generate(url_base)
        except Exception:
            alt_base = ("http://127.0.0.1:11434" if "localhost" in url_base else url_base)
            return _try_generate(alt_base)
    resp.raise_for_status()
    data = resp.json()
    msg = data.get("message") or {}
    content = msg.get("content")
    if content:
        return content
    try:
        return data.get("messages", [{}])[-1].get("content", "")
    except Exception:
        return json.dumps(data)

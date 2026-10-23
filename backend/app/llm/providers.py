"""Raw HTTP calls to Groq, Gemini and Ollama. No SDKs, so install stays trivial.

Every function raises ProviderError with a `kind` the client uses to decide
what to do next:
    rate_limit  -> temporary (or daily) quota hit: cool down, try next endpoint
    server      -> 5xx / timeout / network: short cooldown, try next endpoint
    auth        -> bad key: disable this endpoint for the rest of the process
    config      -> model not found etc.: disable this endpoint
    bad_request -> provider refused this prompt: try next endpoint, no cooldown
"""
from __future__ import annotations

import re

import requests

from .config import Endpoint

GROQ_URL = "https://api.groq.com/openai/v1"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"


class ProviderError(Exception):
    def __init__(self, message: str, kind: str, retry_after: float | None = None,
                 daily: bool = False):
        super().__init__(message)
        self.kind = kind
        self.retry_after = retry_after
        self.daily = daily


def _retry_after(headers) -> float | None:
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
        return float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def _raise_for_response(resp) -> None:
    status = resp.status_code
    if status < 400:
        return
    body = (getattr(resp, "text", "") or "")[:500]
    low = body.lower()
    msg = f"HTTP {status}: {body[:200]}"
    if status in (401, 403):
        raise ProviderError(msg, "auth")
    if status == 400 and any(s in low for s in ("api_key_invalid", "api key not valid", "invalid api key")):
        raise ProviderError(msg, "auth")
    if status == 404:
        raise ProviderError(msg, "config")
    if status == 429:
        daily = any(s in low for s in ("perday", "per day", "daily"))
        raise ProviderError(msg, "rate_limit", _retry_after(resp.headers), daily)
    if status >= 500:
        raise ProviderError(msg, "server", _retry_after(resp.headers))
    raise ProviderError(msg, "bad_request")


def _post(url: str, *, headers: dict, payload: dict, timeout: float):
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise ProviderError(f"timeout: {exc}", "server") from exc
    except requests.exceptions.RequestException as exc:
        raise ProviderError(f"network error: {exc}", "server") from exc
    _raise_for_response(resp)
    try:
        return resp.json()
    except ValueError as exc:
        raise ProviderError("provider returned non-JSON body", "server") from exc


def _get(url: str, *, headers: dict, timeout: float):
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        raise ProviderError(f"network error: {exc}", "server") from exc
    _raise_for_response(resp)
    return resp.json()


# ---------------------------------------------------------------- completion

def _groq(ep: Endpoint, system, prompt, json_mode, temperature, max_tokens) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}]
    payload = {"model": ep.model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    data = _post(f"{GROQ_URL}/chat/completions",
                 headers={"Authorization": f"Bearer {ep.api_key}"},
                 payload=payload, timeout=ep.timeout)
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"unexpected Groq response shape: {str(data)[:200]}", "server") from exc


def _gemini(ep: Endpoint, system, prompt, json_mode, temperature, max_tokens) -> str:
    config: dict = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if json_mode:
        config["responseMimeType"] = "application/json"
    if "thinking_budget" in ep.extra:
        config["thinkingConfig"] = {"thinkingBudget": ep.extra["thinking_budget"]}
    elif json_mode and re.search(r"2\.5-flash", ep.model or ""):
        # Gemini 2.5 Flash "thinks" first and those tokens count against maxOutputTokens, which cut long JSON
        # replies off mid-way. Structured extraction does not need thinking, so it is off unless .env says otherwise.
        config["thinkingConfig"] = {"thinkingBudget": 0}
    payload: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": config,
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    data = _post(f"{GEMINI_URL}/models/{ep.model}:generateContent",
                 headers={"x-goog-api-key": ep.api_key or ""},  # header, so it never lands in URLs/logs
                 payload=payload, timeout=ep.timeout)
    candidates = data.get("candidates") or []
    if not candidates:
        feedback = data.get("promptFeedback") or {}
        raise ProviderError(f"Gemini returned no candidates: {str(feedback)[:200]}", "bad_request")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts)


def _ollama(ep: Endpoint, system, prompt, json_mode, temperature, max_tokens) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}]
    payload = {"model": ep.model, "messages": messages, "stream": False,
               "options": {"temperature": temperature, "num_predict": max_tokens}}
    if json_mode:
        payload["format"] = "json"
    data = _post(f"{ep.host}/api/chat", headers={}, payload=payload, timeout=ep.timeout)
    try:
        return data["message"]["content"] or ""
    except (KeyError, TypeError) as exc:
        raise ProviderError(f"unexpected Ollama response shape: {str(data)[:200]}", "server") from exc


_DISPATCH = {"groq": _groq, "gemini": _gemini, "ollama": _ollama}


def complete(ep: Endpoint, system, prompt, json_mode, temperature, max_tokens) -> str:
    return _DISPATCH[ep.provider](ep, system, prompt, json_mode, temperature, max_tokens)


# ------------------------------------------------------------- model listing

def list_models(ep: Endpoint) -> list[str]:
    """Used by the health-check CLI to show which model names this key can use."""
    if ep.provider == "groq":
        data = _get(f"{GROQ_URL}/models", headers={"Authorization": f"Bearer {ep.api_key}"},
                    timeout=ep.timeout)
        return sorted(m["id"] for m in data.get("data", []))
    if ep.provider == "gemini":
        data = _get(f"{GEMINI_URL}/models?pageSize=200", headers={"x-goog-api-key": ep.api_key or ""},
                    timeout=ep.timeout)
        return sorted(
            m["name"].removeprefix("models/") for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", []))
    if ep.provider == "ollama":
        data = _get(f"{ep.host}/api/tags", headers={}, timeout=ep.timeout)
        return sorted(m["name"] for m in data.get("models", []))
    return []

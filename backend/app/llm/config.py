"""Reads provider settings from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import MutableMapping

BACKEND_DIR = Path(__file__).resolve().parents[2]

DEFAULT_ORDER = "groq,gemini,ollama"
DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile,llama-3.1-8b-instant",
    "gemini": "gemini-2.5-flash,gemini-2.5-flash-lite",
    "ollama": "llama3.2:3b",
}
KEY_ENV_PREFIX = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}
PLACEHOLDER_PREFIXES = ("your_", "paste", "xxx", "changeme")


@dataclass
class Endpoint:
    """One concrete thing we can call: a provider + key + model."""

    provider: str  # groq | gemini | ollama
    label: str  # unique, safe to log, e.g. "gemini:key2:gemini-2.5-flash"
    model: str
    api_key: str | None = None
    host: str | None = None
    timeout: float = 30.0
    extra: dict = field(default_factory=dict)


def load_dotenv(path: str | os.PathLike | None = None,
                environ: MutableMapping[str, str] | None = None) -> None:
    """Tiny .env loader. Real environment variables always win over the file."""
    environ = os.environ if environ is None else environ
    env_path = Path(path) if path else BACKEND_DIR / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and name not in environ:
            environ[name] = value


def _csv(value: str | None, default: str) -> list[str]:
    raw = value if value and value.strip() else default
    return [part.strip() for part in raw.split(",") if part.strip()]


def _looks_like_placeholder(value: str) -> bool:
    return value.lower().startswith(PLACEHOLDER_PREFIXES)


MAX_KEYS = 30   # GROQ_API_KEY1 .. GROQ_API_KEY30


def collect_keys(provider: str, environ: MutableMapping[str, str]) -> list[tuple[str, str]]:
    """Returns [(key_name, key_value)] for e.g. GEMINI_API_KEY, GEMINI_API_KEY1..30."""
    prefix = KEY_ENV_PREFIX[provider]
    names = [prefix] + [f"{prefix}{i}" for i in range(1, MAX_KEYS + 1)]
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in names:
        value = (environ.get(name) or "").strip()
        if not value or _looks_like_placeholder(value) or value in seen:
            continue
        seen.add(value)
        found.append((name.replace(prefix, "key").lower().replace("api_", ""), value))
    return found


def build_endpoints(environ: MutableMapping[str, str] | None = None) -> list[Endpoint]:
    """Builds the ordered list of endpoints to try.

    Order = provider order, then model (best first), then key. Quotas are per
    model and per key/project, so every extra model or key adds headroom.
    """
    environ = os.environ if environ is None else environ
    order = [p.lower() for p in _csv(environ.get("LLM_PROVIDER_ORDER"), DEFAULT_ORDER)]
    timeout = float(environ.get("LLM_TIMEOUT") or 30)
    local_timeout = float(environ.get("LLM_OLLAMA_TIMEOUT") or 120)
    endpoints: list[Endpoint] = []

    for provider in order:
        if provider in ("groq", "gemini"):
            keys = collect_keys(provider, environ)
            models = _csv(
                environ.get(f"{provider.upper()}_MODELS") or environ.get(f"{provider.upper()}_MODEL"),
                DEFAULT_MODELS[provider],
            )
            extra: dict = {}
            if provider == "gemini" and (environ.get("GEMINI_THINKING_BUDGET") or "").strip():
                extra["thinking_budget"] = int(environ["GEMINI_THINKING_BUDGET"])
            for model in models:
                for key_name, key in keys:
                    endpoints.append(Endpoint(
                        provider=provider,
                        label=f"{provider}:{key_name}:{model}",
                        model=model,
                        api_key=key,
                        timeout=timeout,
                        extra=dict(extra),
                    ))
        elif provider == "ollama":
            host = (environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
            for model in _csv(environ.get("OLLAMA_MODEL"), DEFAULT_MODELS["ollama"]):
                endpoints.append(Endpoint(
                    provider="ollama",
                    label=f"ollama:{model}",
                    model=model,
                    host=host,
                    timeout=local_timeout,
                ))
    return endpoints


def mask(secret: str | None) -> str:
    if not secret:
        return "-"
    return "..." + secret[-4:] if len(secret) > 8 else "***"

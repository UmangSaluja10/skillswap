"""LLMClient: one entry point for every AI call in SkillSwap.

    from app.llm import call_llm, call_llm_json

Tries endpoints in order (Groq -> Gemini keys/models -> Ollama). Rate-limited
endpoints are put on cooldown, bad keys are switched off, and successful
answers can be cached on disk so the demo never depends on a live API call.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Type, TypeVar

from pydantic import BaseModel, ValidationError

from . import providers
from .cache import DiskCache, make_key
from .config import BACKEND_DIR, Endpoint, build_endpoints, load_dotenv
from .providers import ProviderError

log = logging.getLogger("skillswap.llm")
T = TypeVar("T", bound=BaseModel)

JSON_SUFFIX = "Respond with a single valid JSON value only. No prose, no markdown fences."
SHORT_COOLDOWN = 20.0  # server errors / timeouts
RATE_LIMIT_COOLDOWN = 60.0  # per-minute limits when the provider gives no Retry-After
DAILY_COOLDOWN = 3600.0  # per-day quota hit: re-check in an hour
MAX_COOLDOWN = 3600.0
FOREVER = float("inf")


class LLMUnavailable(Exception):
    """Every endpoint failed or is cooling down."""

    def __init__(self, attempts: list[tuple[str, str]], retry_in: float | None = None):
        self.attempts = attempts
        self.retry_in = retry_in
        summary = ", ".join(f"{label}={kind}" for label, kind in attempts) or "no endpoints configured"
        super().__init__(f"No LLM provider could answer ({summary})")


class LLMInvalidOutput(Exception):
    """The model kept returning JSON that fails validation."""


@dataclass
class LLMResult:
    text: str
    provider: str
    label: str
    model: str
    latency_s: float
    cached: bool = False
    failed_attempts: list[tuple[str, str]] = field(default_factory=list)


def looks_truncated(text: str) -> bool:
    """True when a JSON reply stops mid-way (the model ran out of output tokens)."""
    s = re.sub(r"```\s*$", "", text.strip()).strip()
    return bool(s) and not s.endswith(("}", "]"))


MAX_RETRY_TOKENS = 8192


def extract_json(text: str):
    """Parses JSON even if the model wrapped it in ```json fences or added prose."""
    s = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()  # reasoning models
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S | re.I)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (s.find("{"), s.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON found in model reply")
    start = min(starts)
    end = s.rfind("}" if s[start] == "{" else "]")
    if end <= start:
        raise ValueError("unterminated JSON in model reply")
    return json.loads(s[start:end + 1])


class LLMClient:
    def __init__(self, endpoints: list[Endpoint], cache: DiskCache | None = None,
                 clock: Callable[[], float] = time.monotonic, max_wait: float = 0.0,
                 sleep: Callable[[float], None] = time.sleep, rotate: bool = False):
        self.endpoints = endpoints
        # rotate=True: keys of the same provider+model take turns (round robin), so one key does not
        # absorb every call and hit its rate limit first. Keys in cooldown are still skipped.
        self.rotate = rotate
        self._turn: dict[tuple[str, str], int] = {}
        self.cache = cache or DiskCache(None)
        self._clock = clock
        self._sleep = sleep
        # If every endpoint is briefly rate-limited, wait up to this many seconds for the
        # first one to recover instead of failing (per-minute limits clear in seconds).
        self.max_wait = max_wait
        self._cooldown_until: dict[str, float] = {}
        self._disabled: dict[str, str] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, load_env_file: bool = True) -> "LLMClient":
        if environ is None:
            if load_env_file:
                load_dotenv()
            environ = os.environ
        cache_path = None
        if (environ.get("LLM_CACHE", "1") or "1").strip() != "0":
            cache_path = environ.get("LLM_CACHE_PATH") or str(BACKEND_DIR / ".llm_cache.sqlite")
        max_wait = float((environ.get("LLM_MAX_WAIT") or "25").strip())
        rotate = (environ.get("LLM_ROTATE_KEYS", "1") or "1").strip() != "0"
        return cls(build_endpoints(environ), DiskCache(cache_path), max_wait=max_wait, rotate=rotate)

    # ---------------------------------------------------- key rotation ---
    def _ordered(self) -> list[Endpoint]:
        """Endpoints in the order to try now: provider and model order is kept, and within each
        provider+model group the keys start from whoever is next in turn."""
        if not self.rotate:
            return self.endpoints
        eps, out, i = self.endpoints, [], 0
        while i < len(eps):
            gid = (eps[i].provider, eps[i].model)
            j = i
            while j < len(eps) and (eps[j].provider, eps[j].model) == gid:
                j += 1
            group = eps[i:j]
            if len(group) > 1:
                with self._lock:
                    start = self._turn.get(gid, 0) % len(group)
                group = group[start:] + group[:start]
            out.extend(group)
            i = j
        return out

    def _advance(self, ep: Endpoint) -> None:
        """The key after the one that just answered goes first next time."""
        if not self.rotate:
            return
        gid = (ep.provider, ep.model)
        group = [e for e in self.endpoints if (e.provider, e.model) == gid]
        if len(group) > 1:
            with self._lock:
                self._turn[gid] = (group.index(ep) + 1) % len(group)

    # ------------------------------------------------------------ state ---
    def status(self) -> list[dict]:
        """For a /health endpoint or the CLI: what is usable right now."""
        now = self._clock()
        rows = []
        for ep in self.endpoints:
            wait = max(0.0, self._cooldown_until.get(ep.label, 0.0) - now)
            if ep.label in self._disabled:
                state = f"disabled ({self._disabled[ep.label]})"
            elif wait > 0:
                state = f"cooling down {wait:.0f}s"
            else:
                state = "ready"
            rows.append({"label": ep.label, "state": state})
        return rows

    def _penalize(self, ep: Endpoint, err: ProviderError) -> None:
        with self._lock:
            if err.kind in ("auth", "config"):
                self._disabled[ep.label] = err.kind
                log.warning("LLM endpoint %s disabled (%s): %s", ep.label, err.kind, str(err)[:160])
                return
            if err.kind == "rate_limit":
                wait = DAILY_COOLDOWN if err.daily else (err.retry_after or RATE_LIMIT_COOLDOWN)
            elif err.kind == "server":
                wait = err.retry_after or SHORT_COOLDOWN
            else:
                return  # bad_request: no cooldown
            wait = min(wait, MAX_COOLDOWN)
            self._cooldown_until[ep.label] = self._clock() + wait
            log.warning("LLM endpoint %s %s, cooling down %.0fs", ep.label, err.kind, wait)

    # ------------------------------------------------------------- core ---
    def call_llm(self, prompt: str, *, system: str | None = None, json_mode: bool = False,
                 temperature: float = 0.2, max_tokens: int = 2048,
                 use_cache: bool | None = None) -> LLMResult:
        if json_mode:
            system = f"{system}\n{JSON_SUFFIX}" if system else JSON_SUFFIX
        if use_cache is None:
            use_cache = temperature <= 0.3
        cache_key = make_key("text", system, prompt, json=json_mode, t=round(temperature, 2), m=max_tokens)
        if use_cache and self.cache.enabled:
            hit = self.cache.get(cache_key)
            if hit:
                text, meta = hit
                return LLMResult(text, meta.get("provider", "cache"), meta.get("label", "cache"),
                                 meta.get("model", ""), 0.0, cached=True)

        attempts: list[tuple[str, str]] = []
        waited = 0.0
        while True:
            for ep in self._ordered():
                if ep.label in self._disabled:
                    continue
                if self._cooldown_until.get(ep.label, 0.0) - self._clock() > 0:
                    attempts.append((ep.label, "cooldown"))
                    continue
                started = time.perf_counter()
                try:
                    text = providers.complete(ep, system, prompt, json_mode, temperature, max_tokens)
                except ProviderError as err:
                    attempts.append((ep.label, err.kind))
                    self._penalize(ep, err)
                    continue
                if not text.strip():
                    attempts.append((ep.label, "empty"))
                    continue
                result = LLMResult(text, ep.provider, ep.label, ep.model,
                                   time.perf_counter() - started, failed_attempts=attempts)
                if use_cache and self.cache.enabled:
                    self.cache.set(cache_key, text,
                                   {"provider": ep.provider, "label": ep.label, "model": ep.model})
                self._advance(ep)
                log.info("LLM answered via %s in %.2fs", ep.label, result.latency_s)
                return result
            # Nothing answered. If the first endpoint to recover is only seconds away, wait for it.
            recover_in = self._soonest_recovery()
            if recover_in is None or waited + recover_in > self.max_wait:
                raise LLMUnavailable(attempts, retry_in=recover_in)
            pause = recover_in + 0.25
            log.info("All LLM endpoints busy; waiting %.1fs for the first to recover", pause)
            self._sleep(pause)
            waited += pause

    def _soonest_recovery(self) -> float | None:
        now = self._clock()
        waits = [until - now for label, until in self._cooldown_until.items()
                 if label not in self._disabled and until - now > 0]
        return min(waits) if waits else None

    # -------------------------------------------------------- JSON mode ---
    def call_llm_json(self, prompt: str, schema: Type[T] | None = None, *, system: str | None = None,
                      retries: int = 2, temperature: float = 0.2, max_tokens: int = 2048,
                      use_cache: bool | None = None, include_schema: bool = True):
        """Returns a validated `schema` instance (or a plain dict/list if schema is None).

        Invalid JSON or a failed validation is sent back to the model with the
        error, up to `retries` times. Only VALIDATED answers are cached.
        """
        full_prompt = prompt
        if schema is not None and include_schema:
            full_prompt += ("\n\nReturn ONLY a JSON object that matches this JSON Schema:\n"
                            + json.dumps(schema.model_json_schema()))
        if use_cache is None:
            use_cache = temperature <= 0.3
        cache_key = make_key("json", system, full_prompt, schema=getattr(schema, "__name__", None),
                             t=round(temperature, 2), m=max_tokens)
        if use_cache and self.cache.enabled:
            hit = self.cache.get(cache_key)
            if hit:
                try:
                    data = json.loads(hit[0])
                    return schema.model_validate(data) if schema else data
                except (ValueError, ValidationError):
                    self.cache.delete(cache_key)

        current = full_prompt
        last_error: Exception | None = None
        tokens = max_tokens
        for _ in range(retries + 1):
            result = self.call_llm(current, system=system, json_mode=True, temperature=temperature,
                                   max_tokens=tokens, use_cache=False)
            try:
                data = extract_json(result.text)
                out = schema.model_validate(data) if schema else data
            except (ValueError, ValidationError) as err:
                last_error = err
                log.warning("Invalid JSON from %s: %s", result.label, str(err)[:200])
                if looks_truncated(result.text) and tokens < MAX_RETRY_TOKENS:
                    tokens = min(tokens * 2, MAX_RETRY_TOKENS)     # the reply was cut off: allow a longer one
                current = (f"{full_prompt}\n\nYour previous reply was rejected: {str(err)[:400]}\n"
                           f"Previous reply:\n{result.text[:1500]}\n"
                           "Reply again with ONLY the corrected JSON.")
                continue
            if use_cache and self.cache.enabled:
                self.cache.set(cache_key, json.dumps(data, ensure_ascii=False),
                               {"provider": result.provider, "label": result.label, "model": result.model})
            return out
        raise LLMInvalidOutput(f"Model output failed validation after {retries + 1} tries: {last_error}")


# --------------------------------------------------- module-level shortcuts
_default: LLMClient | None = None
_default_lock = threading.Lock()


def get_client() -> LLMClient:
    global _default
    with _default_lock:
        if _default is None:
            _default = LLMClient.from_env()
        return _default


def reset_client() -> None:
    """Forget the shared client (tests, or after editing .env)."""
    global _default
    with _default_lock:
        _default = None


def call_llm(prompt: str, **kwargs) -> LLMResult:
    return get_client().call_llm(prompt, **kwargs)


def call_llm_json(prompt: str, schema: Type[T] | None = None, **kwargs):
    return get_client().call_llm_json(prompt, schema, **kwargs)

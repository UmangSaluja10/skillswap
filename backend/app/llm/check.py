"""Health check. Run this on YOUR laptop before the event:

    python -m app.llm.check            # ping every configured key/model
    python -m app.llm.check --models   # list the model names each key can use

Reads keys from backend/.env or the environment. Never prints a full key.
"""
from __future__ import annotations

import argparse
import sys
import time

from . import providers
from .client import LLMClient
from .config import mask
from .providers import ProviderError

HINTS = {
    "auth": "key rejected - check it was copied fully, and that the Gemini key's project has the API enabled",
    "config": "model not found - run with --models and copy a valid name into GROQ_MODELS / GEMINI_MODELS",
    "rate_limit": "quota hit right now - this one will be skipped and the next endpoint used",
    "server": "no answer (network, timeout or provider outage)",
    "bad_request": "provider refused the request",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SkillSwap LLM health check")
    parser.add_argument("--models", action="store_true", help="list models available to each key")
    args = parser.parse_args(argv)

    client = LLMClient.from_env()
    if not client.endpoints:
        print("No endpoints configured. Copy .env.example to .env and add at least one key.")
        return 1

    if args.models:
        shown: set[tuple[str, str]] = set()
        for ep in client.endpoints:
            ident = (ep.provider, ep.api_key or ep.host or "")
            if ident in shown:
                continue
            shown.add(ident)
            print(f"\n[{ep.provider} {mask(ep.api_key)}]")
            try:
                for name in providers.list_models(ep):
                    print("  ", name)
            except ProviderError as err:
                print(f"   FAIL ({err.kind}): {err}")
        return 0

    ok = 0
    per_provider: dict[str, list[int]] = {}
    print(f"Checking {len(client.endpoints)} endpoint(s)...\n")
    for ep in client.endpoints:
        started = time.perf_counter()
        try:
            text = providers.complete(ep, None, "Reply with exactly one word: pong", False, 0.0, 16)
        except ProviderError as err:
            per_provider.setdefault(ep.provider, [0, 0])[1] += 1
            print(f"FAIL  {ep.label:<45} {err.kind}: {HINTS.get(err.kind, '')}")
            print(f"      detail: {str(err)[:160]}")
            continue
        ok += 1
        per_provider.setdefault(ep.provider, [0, 0])[0] += 1
        per_provider[ep.provider][1] += 1
        print(f"OK    {ep.label:<45} {time.perf_counter() - started:5.2f}s  -> {text.strip()[:30]!r}")
    print(f"\n{ok}/{len(client.endpoints)} endpoints working. " + ", ".join(
        f"{name}: {good}/{total}" for name, (good, total) in per_provider.items()))
    groq_keys = [e for e in client.endpoints if e.provider == "groq"]
    if len({e.api_key for e in groq_keys}) > 1:
        print("Note: Groq limits apply per ACCOUNT (organization), not per key. Several keys only add "
              "capacity when they belong to different accounts, e.g. one per teammate.")
    if ok == 0:
        print("Nothing usable. Fix the FAIL lines above, or start Ollama as the offline backup.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

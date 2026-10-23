"""Try the resume parser on real files (run on your laptop, with keys in .env):

    python -m app.resume.cli samples/*.pdf          (wildcards work on Windows too)
    python -m app.resume.cli my_resume.pdf --json
    python -m app.resume.cli --text pasted_resume.txt
    python -m app.resume.cli my_resume.pdf --extract-only   (just show the text read from the PDF; no AI)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from .parser import parse_resume_pdf, parse_resume_text
from .pdf import ResumeError, extract_text


def expand_paths(args: list[str]) -> list[str]:
    """PowerShell/cmd don't expand wildcards like bash does, so do it here."""
    out: list[str] = []
    for arg in args:
        if any(ch in arg for ch in "*?["):
            matches = sorted(glob.glob(arg))
            out.extend(matches or [arg])
        else:
            out.append(arg)
    return out


def _show(label: str, profile, as_json: bool) -> None:
    print(f"\n=== {label} ===")
    if as_json:
        print(profile.model_dump_json(indent=2))
        return
    print(f"{profile.name or '(no name)'} - {profile.headline}")
    print("Domains:", ", ".join(profile.domains) or "-")
    for s in profile.skills:
        print(f"  {s.name:<18} {s.level:<13} {s.projects} project(s)   \"{s.evidence}\"")
    for p in profile.projects:
        print(f"  project: {p.title} [{', '.join(p.tech)}]")
    for w in profile.warnings:
        print("  warning:", w)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SkillSwap resume parser")
    ap.add_argument("pdfs", nargs="*", help="PDF files (wildcards allowed)")
    ap.add_argument("--text", help="a .txt file with pasted resume text")
    ap.add_argument("--json", action="store_true", help="print the full JSON profile")
    ap.add_argument("--extract-only", action="store_true", help="only show the text read from each PDF")
    args = ap.parse_args(argv)
    paths = expand_paths(args.pdfs)
    if not paths and not args.text:
        ap.error("give at least one PDF, or --text FILE")

    failures = 0
    if args.extract_only:
        for path in paths:
            print(f"\n=== {path} ===")
            try:
                print(extract_text(path))
            except (ResumeError, OSError) as exc:
                failures += 1
                print(f"FAILED: {exc}")
        return 1 if failures else 0

    from app.llm import LLMInvalidOutput, LLMUnavailable
    jobs = [(p, lambda p=p: parse_resume_pdf(p)) for p in paths]
    if args.text:
        jobs.append((args.text, lambda: parse_resume_text(Path(args.text).read_text(encoding="utf-8"))))
    for label, job in jobs:
        try:
            _show(label, job(), args.json)
        except (ResumeError, LLMUnavailable, LLMInvalidOutput, OSError) as exc:
            failures += 1
            print(f"\n=== {label} ===\nFAILED: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

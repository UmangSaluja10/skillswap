"""Try the planner on the fictional team.

    python -m app.planner.demo            # canned plan: no keys needed, shows assign/schedule/validate
    python -m app.planner.demo --live     # asks your real LLM (Groq/Gemini) to plan
    python -m app.planner.demo --review   # you are the leader: review the draft, ask for changes, confirm
                                          # (combine with --live for the real model)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from ..matching.demo import sample_people
from ..models import Member
from ..reputation.service import new_project
from .graph import apply_plan, plan_project
from .session import PlanSession
from ..llm.client import LLMInvalidOutput, LLMUnavailable

BUSY = (LLMUnavailable, LLMInvalidOutput)
from .schemas import PlanItem, PlanLLMOutput

CANNED = [
    ("Design the app screens in Figma", "UI/UX Design", 2),
    ("Build the resume parsing API", "Backend", 2),
    ("Implement the matching endpoint", "Machine Learning", 2),
    ("Build SwiftUI profile and match screens", "Mobile (iOS)", 3),
    ("Connect the app to the API", "Mobile (iOS)", 2),
    ("Record the promo video", "Content & Marketing", 1),
    ("Seed demo data and test end to end", "Backend", 1),
]


def canned_llm(prompt, schema, system=None):
    """Stands in for the real model. It reacts to two kinds of requests so you can see a revision work."""
    plan = list(CANNED)
    asks = prompt.split("Changes the project leader asked for", 1)[-1].lower() if "Changes the project" in prompt else ""
    asks += prompt.split("Extra requirements from the project leader", 1)[-1].lower() if "Extra requirements" in prompt else ""
    if "video" in asks or "promo" in asks:
        plan = [x for x in plan if "promo" not in x[0].lower()]
    if "test" in asks or "security" in asks:
        plan.insert(-1, ("Security and edge-case testing", "Backend", 2))
    return PlanLLMOutput(milestones=[PlanItem(title=t, domain=d, weight=w) for t, d, w in plan])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use the real LLM instead of the canned plan")
    ap.add_argument("--review", action="store_true", help="act as the project leader: review and revise the draft")
    ap.add_argument("--fixed-end", action="store_true",
                    help="the end cannot be extended (like a real event); the default lets the leader extend it")
    ap.add_argument("--hours", type=float, default=24, help="time available (default 24)")
    args = ap.parse_args()

    people = {p.uid: p for p in sample_people()}
    team = [people[u] for u in ("aditi", "meera", "ananya")]
    now = datetime.now(timezone.utc)
    project = new_project("skillswap", "SkillSwap", "aditi", {}, now,
                          "A campus app where students upload a resume, get matched with complementary "
                          "teammates for projects, and track milestones with AI-planned deadlines.")
    for p in team[1:]:
        project.members.append(Member(uid=p.uid))

    names = {p.uid: p.name for p in team}
    end = now + timedelta(hours=args.hours)
    cap = end if args.fixed_end else None

    def show(milestones):
        for m in milestones:
            print(f"  {m.due.astimezone():%a %H:%M}  [{m.domain:22s}] w{m.weight}  {names[m.assignee_uid]:13s} {m.title}")

    if args.review:
        from langgraph.checkpoint.memory import MemorySaver
        session = PlanSession(llm=None if args.live else canned_llm, checkpointer=MemorySaver())
        extra = input("Anything specific to include? (Enter to skip): ")
        while True:
            try:
                draft = session.start(project, team, now=now, end=end, extra=extra, max_end=cap)
                break
            except BUSY as err:
                print(f"\nThe AI service is busy or gave an unreadable answer ({str(err)[:90]}).")
                if input("Press Enter to try again, or type q to quit: ").strip().lower() == "q":
                    return
        while True:
            if draft.status == "failed":
                print("Could not create a plan:", "; ".join(draft.errors)); return
            print(f"\nDRAFT (revisions left: {draft.revisions_left})")
            show(draft.milestones)
            if draft.extended and draft.end:
                print(f"  project deadline extended to {draft.end.astimezone():%a %d %b %H:%M}")
            if draft.notes:
                print("  planner:", draft.notes)
            for w in draft.warnings:
                print("  warning:", w)
            if draft.status == "revision_failed":
                print("  The last change could not be applied:", "; ".join(draft.errors))
            answer = input("\nLooks good? Type y to confirm, or describe what to change: ").strip()
            if answer.lower() in ("y", "yes"):
                result = session.confirm(draft.session_id, project, team, now=now, end=end)
                break
            while True:
                try:
                    draft = session.revise(draft.session_id, project, team, now=now, end=end, feedback=answer, max_end=cap)
                    break
                except BUSY as err:     # your request is kept: trying again continues it
                    print(f"\nThe AI service is busy or gave an unreadable answer ({str(err)[:90]}).")
                    if input("Press Enter to retry your last request, or type q to stop: ").strip().lower() == "q":
                        return
        apply_plan(project, result)
        print(f"\nCONFIRMED: {len(project.milestones)} milestones applied. "
              f"Project deadline: {project.deadline.astimezone():%a %d %b %H:%M}")
        show(project.milestones)
        return

    result = plan_project(project, team, now=now, end=end, llm=None if args.live else canned_llm)
    print(f"ok={result.ok}  LLM calls={result.attempts}\n")
    show(result.milestones)
    for w in result.warnings:
        print("  warning:", w)
    for e in result.errors:
        print("  error:", e)
    if result.gaps:
        print("  recruiting suggestions:", ", ".join(g.role for g in result.gaps))
    if result.ok:
        apply_plan(project, result)
        print(f"\nApplied: {len(project.milestones)} milestones, {len(project.needs)} open role(s) on the project.")


if __name__ == "__main__":
    main()

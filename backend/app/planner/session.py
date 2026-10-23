"""Human-in-the-loop planning: draft -> leader reviews -> confirm, or ask for changes -> new draft.

    session = PlanSession()
    draft = session.start(project, team, now=now, end=end, extra="Include a testing milestone")
    # show draft.milestones to the leader ...
    draft = session.revise(draft.session_id, project, team, now=now, end=end, feedback="Drop the promo video")
    result = session.confirm(draft.session_id, project, team, now=now, end=end)   # optional edits=[...]
    apply_plan(project, result)

Only the project leader may start, revise or confirm (the API layer checks `require_leader`).
The graph's paused state is saved by a checkpointer (SQLite file by default, memory as a fallback).
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from ..matching.engine import Matcher
from ..models import Milestone, Need, Person, Project
from ..reputation.service import RuleError
from .graph import DEFAULT_TZ, MAX_REVISIONS, LLMFn, build_graph, result_from_state
from .schemas import PlanResult


class Draft(BaseModel):
    session_id: str
    status: Literal["review", "failed", "revision_failed"]
    milestones: list[Milestone] = Field(default_factory=list)
    gaps: list[Need] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    revisions: int = 0
    revisions_left: int = MAX_REVISIONS
    notes: str = ""                  # the planner's own note: what it changed, or why it could not
    changed: bool | None = None      # after a revision: did the draft actually change? (None for the first draft)
    end: datetime | None = None      # the project end this draft was planned for
    extended: bool = False           # True when the leader's deadline request pushed the project end later


def default_checkpointer(path: str | None = ".planner_sessions.sqlite"):
    """SQLite file so a draft survives a server restart; falls back to memory if the package is missing."""
    if path:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            return SqliteSaver(sqlite3.connect(path, check_same_thread=False))
        except Exception:
            pass
    return MemorySaver()


def _signature(milestones: list[dict]) -> list[tuple]:
    return [(m["title"], m["domain"], m["assignee_uid"], str(m["due"]), m["weight"]) for m in milestones]


def require_leader(project: Project, uid: str) -> None:
    if uid != project.leader_uid:
        raise RuleError("Only the project leader can plan the project.")
    if project.status != "active":
        raise RuleError("This project is completed, so it can no longer be changed.")


class PlanSession:
    def __init__(self, llm: LLMFn | None = None, matcher: Matcher | None = None, checkpointer=None) -> None:
        self.graph = build_graph(llm, matcher, review=True, checkpointer=checkpointer or default_checkpointer())

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _cfg(session_id: str, project: Project, people: list[Person], now: datetime, end: datetime,
             tz: str = DEFAULT_TZ, max_end: datetime | None = None) -> dict:
        return {"configurable": {"thread_id": session_id, "project": project, "people": people,
                                 "now": now, "end": end, "tz": tz, "max_end": max_end}}

    def _snapshot(self, cfg: dict):
        snap = self.graph.get_state(cfg)
        if not snap.values:
            raise RuleError("This planning session was not found. Start a new plan.")
        return snap

    def _to_draft(self, session_id: str, project: Project, cfg: dict) -> Draft:
        snap = self._snapshot(cfg)
        values = snap.values
        original_end = cfg["configurable"]["end"]
        revisions = values.get("revisions", 0)
        left = max(0, MAX_REVISIONS - revisions)
        if snap.next:                                    # paused at "review": a draft is waiting
            r = result_from_state(values, project, original_end)
            warnings, changed = list(r.warnings), None
            if revisions and values.get("good"):
                changed = _signature(values["milestones"]) != _signature(values["good"]["milestones"])
                if not changed:
                    warnings.insert(0, "The planner made no changes to the draft."
                                    + ("" if values.get("notes") else " Try describing the change differently, "
                                       "or edit owners and deadlines directly."))
            return Draft(session_id=session_id, status="review", milestones=r.milestones, gaps=r.gaps,
                         warnings=warnings, revisions=revisions, revisions_left=left,
                         notes=values.get("notes", ""), changed=changed, end=r.end, extended=r.extended)
        r = result_from_state(values, project, original_end)   # the graph ended without a draft: validation failed
        good = values.get("good")
        if good and revisions:                           # a revision failed; the previous draft is still usable
            prev = result_from_state({**good, "attempt": 0}, project, original_end)
            return Draft(session_id=session_id, status="revision_failed", milestones=prev.milestones, gaps=prev.gaps,
                         warnings=prev.warnings, errors=r.errors or ["The revised plan could not be validated."],
                         revisions=revisions, revisions_left=left, notes=values.get("notes", ""), changed=False,
                         end=prev.end, extended=prev.extended)
        return Draft(session_id=session_id, status="failed", errors=r.errors or ["No plan could be created."],
                     revisions=revisions, revisions_left=left)

    # ------------------------------------------------------------------ public steps
    def start(self, project: Project, people: list[Person], *, now: datetime, end: datetime,
              brief: str | None = None, extra: str = "", tz: str = DEFAULT_TZ,
              max_end: datetime | None = None) -> Draft:
        if not people:
            raise RuleError("The project has no team members to plan for.")
        session_id = uuid.uuid4().hex[:12]
        cfg = self._cfg(session_id, project, people, now, end, tz, max_end)
        self.graph.invoke(
            {"brief": (brief or project.description or project.title).strip(),
             "requirements": [extra.strip()[:600]] if extra.strip() else [], "feedback": [], "attempt": 0,
             "revisions": 0, "warnings": [], "gaps": []}, cfg)
        return self._to_draft(session_id, project, cfg)

    def revise(self, session_id: str, project: Project, people: list[Person], *, now: datetime, end: datetime,
               feedback: str, tz: str = DEFAULT_TZ, max_end: datetime | None = None) -> Draft:
        if not feedback.strip():
            raise RuleError("Tell the planner what to change, or confirm the plan as it is.")
        cfg = self._cfg(session_id, project, people, now, end, tz, max_end)
        snap = self._snapshot(cfg)
        if snap.values.get("revisions", 0) >= MAX_REVISIONS:
            raise RuleError(f"You have used all {MAX_REVISIONS} revisions. Confirm the plan or edit tasks directly.")
        if snap.next and snap.next[0] == "review":
            self.graph.invoke(Command(resume={"action": "revise", "feedback": feedback}), cfg)
        elif snap.next:                                  # an earlier call failed half way (e.g. LLM down): continue it
            self.graph.invoke(None, cfg)
        else:                                            # the last revision failed validation: start a new one
            raise RuleError("The last revision could not be completed. Confirm the previous draft or start again.")
        return self._to_draft(session_id, project, cfg)

    def confirm(self, session_id: str, project: Project, people: list[Person], *, now: datetime, end: datetime,
                edits: list[dict] | None = None) -> PlanResult:
        """Accepts the draft. `edits` = the leader's own changes to owners/deadlines (checked here)."""
        cfg = self._cfg(session_id, project, people, now, end)
        snap = self._snapshot(cfg)
        values = dict(snap.values)
        if snap.next and snap.next[0] == "review":
            self.graph.invoke(Command(resume={"action": "confirm"}), cfg)
        elif values.get("good") and values.get("revisions"):    # the last revision failed: use the last good draft
            values = {**values, **values["good"], "plan_errors": [], "fatal_errors": []}
        elif snap.next:
            raise RuleError("The plan is not ready to confirm yet.")
        result = result_from_state(values, project, end)
        if not result.ok:
            raise RuleError("This plan has errors, so it cannot be confirmed.")
        if edits:
            result.milestones = apply_edits(result.milestones, edits, people, now, result.end or end)
        return result

    def current_draft(self, session_id: str, project: Project, people: list[Person], *, now: datetime,
                      end: datetime) -> Draft:
        return self._to_draft(session_id, project, self._cfg(session_id, project, people, now, end))


def apply_edits(milestones: list[Milestone], edits: list[dict], people: list[Person],
                now: datetime, end: datetime) -> list[Milestone]:
    """The leader's direct changes: each edit is {"id", "assignee_uid"?, "due"?, "title"?}. Nothing else changes."""
    from pydantic import TypeAdapter
    from ..timeutil import UTCDateTime
    members = {p.uid for p in people}
    by_id = {m.id: m for m in milestones}
    out = {m.id: m.model_copy() for m in milestones}
    for e in edits:
        mid = e.get("id")
        if mid not in by_id:
            raise RuleError(f"Unknown milestone '{mid}'.")
        m = out[mid]
        if "assignee_uid" in e:
            if e["assignee_uid"] not in members:
                raise RuleError("A task can only be given to a member of the project.")
            m.assignee_uid = e["assignee_uid"]
        if "due" in e:
            due = TypeAdapter(UTCDateTime).validate_python(e["due"])
            if not (now < due <= end):
                raise RuleError(f"The deadline for '{m.title}' must be between now and the end of the project.")
            m.due = due
        if "title" in e:
            title = " ".join(str(e["title"]).split())
            if len(title) < 3 or len(title) > 90:
                raise RuleError("A title must be between 3 and 90 characters.")
            m.title = title
    return [out[m.id] for m in milestones]

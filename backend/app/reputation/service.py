"""The rules: who may mark what, leader accountability, and the temporary-leader tag.

Pure functions on `Project` and a `{uid: UserRep}` dict. They change those objects in place and
raise `RuleError` (with a message that is safe to show the student) when a rule is broken. The API
layer loads from Firebase, calls these, and saves what changed.

Rules in one place:
  * The project leader marks teammates' tasks. Nobody marks their own task.
  * A leader's own overdue task is marked as missed only when ALL teammates report it.
  * Marking "missed" needs the deadline to have passed. Excused / handed-over never cost anything.
  * The owner is the leader at the start and can transfer the tag to a member. The tag is removed
    when the project completes, but every 'Led: <project>' stays in the profile history.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from ..models import Member, Milestone, Project
from .models import Event, LeadEntry, UserRep

MARK_OUTCOMES = ("done", "missed", "abandoned", "excused")


class RuleError(ValueError):
    pass


def _rep(users: dict[str, UserRep], uid: str) -> UserRep:
    return users.setdefault(uid, UserRep(uid=uid))


def _milestone(project: Project, milestone_id: str) -> Milestone:
    for m in project.milestones:
        if m.id == milestone_id:
            return m
    raise RuleError("That milestone does not exist in this project.")


def _require_active(project: Project) -> None:
    if project.status != "active":
        raise RuleError("This project is completed, so it can no longer be changed.")


def _add_event(users: dict[str, UserRep], m: Milestone, project: Project, uid: str, kind: str,
               at: datetime, marked_by: str, note: str = "") -> Event:
    event = Event(id=uuid.uuid4().hex[:12], uid=uid, domain=m.domain, project_id=project.id,
                  milestone_id=m.id, kind=kind, weight=m.weight, at=at, marked_by=marked_by, note=note)
    _rep(users, uid).events.append(event)
    return event


# ---------------------------------------------------------------- leadership tag

def new_project(project_id: str, title: str, owner_uid: str, users: dict[str, UserRep], now: datetime,
                description: str = "") -> Project:
    """Creates a project; the owner is its first leader (tag on, history entry opened)."""
    project = Project(id=project_id, title=title, description=description, owner_uid=owner_uid,
                      leader_uid=owner_uid, members=[Member(uid=owner_uid, role="Owner")], created_at=now)
    _rep(users, owner_uid).lead_history.append(
        LeadEntry(project_id=project_id, project_title=title, since=now))
    return project


def _close_lead(users: dict[str, UserRep], project: Project, uid: str, now: datetime, reason: str) -> None:
    for entry in _rep(users, uid).lead_history:
        if entry.project_id == project.id and entry.until is None:
            entry.until = now
            entry.ended_reason = reason  # type: ignore[assignment]


def transfer_leadership(project: Project, users: dict[str, UserRep], new_leader_uid: str,
                        by_uid: str, now: datetime) -> None:
    _require_active(project)
    if by_uid != project.leader_uid:
        raise RuleError("Only the current leader can hand over leadership.")
    if new_leader_uid == project.leader_uid:
        raise RuleError("That person is already the leader.")
    if new_leader_uid not in project.member_uids():
        raise RuleError("Leadership can only be given to a member of the project.")
    _close_lead(users, project, project.leader_uid, now, "transferred")
    project.leader_uid = new_leader_uid
    _rep(users, new_leader_uid).lead_history.append(
        LeadEntry(project_id=project.id, project_title=project.title, since=now))


def complete_project(project: Project, users: dict[str, UserRep], by_uid: str, now: datetime) -> None:
    """Leader tag goes away; the 'Led: <project>' history entry stays for good."""
    _require_active(project)
    if by_uid != project.leader_uid:
        raise RuleError("Only the project leader can mark the project as completed.")
    _close_lead(users, project, project.leader_uid, now, "completed")
    project.status = "completed"
    project.completed_at = now


def active_leader_tags(rep: UserRep | None) -> list[str]:
    """Projects this student leads right now (shown as a tag on their profile)."""
    return [e.project_title for e in (rep.lead_history if rep else []) if e.until is None]


def lead_history(rep: UserRep | None) -> list[str]:
    """Permanent history lines, newest first: 'Led: Campus Marketplace (12 Oct 2026 - 24 Oct 2026, completed)'."""
    lines = []
    for e in sorted(rep.lead_history if rep else [], key=lambda x: x.since, reverse=True):
        start = e.since.strftime("%d %b %Y")
        if e.until is None:
            span = f"since {start}, ongoing"
        else:
            span = f"{start} - {e.until.strftime('%d %b %Y')}, {e.ended_reason}"
        lines.append(f"Led: {e.project_title} ({span})")
    return lines


# ---------------------------------------------------------------- marking tasks

def _check_marker(project: Project, m: Milestone, by_uid: str) -> None:
    if by_uid not in project.member_uids():
        raise RuleError("Only members of the project can do this.")
    if by_uid == m.assignee_uid:
        raise RuleError("Nobody can mark their own task. Another teammate or the leader does it.")
    if m.assignee_uid == project.leader_uid:
        return  # the leader's own task: any other member may approve it
    if by_uid != project.leader_uid:
        raise RuleError("Only the project leader can mark a teammate's task.")


def mark_milestone(project: Project, users: dict[str, UserRep], milestone_id: str, outcome: str,
                   by_uid: str, now: datetime, note: str = "", completed_at: datetime | None = None) -> Event:
    """outcome: done | missed | abandoned | excused."""
    _require_active(project)
    if outcome not in MARK_OUTCOMES:
        raise RuleError(f"Outcome must be one of: {', '.join(MARK_OUTCOMES)}.")
    m = _milestone(project, milestone_id)
    if m.status != "pending":
        raise RuleError("This task has already been marked.")
    _check_marker(project, m, by_uid)

    if outcome == "done":
        finished = completed_at or now
        kind = "on_time" if finished <= m.due else "late"
        m.status, m.completed_at = "done", finished
    else:
        if outcome == "missed" and now <= m.due:
            raise RuleError("The deadline has not passed yet, so this task cannot be marked as missed.")
        kind = outcome
        m.status = outcome  # type: ignore[assignment]
    return _add_event(users, m, project, m.assignee_uid, kind, now, by_uid, note)


def hand_over(project: Project, users: dict[str, UserRep], milestone_id: str, to_uid: str,
              by_uid: str, now: datetime, note: str = "") -> Event:
    """A proper handover: no penalty, and the task continues with someone else."""
    _require_active(project)
    m = _milestone(project, milestone_id)
    if m.status != "pending":
        raise RuleError("This task has already been marked.")
    if by_uid not in (project.leader_uid, m.assignee_uid):
        raise RuleError("Only the assignee or the project leader can hand a task over.")
    if to_uid not in project.member_uids():
        raise RuleError("A task can only be handed to a member of the project.")
    if to_uid == m.assignee_uid:
        raise RuleError("That person already has this task.")
    event = _add_event(users, m, project, m.assignee_uid, "handed_over", now, by_uid, note)
    m.handover_from.append(m.assignee_uid)
    m.assignee_uid = to_uid
    m.leader_reports = []
    return event


# ---------------------------------------------------------------- leader accountability

def report_leader_missed(project: Project, users: dict[str, UserRep], milestone_id: str,
                         reporter_uid: str, now: datetime) -> dict:
    """A teammate reports that the leader's own task is overdue.

    The task is marked missed only when EVERY teammate has reported it (so one friend of the
    leader, or one emergency the others know about, is enough to stop it).
    """
    _require_active(project)
    m = _milestone(project, milestone_id)
    if m.assignee_uid != project.leader_uid:
        raise RuleError("Teammates report only the leader's own tasks. The leader marks everyone else's.")
    if m.status != "pending":
        raise RuleError("This task has already been marked.")
    if now <= m.due:
        raise RuleError("The deadline has not passed yet.")
    if reporter_uid == project.leader_uid or reporter_uid not in project.member_uids():
        raise RuleError("Only teammates of the leader can report this.")
    if reporter_uid not in m.leader_reports:
        m.leader_reports.append(reporter_uid)

    teammates = project.member_uids() - {project.leader_uid}
    applied = teammates.issubset(set(m.leader_reports))
    if applied:
        m.status = "missed"
        _add_event(users, m, project, m.assignee_uid, "missed", now, "team",
                   "Reported by all teammates")
    return {"applied": applied, "reports": len(set(m.leader_reports) & teammates), "needed": len(teammates)}

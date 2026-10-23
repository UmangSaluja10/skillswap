"""What we store per student for reputation: a list of events and a leadership history.

Everything shown on a profile (points, reliability, level) is calculated from these events,
so a wrong mark can be corrected and the numbers simply recompute.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..timeutil import UTCDateTime

# on_time / late            -> task finished and approved
# missed / abandoned        -> counted against reliability (abandoned = left with no handover)
# handed_over / excused     -> recorded, but never penalised (proper handover, real emergency)
EventKind = Literal["on_time", "late", "missed", "abandoned", "handed_over", "excused"]


class Event(BaseModel):
    id: str
    uid: str
    domain: str
    project_id: str
    milestone_id: str
    kind: EventKind
    weight: int = Field(default=1, ge=1, le=3)
    at: UTCDateTime
    marked_by: str  # a user id, or "team" when every teammate reported the leader
    note: str = ""


class LeadEntry(BaseModel):
    """One line of 'Led: <project>' on a profile. until=None means the tag is still active."""

    project_id: str
    project_title: str
    since: UTCDateTime
    until: UTCDateTime | None = None
    ended_reason: Literal["transferred", "completed"] | None = None


class UserRep(BaseModel):
    uid: str
    events: list[Event] = Field(default_factory=list)
    lead_history: list[LeadEntry] = Field(default_factory=list)


class DomainStanding(BaseModel):
    domain: str
    points: int
    counted: int  # tasks that count towards reliability
    reliability: float | None  # 0..1, None until enough tasks have been reviewed
    level: Literal["Newcomer", "Contributor", "Trusted"]
    label: str  # "New", "Building reliability", "Reliable", "Highly reliable"

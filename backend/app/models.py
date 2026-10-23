"""Shared data shapes: people, projects, open needs and milestones.

Module 4 (the AI planner) produces `Milestone`s; the API layer saves these straight to
Firebase with `.model_dump(mode="json")` and loads them back with `Model(**data)`.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator

from .avatars import avatar_file, normalize_gender
from .reputation.models import UserRep
from .resume.schemas import Profile
from .timeutil import UTCDateTime


class Person(BaseModel):
    uid: str
    name: str
    profile: Profile
    rep: UserRep | None = None
    available: bool = True  # switched off while someone is fully booked
    gender: str = "unspecified"  # optional; ONLY used to pick the avatar, never for matching

    @field_validator("gender", mode="before")
    @classmethod
    def _gender(cls, v):
        return normalize_gender(v)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def avatar(self) -> str:
        """File name of the automatically chosen avatar, e.g. 'avatar_female.png'."""
        return avatar_file(self.gender)


class Need(BaseModel):
    """A role a project is still looking for."""

    role: str = Field(min_length=1, max_length=80)
    domain: str | None = None  # one of resume.taxonomy.DOMAINS
    skills: list[str] = Field(default_factory=list)
    description: str = Field(default="", max_length=400)
    filled_by: str | None = None


class Member(BaseModel):
    uid: str
    role: str = ""


MilestoneStatus = Literal["pending", "done", "missed", "abandoned", "excused"]


class Milestone(BaseModel):
    id: str
    title: str
    domain: str
    assignee_uid: str
    due: UTCDateTime
    weight: int = Field(default=1, ge=1, le=3)  # 1 small, 3 big (set by the planner)
    status: MilestoneStatus = "pending"
    completed_at: UTCDateTime | None = None
    proof_url: str = ""
    leader_reports: list[str] = Field(default_factory=list)  # teammates who reported a leader's overdue task
    handover_from: list[str] = Field(default_factory=list)


class Project(BaseModel):
    id: str
    title: str
    description: str = ""
    owner_uid: str
    leader_uid: str  # starts as the owner; can be transferred
    status: Literal["active", "completed"] = "active"
    members: list[Member] = Field(default_factory=list)
    needs: list[Need] = Field(default_factory=list)
    milestones: list[Milestone] = Field(default_factory=list)
    created_at: UTCDateTime
    deadline: UTCDateTime | None = None      # the planned end; the leader can extend it while planning
    completed_at: UTCDateTime | None = None

    def member_uids(self) -> set[str]:
        return {m.uid for m in self.members}

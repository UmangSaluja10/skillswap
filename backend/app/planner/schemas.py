"""Shapes for the planner: what the LLM must return, and what the planner returns."""
from __future__ import annotations

from datetime import datetime

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from ..models import Milestone, Need


class PlanItem(BaseModel):
    """One milestone as the LLM proposes it (no owner, no date yet: code adds those)."""

    title: str = Field(min_length=3, max_length=90)
    description: str = Field(default="", max_length=240)
    domain: str = Field(min_length=1, max_length=60)
    weight: int = Field(default=1, ge=1, le=3)
    # Only set when the project leader explicitly asked to change the owner or the deadline of this milestone.
    assignee_name: str | None = Field(default=None, max_length=80)
    due_at: str | None = Field(default=None, max_length=40)     # ISO 8601 with offset, e.g. 2026-10-24T07:00:00+05:30

    @field_validator("assignee_name", "due_at", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        v = None if v is None else " ".join(str(v).split())
        return v or None

    @field_validator("title", "description", "domain", mode="before")
    @classmethod
    def _strip(cls, v):
        return " ".join(str(v).split()) if v is not None else ""

    @field_validator("weight", mode="before")
    @classmethod
    def _weight(cls, v):
        words = {"small": 1, "low": 1, "medium": 2, "large": 3, "big": 3, "high": 3}
        if isinstance(v, str) and v.strip().lower() in words:
            return words[v.strip().lower()]
        try:
            return max(1, min(3, int(float(v))))
        except (TypeError, ValueError):
            return 1


class PlanLLMOutput(BaseModel):
    milestones: list[PlanItem] = Field(default_factory=list)
    notes: str = Field(default="", max_length=400)   # what changed, or why a request could not be done

    @field_validator("notes", mode="before")
    @classmethod
    def _notes(cls, v):
        return "" if v is None else " ".join(str(v).split())[:400]


class PlanResult(BaseModel):
    ok: bool
    milestones: list[Milestone] = Field(default_factory=list)
    gaps: list[Need] = Field(default_factory=list)       # domains the plan needs but nobody on the team covers
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)      # why it failed, when ok is False
    attempts: int = 0
    notes: str = ""
    end: datetime | None = None          # the project end this plan was made for (may be extended)
    extended: bool = False

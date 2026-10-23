"""Two shapes: what we ask the LLM for (loose), and the cleaned Profile we store (strict)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Level = Literal["beginner", "intermediate", "advanced"]
LEVEL_RANK = {"beginner": 0, "intermediate": 1, "advanced": 2}
_LEVEL_SYNONYMS = {
    "novice": "beginner", "basic": "beginner", "entry": "beginner", "learning": "beginner",
    "intermediate": "intermediate", "medium": "intermediate", "proficient": "intermediate",
    "advanced": "advanced", "expert": "advanced", "senior": "advanced", "strong": "advanced",
}


class SkillOut(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    level: Level
    projects: int = Field(default=0, ge=0, le=50)
    evidence: str = Field(default="", max_length=300)

    @field_validator("level", mode="before")
    @classmethod
    def _level(cls, v):
        key = str(v).strip().lower()
        return _LEVEL_SYNONYMS.get(key, key)

    @field_validator("projects", mode="before")
    @classmethod
    def _projects(cls, v):
        try:
            return max(0, min(50, int(float(v))))
        except (TypeError, ValueError):
            return 0

    @field_validator("evidence", mode="before")
    @classmethod
    def _evidence(cls, v):
        return "" if v is None else str(v)[:300]


class ProjectOut(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    tech: list[str] = Field(default_factory=list)
    summary: str = Field(default="", max_length=400)

    @field_validator("summary", mode="before")
    @classmethod
    def _summary(cls, v):
        return "" if v is None else str(v)[:400]


class ResumeLLMOutput(BaseModel):
    """What the model must return."""

    name: str | None = None
    headline: str = Field(default="", max_length=200)
    education: str | None = None
    domains: list[str] = Field(default_factory=list)
    skills: list[SkillOut] = Field(default_factory=list)
    projects: list[ProjectOut] = Field(default_factory=list)


class Profile(BaseModel):
    """Cleaned result. This is what gets saved on the user and embedded in module 3."""

    name: str | None = None
    headline: str = ""
    education: str | None = None
    domains: list[str] = Field(default_factory=list)
    skills: list[SkillOut] = Field(default_factory=list)
    projects: list[ProjectOut] = Field(default_factory=list)
    skill_summary: str = ""  # deterministic text built from the fields above, used for embeddings
    warnings: list[str] = Field(default_factory=list)

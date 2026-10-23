"""Times are always timezone-aware UTC, stored as ISO strings (what Firebase keeps)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from pydantic import BeforeValidator


def _aware(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, datetime) and value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


UTCDateTime = Annotated[datetime, BeforeValidator(_aware)]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)

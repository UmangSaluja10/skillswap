"""Turns a student's events into per-domain points, reliability and level.

* points      = earned recognition. Only finished work adds points; they never go down.
* reliability = weighted on-time rate for that domain, tracked separately per domain.
                Recent work counts more, so a student who starts delivering recovers on their own.
* excused / handed_over events are recorded but never counted against anyone.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from .models import DomainStanding, Event, UserRep

HALF_LIFE_DAYS = 60       # an event's influence halves every 60 days
MIN_DECAY = 0.15          # old work never disappears completely
MIN_COUNTED = 3           # fewer reviewed tasks than this -> shown as "New"

OUTCOME_SCORE = {"on_time": 1.0, "late": 0.5, "missed": 0.0, "abandoned": 0.0}
EXTRA_WEIGHT = {"abandoned": 1.5}   # leaving with no handover hurts the team more than a miss
POINTS = {"on_time": 10, "late": 5}

CONTRIBUTOR_POINTS = 20
TRUSTED_POINTS = 80
TRUSTED_MIN_RELIABILITY = 0.75


def _decay(event: Event, now: datetime) -> float:
    age_days = max(0.0, (now - event.at).total_seconds() / 86400)
    return max(MIN_DECAY, 0.5 ** (age_days / HALF_LIFE_DAYS))


def _reliability(events: list[Event], now: datetime) -> tuple[int, float | None]:
    counted = [e for e in events if e.kind in OUTCOME_SCORE]
    if len(counted) < MIN_COUNTED:
        return len(counted), None
    total = got = 0.0
    for e in counted:
        w = e.weight * EXTRA_WEIGHT.get(e.kind, 1.0) * _decay(e, now)
        total += w
        got += w * OUTCOME_SCORE[e.kind]
    return len(counted), (got / total if total else None)


def _level(points: int, reliability: float | None) -> str:
    if points >= TRUSTED_POINTS and reliability is not None and reliability >= TRUSTED_MIN_RELIABILITY:
        return "Trusted"
    if points >= CONTRIBUTOR_POINTS:
        return "Contributor"
    return "Newcomer"


def _label(reliability: float | None) -> str:
    if reliability is None:
        return "New"
    if reliability >= 0.85:
        return "Highly reliable"
    if reliability >= 0.65:
        return "Reliable"
    return "Building reliability"


def _standing(domain: str, events: list[Event], now: datetime) -> DomainStanding:
    points = sum(POINTS.get(e.kind, 0) * e.weight for e in events)
    counted, reliability = _reliability(events, now)
    return DomainStanding(domain=domain, points=points, counted=counted, reliability=reliability,
                          level=_level(points, reliability), label=_label(reliability))


def standings(rep: UserRep | None, now: datetime) -> dict[str, DomainStanding]:
    """One standing per domain the student has any event in."""
    if rep is None:
        return {}
    by_domain: dict[str, list[Event]] = defaultdict(list)
    for e in rep.events:
        by_domain[e.domain].append(e)
    return {d: _standing(d, evs, now) for d, evs in sorted(by_domain.items())}


def overall(rep: UserRep | None, now: datetime) -> DomainStanding:
    return _standing("Overall", list(rep.events) if rep else [], now)


def standing_for(rep: UserRep | None, domain: str | None, now: datetime) -> DomainStanding:
    """A domain's standing, or the overall one when no domain is given. Never None."""
    if domain is None:
        return overall(rep, now)
    return standings(rep, now).get(domain) or _standing(domain, [], now)

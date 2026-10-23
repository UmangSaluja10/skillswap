"""Complementary matching and domain search.

score = fit  -  overlap with the project owner  +  reputation in that domain

  fit        = skill coverage (how much of what the role needs, weighted by level)
             + semantic similarity (embeddings: "SwiftUI" ~ "iOS app")
             + domain match (profile domain, or a skill that belongs to the domain)
  overlap    = share of the candidate's skills the owner already has (we want complementary people)
  reputation = reliability and level in THAT domain only (a great designer is not ranked on backend work)

The "why" line is built from the numbers, not written by an LLM, so it can never invent a fact
and costs no API quota.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime

from ..models import Need, Person, Project
from ..reputation.models import DomainStanding
from ..reputation.scoring import standing_for
from ..resume.schemas import LEVEL_RANK
from ..resume.taxonomy import canonical_skill, normalize_domain, skill_domain
from .embeddings import EmbeddingStore

W_COVERAGE, W_SEMANTIC, W_DOMAIN = 0.45, 0.35, 0.20
OVERLAP_PENALTY = 0.10
REPUTATION_WEIGHT = 0.10
LEVEL_CREDIT = {"beginner": 0.4, "intermediate": 0.75, "advanced": 1.0}
LEVEL_SIGNAL = {"Newcomer": 0.0, "Contributor": 0.5, "Trusted": 1.0}
MIN_SCORE = 0.05
MIN_SEMANTIC_ALONE = 0.55   # with no skill or domain link, similarity alone must be strong to be listed


@dataclass
class Match:
    uid: str
    name: str
    score: float                       # 0..1
    percent: int
    domain: str | None
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    complements: list[str] = field(default_factory=list)   # matched skills the owner does not have
    breakdown: dict[str, float | None] = field(default_factory=dict)
    standing: DomainStanding | None = None
    why: str = ""
    need_role: str = ""
    project_id: str | None = None


def person_text(p: Person) -> str:
    return p.profile.skill_summary or ". ".join(
        [p.profile.headline, "Skills: " + ", ".join(s.name for s in p.profile.skills)])


def need_text(need: Need, extra: str = "") -> str:
    parts = [need.role]
    if need.domain:
        parts.append(f"Domain: {need.domain}")
    if need.skills:
        parts.append("Skills: " + ", ".join(need.skills))
    if need.description:
        parts.append(need.description)
    if extra:
        parts.append(extra)
    return ". ".join(parts)


def _skill_map(p: Person) -> dict[str, str]:
    return {canonical_skill(s.name).lower(): s.level for s in p.profile.skills}


def _domain_skills(p: Person, domain: str):
    return [s for s in p.profile.skills if skill_domain(s.name) == domain]


def _domain_match(p: Person, domain: str | None) -> float | None:
    if domain is None:
        return None
    if domain in p.profile.domains:
        return 1.0
    return 0.6 if _domain_skills(p, domain) else 0.0


def _domain_depth(p: Person, domain: str) -> float:
    """How deep the person is in a domain: two advanced skills there = 1.0."""
    credits = sorted((LEVEL_CREDIT[s.level] for s in _domain_skills(p, domain)), reverse=True)[:3]
    return min(1.0, sum(credits) / 2.0)


class Matcher:
    def __init__(self, store: EmbeddingStore | None = None, now: datetime | None = None) -> None:
        self.store = store or EmbeddingStore(path=os.environ.get("EMBED_CACHE_PATH", ".embed_cache.sqlite"))
        self._now = now

    def _clock(self) -> datetime:
        from ..timeutil import utcnow
        return self._now or utcnow()

    # ------------------------------------------------------------------ scoring one pair
    def score(self, person: Person, need: Need, owner: Person | None = None, extra_text: str = "",
              project_id: str | None = None) -> Match:
        have = _skill_map(person)
        wanted = [canonical_skill(s) for s in need.skills]
        credits = [LEVEL_CREDIT[have[w.lower()]] if w.lower() in have else 0.0 for w in wanted]
        coverage = sum(credits) / len(wanted) if wanted else None
        if coverage is None and need.domain:       # no skills listed: judge by depth in the domain
            coverage = _domain_depth(person, need.domain)
        matched = [w for w, c in zip(wanted, credits) if c > 0]
        missing = [w for w, c in zip(wanted, credits) if c == 0]

        pv, nv = self.store.embed([person_text(person), need_text(need, extra_text)])
        semantic = self.store.calibrated(float(pv @ nv))
        domain = _domain_match(person, need.domain)

        parts = {"coverage": (coverage, W_COVERAGE), "semantic": (semantic, W_SEMANTIC),
                 "domain": (domain, W_DOMAIN)}
        live = {k: v for k, v in parts.items() if v[0] is not None}
        fit = sum(v * w for v, w in live.values()) / sum(w for _, w in live.values())

        overlap = 0.0
        complements = list(matched)
        if owner is not None:
            owned = set(_skill_map(owner))
            top = [canonical_skill(s.name).lower() for s in person.profile.skills[:10]]
            overlap = sum(1 for s in top if s in owned) / len(top) if top else 0.0
            complements = [m for m in matched if m.lower() not in owned]

        standing = standing_for(person.rep, need.domain, self._clock())
        rel_signal = 0.0 if standing.reliability is None else (standing.reliability - 0.5) * 2
        rep = 0.7 * rel_signal + 0.3 * LEVEL_SIGNAL[standing.level]

        final = min(1.0, max(0.0, fit - OVERLAP_PENALTY * overlap + REPUTATION_WEIGHT * rep))
        match = Match(
            uid=person.uid, name=person.name, score=round(final, 4), percent=round(final * 100),
            domain=need.domain, matched_skills=matched, missing_skills=missing, complements=complements,
            breakdown={"coverage": coverage, "semantic": round(semantic, 3), "domain": domain,
                       "overlap": round(overlap, 3), "reputation": round(rep, 3)},
            standing=standing, need_role=need.role, project_id=project_id)
        match.why = explain(match, person, owner, need)
        return match

    # ------------------------------------------------------------------ people for a project role
    def recommend_people(self, project: Project, need: Need, people: list[Person],
                         owner: Person | None = None, k: int = 5) -> list[Match]:
        taken = project.member_uids() | {project.owner_uid}
        pool = [p for p in people if p.uid not in taken and p.available]
        extra = project.title + (". " + project.description if project.description else "")
        matches = [self.score(p, need, owner, extra, project.id) for p in pool]
        return _top(matches, k)

    # ------------------------------------------------------------------ projects for a person
    def recommend_projects(self, person: Person, projects: list[Project], people_by_uid: dict[str, Person],
                           k: int = 5) -> list[Match]:
        """For each open project, the role this person fits best."""
        out: list[Match] = []
        for project in projects:
            if project.status != "active" or person.uid in project.member_uids():
                continue
            owner = people_by_uid.get(project.owner_uid)
            extra = project.title + (". " + project.description if project.description else "")
            best = None
            for need in project.needs:
                if need.filled_by:
                    continue
                m = self.score(person, need, owner, extra, project.id)
                if best is None or m.score > best.score:
                    best = m
            if best is not None:
                best.name = project.title        # for this view the card is the project
                out.append(best)
        return _top(out, k)

    # ------------------------------------------------------------------ "find people" for a hackathon
    def search_people(self, people: list[Person], domains: list[str] | None = None,
                      skills: list[str] | None = None, query: str = "", match: str = "any",
                      exclude: set[str] | None = None, k: int = 10, owner: Person | None = None) -> list[Match]:
        """Someone with a hackathon opportunity searches by domain (and optionally skills / free text).

        match="any": has at least one of the domains; match="all": has every domain.
        """
        wanted = [d for d in (normalize_domain(d) or d for d in (domains or [])) if d]
        needs = [Need(role=query or (f"{d} teammate" if d else "Teammate"), domain=d, skills=skills or [],
                      description=query) for d in (wanted or [None])]
        results: list[Match] = []
        for p in people:
            if not p.available or p.uid in (exclude or set()):
                continue
            scored = [self.score(p, n, owner) for n in needs]
            if wanted:
                hits = [m for m in scored if (m.breakdown["domain"] or 0) > 0]
                if (match == "all" and len(hits) < len(wanted)) or not hits:
                    continue
                best = max(hits, key=lambda m: m.score)
                if match == "all":
                    best = _average(hits, best)
            else:
                best = max(scored, key=lambda m: m.score)
            results.append(best)
        return _top(results, k)


def _average(hits: list[Match], best: Match) -> Match:
    """Several domains required: average the per-domain scores and show every domain's line."""
    mean = sum(m.score for m in hits) / len(hits)
    domains = ", ".join(m.domain for m in hits if m.domain)
    why = " ".join(f"[{m.domain}] {m.why}" for m in hits)   # build before any field is overwritten
    best.score, best.percent, best.domain, best.why = round(mean, 4), round(mean * 100), domains, why
    return best


def _relevant(m: Match) -> bool:
    """Hide people with no real link to the role: no needed skill, no domain match, weak similarity."""
    return bool(m.matched_skills) or (m.breakdown.get("domain") or 0) > 0 \
        or (m.breakdown.get("semantic") or 0) >= MIN_SEMANTIC_ALONE


def _top(matches: list[Match], k: int) -> list[Match]:
    keep = [m for m in matches if m.score >= MIN_SCORE and _relevant(m)]
    return sorted(keep, key=lambda m: (-m.score, m.name.lower()))[:k]


# ---------------------------------------------------------------------- the "why this match" line

def explain(m: Match, person: Person, owner: Person | None, need: Need) -> str:
    bits: list[str] = []
    if m.matched_skills:
        total = len(m.matched_skills) + len(m.missing_skills)
        bits.append(f"covers {len(m.matched_skills)} of {total} needed skills "
                    f"({', '.join(m.matched_skills[:4])})")
    elif need.domain and (m.breakdown.get("domain") or 0) > 0:
        names = [s.name for s in _domain_skills(person, need.domain)][:3]
        bits.append(f"works in {need.domain}" + (f" ({', '.join(names)})" if names else ""))
    pool = [s.name for s in _domain_skills(person, need.domain)] if need.domain and not m.matched_skills else m.matched_skills
    top_levels = [s for s in person.profile.skills if s.name in pool]
    advanced = [s.name for s in top_levels if LEVEL_RANK[s.level] == 2]
    if advanced:
        bits.append(f"advanced in {', '.join(advanced[:2])}")
    st = m.standing
    if st and st.reliability is not None:
        bits.append(f"{round(st.reliability * 100)}% reliable in {st.domain} over {st.counted} tasks")
    elif st and st.points >= 20:
        bits.append(f"{st.level} in {st.domain}")
    if owner is not None and m.complements:
        bits.append(f"brings {', '.join(m.complements[:2])}, which {owner.name.split()[0]} does not list")
    if not bits:
        doms = ", ".join(person.profile.domains[:2])
        bits.append(f"background in {doms}" if doms else "a related background")
    text = "; ".join(bits)
    return re.sub(r"\s+", " ", text[0].upper() + text[1:] + ".")

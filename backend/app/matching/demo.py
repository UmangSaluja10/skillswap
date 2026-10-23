"""Fictional people + a project, so you can see matching work without any API key or resume.

    python -m app.matching.demo
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from ..models import Member, Milestone, Need, Person
from ..reputation.service import mark_milestone, new_project
from ..resume.parser import build_skill_summary
from ..resume.schemas import Profile, ProjectOut, SkillOut
from .embeddings import EmbeddingStore, get_embedder
from .engine import Matcher

LV = {"b": "beginner", "i": "intermediate", "a": "advanced"}


def _person(uid, name, headline, domains, skills, projects=()):
    prof = Profile(name=name, headline=headline, domains=list(domains),
                   skills=[SkillOut(name=n, level=LV[lv], projects=p) for n, lv, p in skills],
                   projects=[ProjectOut(title=t, tech=tech) for t, tech in projects])
    prof.skill_summary = build_skill_summary(prof)
    return Person(uid=uid, name=name, profile=prof)


def sample_people() -> list[Person]:
    return [
        _person("aditi", "Aditi Gupta", "3rd-year CSE, ML and backend", ["Machine Learning", "Backend"],
                [("Python", "a", 4), ("Flask", "a", 3), ("LangGraph", "i", 1), ("NLP", "i", 2)]),
        _person("rohan", "Rohan Mehta", "Backend developer", ["Backend", "DevOps & Cloud"],
                [("Python", "a", 4), ("Flask", "a", 3), ("PostgreSQL", "i", 2), ("Docker", "i", 2), ("REST APIs", "a", 3)]),
        _person("ananya", "Ananya Rao", "UI/UX designer", ["UI/UX Design", "Frontend"],
                [("Figma", "a", 5), ("User Research", "i", 2), ("Prototyping", "a", 3), ("HTML/CSS", "i", 2)]),
        _person("kabir", "Kabir Sethi", "ML engineer", ["Machine Learning", "Data Science & Analytics"],
                [("Python", "a", 4), ("PyTorch", "a", 3), ("scikit-learn", "i", 2), ("Pandas", "i", 2)]),
        _person("meera", "Meera Joshi", "iOS developer", ["Mobile (iOS)"],
                [("Swift", "a", 3), ("SwiftUI", "a", 3), ("Firebase", "i", 2), ("Xcode", "i", 3)]),
        _person("dev", "Dev Malhotra", "Flutter developer", ["Cross-platform Mobile"],
                [("Flutter", "a", 4), ("Dart", "a", 4), ("Firebase", "i", 2)]),
        _person("sara", "Sara Khan", "Frontend developer", ["Frontend"],
                [("React", "i", 3), ("JavaScript", "i", 3), ("Tailwind CSS", "i", 2), ("HTML/CSS", "a", 3)]),
        _person("neel", "Neel Arora", "Content and marketing", ["Content & Marketing"],
                [("Video Editing", "i", 3), ("Copywriting", "i", 2), ("Social Media", "i", 2)]),
    ]


def sample_project(people):
    """Aditi (ML/backend) is building an iOS app and still needs an iOS dev and a designer."""
    now = datetime.now(timezone.utc)
    users: dict = {}
    project = new_project("skillswap", "SkillSwap", "aditi", users, now - timedelta(days=3),
                          "Campus app that matches students for projects")
    project.needs = [
        Need(role="iOS developer", domain="Mobile (iOS)", skills=["Swift", "SwiftUI", "Firebase"],
             description="Build the SwiftUI screens"),
        Need(role="UI/UX designer", domain="UI/UX Design", skills=["Figma", "Prototyping"]),
    ]
    return project, users


def main() -> None:
    ap = argparse.ArgumentParser(description="Show matching on fictional people.")
    ap.add_argument("--embedder", choices=["auto", "hash", "minilm"], default="auto")
    args = ap.parse_args()

    embedder = get_embedder(args.embedder)
    print(f"Embedder: {embedder.name}\n")
    matcher = Matcher(EmbeddingStore(embedder))
    people = sample_people()
    by_uid = {p.uid: p for p in people}
    project, _ = sample_project(people)
    owner = by_uid["aditi"]

    for need in project.needs:
        print(f"== Project '{project.title}' needs: {need.role}")
        for m in matcher.recommend_people(project, need, people, owner, k=3):
            print(f"  {m.percent:3d}%  {m.name:14s} {m.why}")
        print()

    print("== Hackathon search: domain 'Mobile (iOS)' OR 'UI/UX Design'")
    for m in matcher.search_people(people, domains=["Mobile (iOS)", "UI/UX Design"], exclude={"aditi"}, k=4):
        print(f"  {m.percent:3d}%  {m.name:14s} [{m.domain}] {m.why}")
    print("\n== Hackathon search: 'Machine Learning' AND 'Backend'")
    for m in matcher.search_people(people, domains=["Machine Learning", "Backend"], match="all", k=4):
        print(f"  {m.percent:3d}%  {m.name:14s} [{m.domain}] {m.why}")

    print("\n== Projects that fit Meera (iOS developer)")
    for m in matcher.recommend_projects(by_uid["meera"], [project], by_uid, k=3):
        print(f"  {m.percent:3d}%  {m.name:14s} as {m.need_role}: {m.why}")


if __name__ == "__main__":
    main()

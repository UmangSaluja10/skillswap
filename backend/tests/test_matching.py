from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.matching import EmbeddingStore, HashingEmbedder, Matcher, get_embedder
from app.matching import embeddings as emb
from app.matching.demo import _person, sample_people, sample_project
from app.models import Member, Milestone, Need, Person
from app.reputation.service import mark_milestone, new_project
from app.resume.taxonomy import skill_domain

NOW = datetime(2026, 10, 24, 9, 0, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)


@pytest.fixture()
def world():
    people = sample_people()
    by_uid = {p.uid: p for p in people}
    project, users = sample_project(people)
    return Matcher(EmbeddingStore(HashingEmbedder()), now=NOW), people, by_uid, project


# ---------------------------------------------------------------- embeddings

def test_hashing_embedder_is_unit_length_and_deterministic():
    e = HashingEmbedder()
    a, b = e.embed(["SwiftUI iOS app"]), e.embed(["SwiftUI iOS app"])
    assert np.allclose(a, b) and abs(float(np.linalg.norm(a)) - 1.0) < 1e-5
    assert e.embed([""]).shape == (1, 384)


def test_related_texts_are_closer_than_unrelated():
    s = EmbeddingStore(HashingEmbedder())
    assert s.similarity("SwiftUI developer", "iOS app developer") > s.similarity("SwiftUI developer", "video editing copywriting")


def test_store_computes_each_text_once():
    class Counting(HashingEmbedder):
        calls = 0
        def embed(self, texts):
            Counting.calls += len(texts)
            return super().embed(texts)
    s = EmbeddingStore(Counting())
    s.embed(["a b c", "d e f"]); s.embed(["a b c", "d e f", "a b c"])
    assert Counting.calls == 2


def test_store_survives_restart_on_disk(tmp_path):
    path = tmp_path / "e.sqlite"
    EmbeddingStore(HashingEmbedder(), path).embed(["hello world"])
    class Boom(HashingEmbedder):
        def embed(self, texts):
            raise AssertionError("should have been cached")
    assert EmbeddingStore(Boom(), path).embed(["hello world"]).shape == (1, 384)


def test_get_embedder_falls_back_when_minilm_missing(monkeypatch):
    def boom(*a, **k):
        raise ImportError("no sentence_transformers")
    monkeypatch.setattr(emb, "MiniLMEmbedder", boom)
    assert isinstance(get_embedder("auto"), HashingEmbedder)
    with pytest.raises(ImportError):
        get_embedder("minilm")
    assert isinstance(get_embedder("hash"), HashingEmbedder)


def test_calibration_is_clamped():
    s = EmbeddingStore(HashingEmbedder())
    assert s.calibrated(-1) == 0.0 and s.calibrated(1) == 1.0


def test_skill_domain_lookup():
    assert skill_domain("SwiftUI") == "Mobile (iOS)"
    assert skill_domain("TensorFlow/Keras") == "Machine Learning"
    assert skill_domain("Python") is None


# ---------------------------------------------------------------- ranking for a project role

def test_ios_role_ranks_ios_developer_first(world):
    m, people, by_uid, project = world
    res = m.recommend_people(project, project.needs[0], people, by_uid["aditi"])
    assert res[0].uid == "meera" and res[0].percent >= 85
    assert res[0].matched_skills == ["Swift", "SwiftUI", "Firebase"]
    assert "aditi" not in {r.uid for r in res}


def test_designer_role_ranks_designer_first(world):
    m, people, by_uid, project = world
    assert m.recommend_people(project, project.needs[1], people, by_uid["aditi"])[0].uid == "ananya"


def test_scores_are_between_zero_and_one(world):
    m, people, by_uid, project = world
    for need in project.needs:
        for r in m.recommend_people(project, need, people, by_uid["aditi"], k=20):
            assert 0.0 <= r.score <= 1.0 and r.percent == round(r.score * 100)


def test_members_and_unavailable_people_are_skipped(world):
    m, people, by_uid, project = world
    project.members.append(Member(uid="meera"))
    by_uid["ananya"].available = False
    res = m.recommend_people(project, project.needs[0], people, by_uid["aditi"], k=20)
    assert "meera" not in {r.uid for r in res}
    assert "ananya" not in {r.uid for r in m.recommend_people(project, project.needs[1], people, by_uid["aditi"], k=20)}


def test_overlap_with_owner_lowers_score_complement_preferred():
    owner = _person("o", "Owner One", "backend", ["Backend"], [("Python", "a", 3), ("Flask", "a", 3), ("Docker", "i", 1)])
    same = _person("s", "Same Skills", "backend", ["Backend"], [("Python", "a", 3), ("Flask", "a", 3), ("Docker", "i", 1)])
    diff = _person("d", "Different Skills", "backend", ["Backend"], [("Go", "a", 3), ("PostgreSQL", "a", 3), ("Redis", "i", 1)])
    m = Matcher(EmbeddingStore(HashingEmbedder()), now=NOW)
    need = Need(role="Backend developer", domain="Backend")
    a, b = m.score(same, need, owner), m.score(diff, need, owner)
    assert a.breakdown["overlap"] == 1.0 and b.breakdown["overlap"] == 0.0
    assert b.score > a.score


def test_complements_exclude_what_owner_already_has():
    owner = _person("o", "Owner One", "ml", ["Machine Learning"], [("Python", "a", 3)])
    cand = _person("c", "Cand", "ml", ["Machine Learning"], [("Python", "a", 3), ("PyTorch", "a", 2)])
    m = Matcher(EmbeddingStore(HashingEmbedder()), now=NOW)
    r = m.score(cand, Need(role="ML", domain="Machine Learning", skills=["Python", "PyTorch"]), owner)
    assert r.matched_skills == ["Python", "PyTorch"] and r.complements == ["PyTorch"]
    assert "PyTorch" in r.why and "Owner" in r.why


def test_skill_names_match_through_aliases():
    cand = _person("c", "Cand", "fe", ["Frontend"], [("React", "i", 2)])
    m = Matcher(EmbeddingStore(HashingEmbedder()), now=NOW)
    r = m.score(cand, Need(role="Frontend", domain="Frontend", skills=["reactjs"]))
    assert r.matched_skills == ["React"]


def test_levels_matter_advanced_beats_beginner():
    adv = _person("a", "Adv", "x", ["Backend"], [("Flask", "a", 3)])
    beg = _person("b", "Beg", "x", ["Backend"], [("Flask", "b", 0)])
    m = Matcher(EmbeddingStore(HashingEmbedder()), now=NOW)
    need = Need(role="Backend", domain="Backend", skills=["Flask"])
    assert m.score(adv, need).score > m.score(beg, need).score


# ---------------------------------------------------------------- reputation in ranking

def _record(person, domain, outcomes):
    """Gives a person a track record in a domain via the real service rules."""
    users = {}
    p = new_project("px", "Past Project", "boss", users, NOW - 30 * HOUR)
    p.members.append(Member(uid=person.uid))
    for i, outcome in enumerate(outcomes):
        p.milestones.append(Milestone(id=f"m{i}", title="t", domain=domain, assignee_uid=person.uid,
                                      due=NOW - 20 * HOUR))
        mark_milestone(p, users, f"m{i}", outcome, "boss", NOW - 10 * HOUR,
                       completed_at=NOW - 25 * HOUR if outcome == "done" else None)
    person.rep = users[person.uid]


def test_reliable_candidate_outranks_identical_unreliable_one():
    mk = lambda uid: _person(uid, uid.title(), "x", ["Backend"], [("Flask", "i", 2)])
    good, bad, fresh = mk("good"), mk("bad"), mk("fresh")
    _record(good, "Backend", ["done"] * 5)
    _record(bad, "Backend", ["missed"] * 4 + ["done"])
    m = Matcher(EmbeddingStore(HashingEmbedder()), now=NOW)
    need = Need(role="Backend", domain="Backend", skills=["Flask"])
    g, b, f = (m.score(p, need) for p in (good, bad, fresh))
    assert g.score > f.score > b.score
    assert "reliable in Backend over 5 tasks" in g.why


def test_reputation_is_domain_specific():
    mk = lambda uid: _person(uid, uid.title(), "x", ["Backend", "Frontend"], [("Flask", "i", 2), ("React", "i", 2)])
    star_fe, plain = mk("fe"), mk("plain")
    _record(star_fe, "Frontend", ["done"] * 6)
    m = Matcher(EmbeddingStore(HashingEmbedder()), now=NOW)
    backend = Need(role="Backend", domain="Backend", skills=["Flask"])
    frontend = Need(role="Frontend", domain="Frontend", skills=["React"])
    assert m.score(star_fe, backend).score == m.score(plain, backend).score
    assert m.score(star_fe, frontend).score > m.score(plain, frontend).score


# ---------------------------------------------------------------- hackathon domain search

def test_search_any_domain(world):
    m, people, *_ = world
    res = m.search_people(people, domains=["Mobile (iOS)", "UI/UX Design"], exclude={"aditi"})
    assert {r.uid for r in res} == {"meera", "ananya"}


def test_search_all_domains_requires_every_domain(world):
    m, people, *_ = world
    res = m.search_people(people, domains=["Machine Learning", "Backend"], match="all")
    assert [r.uid for r in res] == ["aditi"]
    assert "[Machine Learning]" in res[0].why and "[Backend]" in res[0].why


def test_search_excludes_requester_and_respects_limit(world):
    m, people, *_ = world
    res = m.search_people(people, domains=["Machine Learning"], exclude={"aditi"}, k=1)
    assert len(res) == 1 and res[0].uid == "kabir"


def test_search_domain_by_skill_without_profile_label():
    p = _person("x", "Skill Only", "dev", [], [("Swift", "a", 2), ("SwiftUI", "a", 2)])
    m = Matcher(EmbeddingStore(HashingEmbedder()), now=NOW)
    res = m.search_people([p], domains=["Mobile (iOS)"])
    assert res and res[0].uid == "x"


def test_search_deeper_expert_ranks_higher():
    deep = _person("deep", "Deep", "x", ["Backend"], [("Flask", "a", 3), ("Django", "a", 2), ("PostgreSQL", "a", 2)])
    thin = _person("thin", "Thin", "x", ["Backend"], [("Flask", "b", 0)])
    m = Matcher(EmbeddingStore(HashingEmbedder()), now=NOW)
    assert [r.uid for r in m.search_people([thin, deep], domains=["Backend"])] == ["deep", "thin"]


def test_search_with_skills_and_free_text(world):
    m, people, *_ = world
    res = m.search_people(people, domains=["Frontend"], skills=["React"])
    assert res[0].uid == "sara"
    res = m.search_people(people, query="video editing for the demo")
    assert res[0].uid == "neel"


def test_search_unknown_domain_or_nobody_matches(world):
    m, people, *_ = world
    assert m.search_people(people, domains=["Game Development"]) == []
    assert m.search_people(people, domains=["Underwater Basket Weaving"]) == []
    assert m.search_people([], domains=["Backend"]) == []


def test_search_domain_input_is_normalised(world):
    m, people, *_ = world
    assert {r.uid for r in m.search_people(people, domains=["ios"])} == {"meera"}


# ---------------------------------------------------------------- projects for a person

def test_recommend_projects_for_ios_developer(world):
    m, people, by_uid, project = world
    res = m.recommend_projects(by_uid["meera"], [project], by_uid)
    assert len(res) == 1 and res[0].need_role == "iOS developer" and res[0].project_id == "skillswap"


def test_recommend_projects_skips_joined_completed_and_filled(world):
    m, people, by_uid, project = world
    meera = by_uid["meera"]
    project.status = "completed"
    assert m.recommend_projects(meera, [project], by_uid) == []
    project.status = "active"
    project.members.append(Member(uid="meera"))
    assert m.recommend_projects(meera, [project], by_uid) == []
    project.members.pop()
    for n in project.needs:
        n.filled_by = "someone"
    assert m.recommend_projects(meera, [project], by_uid) == []


# ---------------------------------------------------------------- the explanation

def test_why_only_mentions_facts_from_the_data(world):
    m, people, by_uid, project = world
    r = m.recommend_people(project, project.needs[0], people, by_uid["aditi"])[0]
    assert r.why.endswith(".") and "Swift" in r.why and "Python" not in r.why


def test_why_has_a_fallback_for_thin_matches():
    p = _person("n", "Nobody", "student", [], [("Cooking", "b", 0)])
    m = Matcher(EmbeddingStore(HashingEmbedder()), now=NOW)
    r = m.score(p, Need(role="Teammate"))
    assert r.why == "A related background."


# ---------------------------------------------------------------- relevance and avatars

def test_people_with_no_link_to_the_role_are_not_listed(world):
    m, people, by_uid, project = world
    ids = {r.uid for r in m.recommend_people(project, project.needs[0], people, by_uid["aditi"], k=20)}
    assert "meera" in ids and "neel" not in ids   # content/marketing person is not an iOS match


def test_fallback_why_mentions_background():
    p = _person("n", "Nobody", "student", ["Game Development"], [("Cooking", "b", 0)])
    m = Matcher(EmbeddingStore(HashingEmbedder()), now=NOW)
    assert m.score(p, Need(role="Teammate")).why == "Background in Game Development."


def test_gender_picks_avatar_automatically():
    from app.avatars import avatar_file, avatar_url, normalize_gender
    assert avatar_file("Female") == "avatar_female.png" and avatar_file(" M ") == "avatar_male.png"
    assert avatar_file("non-binary") == avatar_file("prefer not to say") == avatar_file(None) == "avatar_neutral.png"
    assert normalize_gender("whatever") == "unspecified"
    assert avatar_url("male") == "/static/avatars/avatar_male.png"


def test_avatar_files_exist():
    import pathlib
    from app.avatars import AVATAR_FILES
    root = pathlib.Path(__file__).resolve().parents[1] / "static" / "avatars"
    for name in set(AVATAR_FILES.values()):
        assert (root / name).is_file(), name


def test_person_avatar_follows_gender_and_survives_json():
    p = _person("a", "A", "x", [], [("Flask", "i", 1)])
    assert p.avatar == "avatar_neutral.png"
    p.gender = "female"
    assert p.avatar == "avatar_female.png"
    dumped = p.model_dump(mode="json")
    assert dumped["avatar"] == "avatar_female.png"
    assert Person(**dumped).avatar == "avatar_female.png"
    assert Person(**{**dumped, "gender": "Male"}).avatar == "avatar_male.png"

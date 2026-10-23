import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from app.matching import EmbeddingStore, HashingEmbedder, Matcher
from app.planner import PlanSession, apply_edits, apply_plan, require_leader
from app.planner.graph import MAX_REVISIONS
from app.planner.prompts import SYSTEM_PROMPT
from app.reputation.service import RuleError
from tests.test_planner import END, GOOD, NOW, FakeLLM, project, team


@pytest.fixture()
def matcher():
    return Matcher(EmbeddingStore(HashingEmbedder()), now=NOW)


def session(llm, matcher, checkpointer=None):
    return PlanSession(llm=llm, matcher=matcher, checkpointer=checkpointer or MemorySaver())


def kw(**extra):
    return dict(now=NOW, end=END, **extra)


def started(llm, matcher, extra=""):
    people = team()
    p = project(people)
    s = session(llm, matcher)
    return s, p, people, s.start(p, people, extra=extra, **kw())


# ---------------------------------------------------------------- start and confirm

def test_start_pauses_with_a_draft_and_one_llm_call(matcher):
    llm = FakeLLM(GOOD)
    s, p, people, draft = started(llm, matcher)
    assert draft.status == "review" and len(draft.milestones) == 6 and len(llm.prompts) == 1
    assert draft.revisions == 0 and draft.revisions_left == MAX_REVISIONS


def test_nothing_is_applied_until_the_leader_confirms(matcher):
    s, p, people, draft = started(FakeLLM(GOOD), matcher)
    assert p.milestones == []
    result = s.confirm(draft.session_id, p, people, **kw())
    assert result.ok and len(result.milestones) == 6 and p.milestones == []
    apply_plan(p, result)
    assert len(p.milestones) == 6


def test_extra_requirements_before_the_first_draft_reach_the_prompt(matcher):
    llm = FakeLLM(GOOD)
    started(llm, matcher, extra="Include a security review milestone")
    assert "Extra requirements from the project leader" in llm.prompts[0]
    assert "security review" in llm.prompts[0]


def test_start_needs_team_members(matcher):
    with pytest.raises(RuleError, match="no team members"):
        session(FakeLLM(GOOD), matcher).start(project(team()), [], **kw())


# ---------------------------------------------------------------- revise

REVISED = [g for g in GOOD if "Seed" not in g[0]] + [("Write security review notes", "Backend", 1)]


def test_revision_sends_feedback_and_current_draft_to_the_model(matcher):
    llm = FakeLLM(GOOD, REVISED)
    s, p, people, d1 = started(llm, matcher)
    d2 = s.revise(d1.session_id, p, people, feedback="Replace the seed task with a security review", **kw())
    assert d2.status == "review" and d2.revisions == 1 and d2.revisions_left == MAX_REVISIONS - 1
    assert "Write security review notes" in [m.title for m in d2.milestones]
    prompt = llm.prompts[1]
    assert "Changes the project leader asked for" in prompt and "security review" in prompt
    assert "Current draft:" in prompt and "1. Design the app screens in Figma [UI/UX Design, weight 2, owner Ananya Rao, due " in prompt


def test_each_revision_is_remembered_and_confirm_uses_the_latest_draft(matcher):
    llm = FakeLLM(GOOD, REVISED)
    s, p, people, d1 = started(llm, matcher)
    d2 = s.revise(d1.session_id, p, people, feedback="swap seed for security", **kw())
    result = s.confirm(d2.session_id, p, people, **kw())
    assert [m.title for m in result.milestones] == [m.title for m in d2.milestones]


def test_second_revision_keeps_the_first_requests(matcher):
    llm = FakeLLM(GOOD, REVISED, REVISED)
    s, p, people, d1 = started(llm, matcher)
    d2 = s.revise(d1.session_id, p, people, feedback="first request", **kw())
    s.revise(d2.session_id, p, people, feedback="second request", **kw())
    assert "first request" in llm.prompts[2] and "second request" in llm.prompts[2]


def test_revision_limit(matcher):
    llm = FakeLLM(GOOD)
    s, p, people, d = started(llm, matcher)
    for i in range(MAX_REVISIONS):
        d = s.revise(d.session_id, p, people, feedback=f"change {i}", **kw())
    assert d.revisions_left == 0
    with pytest.raises(RuleError, match="used all"):
        s.revise(d.session_id, p, people, feedback="one more", **kw())
    assert s.confirm(d.session_id, p, people, **kw()).ok     # confirming is still possible


def test_empty_feedback_is_refused(matcher):
    s, p, people, d = started(FakeLLM(GOOD), matcher)
    with pytest.raises(RuleError, match="what to change"):
        s.revise(d.session_id, p, people, feedback="   ", **kw())


def test_unknown_session(matcher):
    s, p, people, d = started(FakeLLM(GOOD), matcher)
    with pytest.raises(RuleError, match="not found"):
        s.confirm("nope", p, people, **kw())


def test_failed_revision_keeps_the_previous_draft_usable(matcher):
    llm = FakeLLM(GOOD, GOOD[:1])          # the revision keeps coming back with 1 milestone
    s, p, people, d1 = started(llm, matcher)
    d2 = s.revise(d1.session_id, p, people, feedback="make it tiny", **kw())
    assert d2.status == "revision_failed" and d2.errors and len(d2.milestones) == 6
    with pytest.raises(RuleError, match="could not be completed"):
        s.revise(d1.session_id, p, people, feedback="again", **kw())
    result = s.confirm(d1.session_id, p, people, **kw())
    assert result.ok and len(result.milestones) == 6


def test_first_draft_that_never_validates_is_a_clear_failure(matcher):
    s, p, people, d = started(FakeLLM(GOOD[:1]), matcher)
    assert d.status == "failed" and d.errors and not d.milestones
    with pytest.raises(RuleError):
        s.confirm(d.session_id, p, people, **kw())


def test_llm_outage_during_revision_can_be_retried(matcher):
    calls = {"n": 0}

    def flaky(prompt, schema, system=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("providers down")
        return FakeLLM(GOOD)(prompt, schema, system)

    s, p, people, d1 = started(flaky, matcher)
    with pytest.raises(RuntimeError):
        s.revise(d1.session_id, p, people, feedback="add something", **kw())
    d2 = s.revise(d1.session_id, p, people, feedback="add something", **kw())     # try again
    assert d2.status == "review" and len(d2.milestones) == 6


# ---------------------------------------------------------------- persistence and plain state

def test_draft_survives_a_server_restart(matcher, tmp_path):
    db = str(tmp_path / "sessions.sqlite")
    people = team()
    p = project(people)
    s1 = PlanSession(llm=FakeLLM(GOOD), matcher=matcher,
                     checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    d = s1.start(p, people, **kw())
    # "restart": brand new objects, same file, and a different fake model
    s2 = PlanSession(llm=FakeLLM(REVISED), matcher=matcher,
                     checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    assert s2.current_draft(d.session_id, p, people, **kw()).status == "review"
    d2 = s2.revise(d.session_id, p, people, feedback="swap seed for security", **kw())
    assert d2.revisions == 1 and s2.confirm(d2.session_id, p, people, **kw()).ok


def test_saved_state_is_plain_data(matcher):
    s, p, people, d = started(FakeLLM(GOOD), matcher)
    values = s.graph.get_state({"configurable": {"thread_id": d.session_id}}).values
    assert all(isinstance(i, dict) for i in values["items"] + values["milestones"])
    assert "project" not in values and "people" not in values


# ---------------------------------------------------------------- leader's direct edits

def test_direct_edits_change_owner_deadline_and_title(matcher):
    s, p, people, d = started(FakeLLM(GOOD), matcher)
    first = d.milestones[0]
    new_due = NOW + timedelta(hours=5)
    result = s.confirm(d.session_id, p, people, **kw(edits=[
        {"id": first.id, "assignee_uid": "aditi", "due": new_due.isoformat(), "title": "Design screens in Figma"}]))
    m = result.milestones[0]
    assert (m.assignee_uid, m.due, m.title) == ("aditi", new_due, "Design screens in Figma")
    assert [x.id for x in result.milestones] == [x.id for x in d.milestones]
    assert result.milestones[1] == d.milestones[1]


@pytest.mark.parametrize("edit,message", [
    ({"id": "zzz"}, "Unknown milestone"),
    ({"id": "skillswap-m1", "assignee_uid": "stranger"}, "member of the project"),
    ({"id": "skillswap-m1", "due": (END + timedelta(days=1)).isoformat()}, "between now and the end"),
    ({"id": "skillswap-m1", "due": (NOW - timedelta(hours=1)).isoformat()}, "between now and the end"),
    ({"id": "skillswap-m1", "title": "x"}, "between 3 and 90"),
])
def test_bad_edits_are_refused(matcher, edit, message):
    s, p, people, d = started(FakeLLM(GOOD), matcher)
    with pytest.raises(RuleError, match=message):
        s.confirm(d.session_id, p, people, **kw(edits=[edit]))


def test_apply_edits_does_not_touch_the_original_list(matcher):
    s, p, people, d = started(FakeLLM(GOOD), matcher)
    before = [m.model_copy() for m in d.milestones]
    apply_edits(d.milestones, [{"id": d.milestones[0].id, "assignee_uid": "aditi"}], people, NOW, END)
    assert d.milestones == before


# ---------------------------------------------------------------- permissions, leader tasks, prompt rules

def test_only_the_leader_may_plan():
    people = team()
    p = project(people)
    require_leader(p, "aditi")
    with pytest.raises(RuleError, match="Only the project leader"):
        require_leader(p, "meera")
    p.status = "completed"
    with pytest.raises(RuleError, match="completed"):
        require_leader(p, "aditi")


def test_product_and_demo_work_goes_to_the_leader_without_a_recruiting_gap(matcher):
    plan = GOOD + [("Final demo and handover", "Product & Management", 1)]
    s, p, people, d = started(FakeLLM(plan), matcher)
    final = next(m for m in d.milestones if m.title == "Final demo and handover")
    assert final.assignee_uid == p.leader_uid and d.gaps == []
    assert not any("Product" in w for w in d.warnings)


def test_leader_assignment_follows_a_transferred_leader(matcher):
    plan = GOOD + [("Final demo and handover", "Product & Management", 1)]
    people = team()
    p = project(people)
    p.leader_uid = "meera"
    d = session(FakeLLM(plan), matcher).start(p, people, **kw())
    assert next(m for m in d.milestones if "handover" in m.title).assignee_uid == "meera"


def test_warnings_do_not_pile_up_across_retries(matcher):
    bad = [("Bad domain item", "Vibes", 1)] + GOOD[:3]
    llm = FakeLLM(bad, GOOD + [("Record promo video", "Content & Marketing", 1)])
    s, p, people, d = started(llm, matcher)
    assert d.status == "review" and sum("Content & Marketing" in w for w in d.warnings) == 1


def test_planner_is_told_ios_screens_are_not_frontend():
    assert "Frontend means websites" in SYSTEM_PROMPT and "Mobile (iOS)" in SYSTEM_PROMPT

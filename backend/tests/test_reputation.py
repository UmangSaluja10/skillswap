from datetime import datetime, timedelta, timezone

import pytest

from app.models import Member, Milestone
from app.reputation import UserRep, overall, standing_for, standings
from app.reputation import scoring
from app.reputation.service import (RuleError, active_leader_tags, complete_project, hand_over,
                                    lead_history, mark_milestone, new_project, report_leader_missed,
                                    transfer_leadership)

T0 = datetime(2026, 10, 23, 15, 30, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)


def team(n_members=2):
    """Aditi leads; members b, c ... are teammates."""
    users: dict[str, UserRep] = {}
    p = new_project("p1", "Campus Marketplace", "aditi", users, T0)
    for uid in ["umang", "aditya", "alok"][: n_members]:
        p.members.append(Member(uid=uid))
    return p, users


def add_ms(p, mid, assignee, due=T0 + 5 * HOUR, domain="Backend", weight=1):
    p.milestones.append(Milestone(id=mid, title=mid, domain=domain, assignee_uid=assignee, due=due, weight=weight))


def test_owner_is_leader_and_has_active_tag():
    p, users = team()
    assert p.leader_uid == "aditi"
    assert active_leader_tags(users["aditi"]) == ["Campus Marketplace"]


def test_leader_marks_teammate_done_on_time_and_late():
    p, users = team()
    add_ms(p, "m1", "umang")
    add_ms(p, "m2", "umang")
    e1 = mark_milestone(p, users, "m1", "done", "aditi", T0 + 2 * HOUR)
    e2 = mark_milestone(p, users, "m2", "done", "aditi", T0 + 9 * HOUR)
    assert (e1.kind, e2.kind) == ("on_time", "late")
    assert p.milestones[0].status == "done"
    assert standings(users["umang"], T0 + 10 * HOUR)["Backend"].points == 15


def test_done_uses_real_completion_time_when_given():
    p, users = team()
    add_ms(p, "m1", "umang")
    e = mark_milestone(p, users, "m1", "done", "aditi", T0 + 20 * HOUR, completed_at=T0 + HOUR)
    assert e.kind == "on_time"


def test_only_leader_can_mark_teammate_task():
    p, users = team()
    add_ms(p, "m1", "umang")
    with pytest.raises(RuleError, match="Only the project leader"):
        mark_milestone(p, users, "m1", "done", "aditya", T0)


def test_nobody_marks_their_own_task():
    p, users = team()
    add_ms(p, "m1", "umang")
    add_ms(p, "m2", "aditi")
    with pytest.raises(RuleError, match="own task"):
        mark_milestone(p, users, "m1", "done", "umang", T0)
    with pytest.raises(RuleError, match="own task"):
        mark_milestone(p, users, "m2", "done", "aditi", T0)


def test_leaders_own_task_can_be_approved_by_another_member():
    p, users = team()
    add_ms(p, "m2", "aditi")
    e = mark_milestone(p, users, "m2", "done", "umang", T0 + HOUR)
    assert e.uid == "aditi" and e.kind == "on_time"


def test_outsider_cannot_mark_anything():
    p, users = team()
    add_ms(p, "m1", "umang")
    with pytest.raises(RuleError, match="members"):
        mark_milestone(p, users, "m1", "done", "stranger", T0)


def test_missed_requires_deadline_passed():
    p, users = team()
    add_ms(p, "m1", "umang")
    with pytest.raises(RuleError, match="not passed"):
        mark_milestone(p, users, "m1", "missed", "aditi", T0 + HOUR)
    e = mark_milestone(p, users, "m1", "missed", "aditi", T0 + 6 * HOUR)
    assert e.kind == "missed"


def test_cannot_mark_twice_or_use_unknown_outcome():
    p, users = team()
    add_ms(p, "m1", "umang")
    mark_milestone(p, users, "m1", "done", "aditi", T0)
    with pytest.raises(RuleError, match="already"):
        mark_milestone(p, users, "m1", "done", "aditi", T0)
    add_ms(p, "m2", "umang")
    with pytest.raises(RuleError, match="Outcome"):
        mark_milestone(p, users, "m2", "banana", "aditi", T0)
    with pytest.raises(RuleError, match="does not exist"):
        mark_milestone(p, users, "nope", "done", "aditi", T0)


def test_excused_never_counts_against_reliability():
    p, users = team()
    for i in range(3):
        add_ms(p, f"d{i}", "umang")
        mark_milestone(p, users, f"d{i}", "done", "aditi", T0 + HOUR)
    add_ms(p, "x", "umang")
    mark_milestone(p, users, "x", "excused", "aditi", T0 + 2 * HOUR, note="family emergency")
    s = standings(users["umang"], T0 + 3 * HOUR)["Backend"]
    assert s.reliability == 1.0 and s.counted == 3


def test_missed_lowers_only_that_domain():
    p, users = team()
    for i in range(3):
        add_ms(p, f"f{i}", "umang", domain="Frontend")
        mark_milestone(p, users, f"f{i}", "done", "aditi", T0 + HOUR)
    for i in range(3):
        add_ms(p, f"b{i}", "umang", domain="Backend")
        mark_milestone(p, users, f"b{i}", "missed", "aditi", T0 + 6 * HOUR)
    st = standings(users["umang"], T0 + 7 * HOUR)
    assert st["Frontend"].reliability == 1.0
    assert st["Backend"].reliability == 0.0
    assert st["Frontend"].label == "Highly reliable" and st["Backend"].label == "Building reliability"
    assert st["Backend"].points == 0


def test_abandon_hurts_more_than_missed():
    p, users = team()
    for i in range(2):
        add_ms(p, f"d{i}", "umang")
        mark_milestone(p, users, f"d{i}", "done", "aditi", T0 + HOUR)
    add_ms(p, "a", "umang")
    mark_milestone(p, users, "a", "abandoned", "aditi", T0 + HOUR)
    add_ms(p, "d", "aditya")
    for i in range(2):
        add_ms(p, f"e{i}", "aditya")
        mark_milestone(p, users, f"e{i}", "done", "aditi", T0 + HOUR)
    add_ms(p, "m", "aditya")
    mark_milestone(p, users, "m", "missed", "aditi", T0 + 6 * HOUR)
    now = T0 + 7 * HOUR
    assert standings(users["umang"], now)["Backend"].reliability < standings(users["aditya"], now)["Backend"].reliability


def test_new_until_three_reviewed_tasks():
    p, users = team()
    for i in range(2):
        add_ms(p, f"d{i}", "umang")
        mark_milestone(p, users, f"d{i}", "done", "aditi", T0 + HOUR)
    s = standings(users["umang"], T0 + 2 * HOUR)["Backend"]
    assert s.reliability is None and s.label == "New"
    assert standing_for(None, "Backend", T0).label == "New"


def test_levels_follow_points_and_reliability():
    assert scoring._level(0, None) == "Newcomer"
    assert scoring._level(25, None) == "Contributor"
    assert scoring._level(100, 0.9) == "Trusted"
    assert scoring._level(100, 0.5) == "Contributor"   # points alone are not trust
    assert scoring._level(100, None) == "Contributor"


def test_weight_scales_points():
    p, users = team()
    add_ms(p, "big", "umang", weight=3)
    mark_milestone(p, users, "big", "done", "aditi", T0)
    assert standings(users["umang"], T0)["Backend"].points == 30


def test_old_failures_fade_so_a_student_can_recover():
    p, users = team()
    for i in range(3):
        add_ms(p, f"m{i}", "umang")
        mark_milestone(p, users, f"m{i}", "missed", "aditi", T0 + 6 * HOUR)
    now_soon = T0 + timedelta(days=1)
    assert standings(users["umang"], now_soon)["Backend"].reliability == 0.0
    # months later: three fresh on-time tasks outweigh the old misses
    later = T0 + timedelta(days=200)
    for i in range(3):
        users["umang"].events.append(users["umang"].events[0].model_copy(
            update={"id": f"n{i}", "kind": "on_time", "at": later}))
    assert standings(users["umang"], later)["Backend"].reliability > 0.8


def test_overall_combines_domains():
    p, users = team()
    for i, d in enumerate(["Backend", "Frontend", "Backend"]):
        add_ms(p, f"m{i}", "umang", domain=d)
        mark_milestone(p, users, f"m{i}", "done", "aditi", T0 + HOUR)
    o = overall(users["umang"], T0 + 2 * HOUR)
    assert o.points == 30 and o.counted == 3 and o.reliability == 1.0


# ---------------------------------------------------------------- leader accountability

def test_leader_task_needs_all_teammates_to_report():
    p, users = team(3)
    add_ms(p, "L", "aditi")
    late = T0 + 6 * HOUR
    r1 = report_leader_missed(p, users, "L", "umang", late)
    r2 = report_leader_missed(p, users, "L", "aditya", late)
    assert (r1["applied"], r2["applied"]) == (False, False) and r2["reports"] == 2 and r2["needed"] == 3
    assert p.milestones[0].status == "pending" and not users["aditi"].events
    r3 = report_leader_missed(p, users, "L", "alok", late)
    assert r3["applied"] is True
    ev = users["aditi"].events[0]
    assert ev.kind == "missed" and ev.marked_by == "team" and ev.domain == "Backend"
    assert p.milestones[0].status == "missed"


def test_duplicate_report_does_not_count_twice():
    p, users = team(2)
    add_ms(p, "L", "aditi")
    late = T0 + 6 * HOUR
    report_leader_missed(p, users, "L", "umang", late)
    again = report_leader_missed(p, users, "L", "umang", late)
    assert again == {"applied": False, "reports": 1, "needed": 2}


def test_report_rules():
    p, users = team(2)
    add_ms(p, "L", "aditi")
    add_ms(p, "T", "umang")
    with pytest.raises(RuleError, match="not passed"):
        report_leader_missed(p, users, "L", "umang", T0)
    with pytest.raises(RuleError, match="only the leader's own"):
        report_leader_missed(p, users, "T", "aditya", T0 + 9 * HOUR)
    with pytest.raises(RuleError, match="teammates of the leader"):
        report_leader_missed(p, users, "L", "aditi", T0 + 9 * HOUR)
    with pytest.raises(RuleError, match="teammates of the leader"):
        report_leader_missed(p, users, "L", "stranger", T0 + 9 * HOUR)


def test_leader_working_alone_cannot_be_reported():
    p, users = team(0)
    add_ms(p, "L", "aditi")
    with pytest.raises(RuleError):
        report_leader_missed(p, users, "L", "aditi", T0 + 9 * HOUR)


def test_one_dissenting_teammate_protects_the_leader():
    p, users = team(2)
    add_ms(p, "L", "aditi")
    report_leader_missed(p, users, "L", "umang", T0 + 9 * HOUR)
    assert p.milestones[0].status == "pending"
    # the leader can still finish it and it simply counts as late
    e = mark_milestone(p, users, "L", "done", "aditya", T0 + 10 * HOUR)
    assert e.kind == "late"


# ---------------------------------------------------------------- handover

def test_proper_handover_is_free_and_task_moves_on():
    p, users = team()
    add_ms(p, "m1", "umang")
    e = hand_over(p, users, "m1", "aditya", "umang", T0 + HOUR, note="exam week")
    assert e.kind == "handed_over" and p.milestones[0].assignee_uid == "aditya"
    assert p.milestones[0].status == "pending" and p.milestones[0].handover_from == ["umang"]
    assert standings(users["umang"], T0)["Backend"].counted == 0
    mark_milestone(p, users, "m1", "done", "aditi", T0 + 2 * HOUR)
    assert standings(users["aditya"], T0 + 3 * HOUR)["Backend"].points == 10


def test_handover_rules():
    p, users = team()
    add_ms(p, "m1", "umang")
    with pytest.raises(RuleError, match="assignee or the project leader"):
        hand_over(p, users, "m1", "aditya", "aditya", T0)
    with pytest.raises(RuleError, match="member of the project"):
        hand_over(p, users, "m1", "stranger", "umang", T0)
    with pytest.raises(RuleError, match="already has"):
        hand_over(p, users, "m1", "umang", "umang", T0)


# ---------------------------------------------------------------- leader tag

def test_transfer_moves_tag_and_keeps_history_for_both():
    p, users = team()
    transfer_leadership(p, users, "umang", "aditi", T0 + HOUR)
    assert p.leader_uid == "umang"
    assert active_leader_tags(users["aditi"]) == []
    assert active_leader_tags(users["umang"]) == ["Campus Marketplace"]
    assert "transferred" in lead_history(users["aditi"])[0]
    assert "ongoing" in lead_history(users["umang"])[0]


def test_transfer_rules():
    p, users = team()
    with pytest.raises(RuleError, match="current leader"):
        transfer_leadership(p, users, "umang", "umang", T0)
    with pytest.raises(RuleError, match="member"):
        transfer_leadership(p, users, "stranger", "aditi", T0)
    with pytest.raises(RuleError, match="already"):
        transfer_leadership(p, users, "aditi", "aditi", T0)


def test_new_leader_marks_tasks_old_leader_cannot():
    p, users = team()
    add_ms(p, "m1", "aditya")
    transfer_leadership(p, users, "umang", "aditi", T0)
    with pytest.raises(RuleError, match="Only the project leader"):
        mark_milestone(p, users, "m1", "done", "aditi", T0)
    assert mark_milestone(p, users, "m1", "done", "umang", T0).kind == "on_time"


def test_completing_project_removes_tag_but_history_stays_forever():
    p, users = team()
    complete_project(p, users, "aditi", T0 + 20 * HOUR)
    assert p.status == "completed" and p.completed_at is not None
    assert active_leader_tags(users["aditi"]) == []
    line = lead_history(users["aditi"])[0]
    assert line.startswith("Led: Campus Marketplace") and "completed" in line


def test_completed_project_is_frozen_and_only_leader_completes():
    p, users = team()
    add_ms(p, "m1", "umang")
    with pytest.raises(RuleError, match="Only the project leader"):
        complete_project(p, users, "umang", T0)
    complete_project(p, users, "aditi", T0)
    with pytest.raises(RuleError, match="completed"):
        mark_milestone(p, users, "m1", "done", "aditi", T0)
    with pytest.raises(RuleError, match="completed"):
        transfer_leadership(p, users, "umang", "aditi", T0)


def test_same_person_can_lead_many_projects_history_lists_each():
    users: dict[str, UserRep] = {}
    p1 = new_project("a", "Alpha", "aditi", users, T0)
    p2 = new_project("b", "Beta", "aditi", users, T0 + 5 * HOUR)
    complete_project(p1, users, "aditi", T0 + 6 * HOUR)
    assert active_leader_tags(users["aditi"]) == ["Beta"]
    lines = lead_history(users["aditi"])
    assert len(lines) == 2 and lines[0].startswith("Led: Beta")


def test_models_roundtrip_through_json_like_firebase():
    from app.models import Project
    p, users = team()
    add_ms(p, "m1", "umang")
    mark_milestone(p, users, "m1", "done", "aditi", T0 + HOUR)
    again = Project(**p.model_dump(mode="json"))
    assert again == p
    rep = UserRep(**users["umang"].model_dump(mode="json"))
    assert rep == users["umang"]

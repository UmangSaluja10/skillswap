from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.matching import EmbeddingStore, HashingEmbedder, Matcher
from app.planner import PlanItem, PlanLLMOutput, PlanSession
from tests.test_planner import END, GOOD, NOW, team, project

IST = ZoneInfo("Asia/Kolkata")


class Script:
    """Fake model: returns the queued outputs in order (the last one repeats) and records prompts."""

    def __init__(self, *outputs):
        self.outputs, self.prompts = list(outputs), []

    def __call__(self, prompt, schema, system=None):
        self.prompts.append(prompt)
        return self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]


def out(rows=GOOD, notes="", **pins):
    """rows like GOOD; pins = {title_start: {"due_at": ..., "assignee_name": ...}}"""
    items = []
    for t, d, w in rows:
        extra = next((v for k, v in pins.items() if t.startswith(k)), {})
        items.append(PlanItem(title=t, domain=d, weight=w, **extra))
    return PlanLLMOutput(milestones=items, notes=notes)


def iso(dt):
    return dt.astimezone(IST).isoformat()


@pytest.fixture()
def matcher():
    return Matcher(EmbeddingStore(HashingEmbedder()), now=NOW)


def start(llm, matcher, end=END):
    people = team()
    p = project(people)
    s = PlanSession(llm=llm, matcher=matcher, checkpointer=MemorySaver())
    return s, p, people, s.start(p, people, now=NOW, end=end)


def revise(s, d, p, people, text, end=END):
    return s.revise(d.session_id, p, people, now=NOW, end=end, feedback=text)


TRACKER = "Build SwiftUI profile"   # the 4th milestone in GOOD


def test_pinned_deadline_is_kept_and_the_rest_flow_around_it(matcher):
    pin = NOW + timedelta(hours=14)
    llm = Script(out(), out(notes="Moved it.", **{TRACKER: {"due_at": iso(pin)}}))
    s, p, people, d1 = start(llm, matcher)
    d2 = revise(s, d1, p, people, "Make the SwiftUI task due in 14 hours, the rest as per that")
    assert d2.changed is True and d2.notes == "Moved it."
    dues = {m.title: m.due for m in d2.milestones}
    assert dues[next(t for t in dues if t.startswith(TRACKER))] == pin     # exactly as asked
    ordered = [m.due for m in d2.milestones]
    assert ordered == sorted(ordered) and len(set(ordered)) == len(ordered)
    idx = next(i for i, m in enumerate(d2.milestones) if m.title.startswith(TRACKER))
    assert all(m.due < pin for m in d2.milestones[:idx]) and all(m.due > pin for m in d2.milestones[idx + 1:])
    assert ordered[-1] <= END - timedelta(hours=2)


def test_pin_after_the_end_extends_the_project_by_default(matcher):
    sunday = END + timedelta(hours=20)
    llm = Script(out(), out(notes="Moved it to Sunday.", **{TRACKER: {"due_at": iso(sunday)}}))
    s, p, people, d1 = start(llm, matcher)
    d2 = revise(s, d1, p, people, "make the SwiftUI task due sunday morning")
    assert d2.status == "review" and d2.changed is True and d2.extended is True
    assert d2.end > END and len(llm.prompts) == 2                        # no retry needed
    pinned = next(m for m in d2.milestones if m.title.startswith(TRACKER))
    assert pinned.due == sunday
    assert all(m.due <= d2.end for m in d2.milestones)
    after = [m for m in d2.milestones if m.due > sunday]
    assert after and all(m.due > sunday for m in after)                  # later tasks follow the pinned one
    assert d2.end >= max(m.due for m in d2.milestones) + timedelta(minutes=15)


def test_extended_end_is_remembered_and_applied_to_the_project(matcher):
    from app.planner import apply_plan
    sunday = END + timedelta(hours=20)
    llm = Script(out(), out(**{TRACKER: {"due_at": iso(sunday)}}))
    s, p, people, d1 = start(llm, matcher)
    d2 = revise(s, d1, p, people, "sunday morning")
    result = s.confirm(d2.session_id, p, people, now=NOW, end=END)
    assert result.end == d2.end and result.extended
    apply_plan(p, result)
    assert p.deadline == d2.end and p.deadline > END


def test_prompt_shows_the_extended_end_in_later_revisions(matcher):
    sunday = END + timedelta(hours=20)
    llm = Script(out(), out(**{TRACKER: {"due_at": iso(sunday)}}), out())
    s, p, people, d1 = start(llm, matcher)
    d2 = revise(s, d1, p, people, "sunday morning")
    revise(s, d2, p, people, "keep going")
    assert "The leader may extend the project end." in llm.prompts[1]
    assert "Project ends: Sun 25 Oct" in llm.prompts[2] or "Project ends: Mon 26 Oct" in llm.prompts[2]


def test_fixed_end_refuses_a_later_deadline_with_a_clear_reason(matcher):
    too_late = END + timedelta(hours=20)
    llm = Script(out(), out(**{TRACKER: {"due_at": iso(too_late)}}),
                 out(notes="Sunday is after the fixed event end on Saturday, so I kept the schedule."))
    people = team()
    p = project(people)
    s = PlanSession(llm=llm, matcher=matcher, checkpointer=MemorySaver())
    d1 = s.start(p, people, now=NOW, end=END, max_end=END)
    assert "The project end is fixed and cannot be extended." in llm.prompts[0]
    d2 = s.revise(d1.session_id, p, people, now=NOW, end=END, feedback="make the tracker due sunday", max_end=END)
    assert "outside the project window" in llm.prompts[2] and "fixed" in llm.prompts[2]   # model told why
    assert d2.status == "review" and d2.changed is False and d2.extended is False and d2.end == END
    assert "fixed event end" in d2.notes
    assert d2.warnings[0] == "The planner made no changes to the draft."


def test_extension_is_capped_at_sixty_days(matcher):
    far = NOW + timedelta(days=120)
    llm = Script(out(), out(**{TRACKER: {"due_at": iso(far)}}), out(notes="That is too far out."))
    s, p, people, d1 = start(llm, matcher)
    d2 = revise(s, d1, p, people, "due in four months")
    assert "outside the project window" in llm.prompts[2] and "60 days" in llm.prompts[2]
    assert d2.extended is False


def test_the_first_draft_is_not_marked_extended(matcher):
    s, p, people, d1 = start(Script(out()), matcher)
    assert d1.extended is False and d1.end == END


def test_user_can_extend_the_window_and_then_the_same_request_works(matcher):
    sunday = NOW + timedelta(hours=44)
    longer_end = NOW + timedelta(hours=72)
    llm = Script(out(), out(**{TRACKER: {"due_at": iso(sunday)}}))
    s, p, people, d1 = start(llm, matcher, end=longer_end)
    d2 = revise(s, d1, p, people, "tracker due sunday morning", end=longer_end)
    assert d2.changed and next(m.due for m in d2.milestones if m.title.startswith(TRACKER)) == sunday


def test_naive_deadline_is_read_in_the_display_time_zone(matcher):
    # 22:00 in India on Friday = 16:30 UTC
    llm = Script(out(), out(**{TRACKER: {"due_at": "2026-10-23T22:00:00"}}))
    s, p, people, d1 = start(llm, matcher)
    d2 = revise(s, d1, p, people, "due 10pm tonight")
    assert next(m.due for m in d2.milestones if m.title.startswith(TRACKER)) == datetime(2026, 10, 23, 16, 30, tzinfo=timezone.utc)


def test_unreadable_deadline_is_ignored_with_a_warning(matcher):
    llm = Script(out(), out(**{TRACKER: {"due_at": "next sunday-ish"}}))
    s, p, people, d1 = start(llm, matcher)
    d2 = revise(s, d1, p, people, "make it due sometime next sunday")
    assert any("unreadable deadline" in w for w in d2.warnings)


def test_pinned_owner_by_exact_or_first_name(matcher):
    llm = Script(out(), out(**{"Build the resume parsing": {"assignee_name": "Meera Joshi"},
                               "Implement matching": {"assignee_name": "ananya"}}))
    s, p, people, d1 = start(llm, matcher)
    d2 = revise(s, d1, p, people, "give the API to Meera and matching to Ananya")
    owner = {m.title: m.assignee_uid for m in d2.milestones}
    assert owner["Build the resume parsing API"] == "meera" and owner["Implement matching endpoint"] == "ananya"


def test_unknown_owner_name_falls_back_to_automatic_with_a_warning(matcher):
    llm = Script(out(), out(**{"Build the resume parsing": {"assignee_name": "Zorro"}}))
    s, p, people, d1 = start(llm, matcher)
    d2 = revise(s, d1, p, people, "give the API to Zorro")
    assert any("no teammate called 'Zorro'" in w for w in d2.warnings)
    assert next(m.assignee_uid for m in d2.milestones if "parsing" in m.title) == "aditi"


def test_pinned_deadlines_out_of_order_are_sent_back_to_the_model(matcher):
    early, late = NOW + timedelta(hours=5), NOW + timedelta(hours=15)
    bad = out(**{"Design the app": {"due_at": iso(late)}, "Build the resume": {"due_at": iso(early)}})
    llm = Script(out(), bad, out(notes="Kept the order."))
    s, p, people, d1 = start(llm, matcher)
    revise(s, d1, p, people, "swap those deadlines")
    assert "must be later" in llm.prompts[2]


def test_pins_survive_later_revisions(matcher):
    pin = NOW + timedelta(hours=14)
    llm = Script(out(), out(**{TRACKER: {"due_at": iso(pin)}}), out())     # revision 2 forgets to repeat the pin
    s, p, people, d1 = start(llm, matcher)
    d2 = revise(s, d1, p, people, "tracker due in 14 hours")
    d3 = revise(s, d2, p, people, "add nothing else, just keep going")
    assert "deadline PINNED" in llm.prompts[2]
    assert next(m.due for m in d3.milestones if m.title.startswith(TRACKER)) == pin


def test_current_draft_lines_show_owner_and_deadline_to_the_model(matcher):
    llm = Script(out(), out())
    s, p, people, d1 = start(llm, matcher)
    revise(s, d1, p, people, "anything")
    assert "owner Ananya Rao, due Fri 23 Oct" in llm.prompts[1]
    assert "Now: Fri 23 Oct 15:30 (Asia/Kolkata, UTC+05:30)" in llm.prompts[1]
    assert "Project ends: Sat 24 Oct 15:30" in llm.prompts[1]


def test_no_op_revision_without_a_note_tells_the_leader_what_to_try(matcher):
    llm = Script(out(), out())
    s, p, people, d1 = start(llm, matcher)
    d2 = revise(s, d1, p, people, "do something impossible")
    assert d2.changed is False and "edit owners and deadlines directly" in d2.warnings[0]


def test_first_draft_has_no_changed_flag(matcher):
    s, p, people, d1 = start(Script(out()), matcher)
    assert d1.changed is None and d1.notes == ""


def test_pins_stay_plain_data_in_the_saved_state(matcher):
    pin = NOW + timedelta(hours=14)
    llm = Script(out(), out(**{TRACKER: {"due_at": iso(pin)}}))
    s, p, people, d1 = start(llm, matcher)
    d2 = revise(s, d1, p, people, "tracker due in 14 hours")
    values = s.graph.get_state({"configurable": {"thread_id": d2.session_id}}).values
    pinned = [i for i in values["items"] if i["due_at"]]
    assert len(pinned) == 1 and isinstance(pinned[0]["due_at"], str)


def test_deadline_with_last_years_date_is_moved_to_the_current_year(matcher):
    sat = END - timedelta(hours=10)
    old_year = sat.replace(year=sat.year - 1)
    llm = Script(out(), out(notes="Moved.", **{TRACKER: {"due_at": iso(old_year)}}))
    s, p, people, d1 = start(llm, matcher)
    d2 = revise(s, d1, p, people, "move the tracker a bit")
    pinned = next(m for m in d2.milestones if m.title.startswith(TRACKER))
    assert d2.status == "review" and d2.changed is True and pinned.due == sat


def test_repeating_current_times_does_not_pin_every_milestone(matcher):
    s, p, people, d1 = start(Script(out()), matcher)
    echo = {m.title: {"due_at": iso(m.due.replace(year=m.due.year - 1))} for m in d1.milestones}
    llm = Script(out(), out(notes="No change.", **echo))
    s, p, people, d1 = start(llm, matcher)
    d2 = revise(s, d1, p, people, "looks fine, just double check")
    assert d2.errors == [] and d2.end == END and not d2.extended

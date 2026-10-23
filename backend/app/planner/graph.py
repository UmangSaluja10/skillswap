"""The planner as a LangGraph graph:

    plan -> assign -> schedule -> validate -+-> review (pauses for the leader) -> done
                 ^______ retry (max 2) _____|          |
                 ^_____________ revise (leader's feedback) ______|

Only `plan` talks to the LLM. Assigning, scheduling and validating are plain code, so the result is
reliable and a draft costs one LLM call. Problems the model can fix (too few milestones, an unknown
domain, duplicate titles) send the graph back to `plan` with the problems as feedback.

`review` uses LangGraph's `interrupt`: the graph stops with a draft, the leader confirms or asks for
changes, and the graph resumes. The saved state is plain JSON-like data; the project and the people
are passed in on every call (they come from the database), never stored in the checkpoint.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..matching.engine import Matcher
from ..models import Milestone, Need, Person, Project
from ..resume.taxonomy import DOMAINS, normalize_domain, skill_domain
from .prompts import SYSTEM_PROMPT, build_prompt
from .schemas import PlanItem, PlanLLMOutput, PlanResult

MAX_RETRIES = 2            # automatic fixes per draft, so at most 3 LLM calls for one draft
MAX_REVISIONS = 3          # times the leader may ask for changes
MIN_MILESTONES = 3
MAX_MILESTONES = 10
LOAD_PENALTY = 0.4         # how strongly assignment avoids piling work on one person
LEADER_DOMAINS = {"Product & Management"}   # coordination / demo work goes to the project leader
STEP = timedelta(minutes=15)
MAX_EXTENSION = timedelta(days=60)   # the leader may push the project end at most this far
TAIL_PER_WEIGHT = timedelta(hours=3)  # room kept for milestones that follow a late pinned deadline

LLMFn = Callable[..., PlanLLMOutput]


class PlanState(TypedDict, total=False):
    """Everything here is plain data so it can be saved in a checkpoint."""
    brief: str
    requirements: list[str]     # what the leader asked for (before the first draft and in revisions)
    feedback: list[str]         # validation problems for the model to fix
    attempt: int
    revisions: int
    items: list[dict]           # PlanItem as dicts
    assignees: list[str]
    milestones: list[dict]      # Milestone as dicts
    gaps: list[dict]            # Need as dicts
    warnings: list[str]
    plan_errors: list[str]      # the model can fix these by planning again
    fatal_errors: list[str]     # retrying will not help (e.g. no time left)
    notes: str                  # the planner's own note: what it changed, or why it could not
    end_eff: str                # the project end for this draft (ISO): the original end, or the extended one
    decision: str               # "confirm" | "revise"
    good: dict                  # last draft that passed validation (kept in case a revision fails)


def _hours(start: datetime, end: datetime) -> float:
    return max(0.0, (end - start).total_seconds() / 3600)


def _round_up(dt: datetime) -> datetime:
    """Next quarter-hour (deadlines like 18:45 look sensible; 18:37 does not)."""
    dt = dt.replace(second=0, microsecond=0)
    rest = dt.minute % 15
    return dt if rest == 0 else dt + timedelta(minutes=15 - rest)


def _ctx(config) -> dict:
    return config["configurable"]


DEFAULT_TZ = "Asia/Kolkata"


def _tz(c: dict) -> ZoneInfo:
    try:
        return ZoneInfo(c.get("tz") or DEFAULT_TZ)
    except Exception:
        return ZoneInfo("UTC")


def _fmt(dt: datetime, tz: ZoneInfo) -> str:
    return dt.astimezone(tz).strftime("%a %d %b %H:%M")


def _parse_due(raw: str | None, tz: ZoneInfo) -> datetime | None:
    """The model's ISO text -> aware UTC datetime. No offset given = the display time zone."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(timezone.utc)


def _fix_year(dt: datetime, now: datetime, tz: ZoneInfo) -> datetime:
    """Models often write a deadline with last year's date. A date in the past is moved to the current
    (or next) year, keeping the month, day and time."""
    if dt >= now - timedelta(hours=1):
        return dt
    local = dt.astimezone(tz)
    for year in (now.astimezone(tz).year, now.astimezone(tz).year + 1):
        try:
            moved = local.replace(year=year).astimezone(timezone.utc)
        except ValueError:                      # 29 Feb
            continue
        if moved >= now - timedelta(hours=1):
            return moved
    return dt


def _find_member(name: str | None, people: list[Person]) -> Person | None:
    """Exact name, or a first name / part of a name that points to exactly one teammate."""
    if not name:
        return None
    key = name.strip().casefold()
    exact = [p for p in people if p.name.casefold() == key]
    if exact:
        return exact[0]
    near = [p for p in people if key in p.name.casefold().split() or key in p.name.casefold()]
    return near[0] if len(near) == 1 else None


def build_graph(llm: LLMFn | None = None, matcher: Matcher | None = None, *, review: bool = False,
                checkpointer=None):
    """Returns the compiled graph. `review=True` adds the pause for the leader (needs a checkpointer).

    Per-call context is passed as `config={"configurable": {"project", "people", "now", "end", ...}}`.
    """

    def default_llm(prompt: str, schema, system: str | None = None) -> PlanLLMOutput:
        from ..llm import call_llm_json
        return call_llm_json(prompt, schema, system=system, temperature=0.2, max_tokens=4096)

    ask = llm or default_llm
    scorer = matcher or Matcher()

    # ------------------------------------------------------------------ 1. plan (LLM)
    def plan(state: PlanState, config) -> dict:
        c = _ctx(config)
        tz = _tz(c)
        people: list[Person] = c["people"]
        team = [f"{p.name}: {', '.join(p.profile.domains) or 'no domain listed'}" for p in people]
        names = {p.uid: p.name for p in people}
        # During a revision, "previous" is the last draft the leader saw, not a failed attempt at the new one.
        base = state.get("good") if state.get("revisions", 0) else None
        previous_items = [PlanItem(**i) for i in (base or {}).get("items", [])]
        previous_milestones = (base or {}).get("milestones", [])
        previous_lines = []
        for n, it in enumerate(previous_items):
            m = previous_milestones[n] if n < len(previous_milestones) else None
            line = f"{it.title} [{it.domain}, weight {it.weight}"
            if m:
                line += f", owner {names.get(m['assignee_uid'], '?')}, due {_fmt(Milestone(**m).due, tz)} " \
                         f"({Milestone(**m).due.astimezone(tz).isoformat(timespec='minutes')})"
            if it.assignee_name:
                line += ", owner PINNED"
            if it.due_at:
                line += ", deadline PINNED"
            previous_lines.append(line + "]" + (f" pinned_due_at={it.due_at}" if it.due_at else "")
                                  + (f" pinned_assignee_name={it.assignee_name}" if it.assignee_name else ""))
        shown_end = _parse_due((base or {}).get("end_eff"), tz) or c["end"]
        cap = c.get("max_end")
        fixed = cap is not None and cap <= shown_end
        clock = (f"Now: {_fmt(c['now'], tz)} ({tz.key}, UTC{c['now'].astimezone(tz).strftime('%z')[:3]}:"
                 f"{c['now'].astimezone(tz).strftime('%z')[3:]}). Today's date is {c['now'].astimezone(tz):%Y-%m-%d}; "
                 f"always write deadlines in the current year. Project ends: {_fmt(shown_end, tz)}. "
                 + ("The project end is fixed and cannot be extended." if fixed
                    else "The leader may extend the project end."))
        prompt = build_prompt(state["brief"], team, _hours(c["now"], shown_end), state.get("feedback", []),
                              state.get("requirements", []), previous_lines, clock)
        out = ask(prompt, PlanLLMOutput, system=SYSTEM_PROMPT)
        carried = {i.title.casefold(): i for i in previous_items}
        warnings: list[str] = []          # recomputed by every pass, so nothing piles up across retries
        items: list[dict] = []
        for it in out.milestones:
            domain = normalize_domain(it.domain) or skill_domain(it.domain) or it.domain
            due_iso, owner = None, it.assignee_name
            if it.due_at:
                due = _parse_due(it.due_at, tz)
                if due is None:
                    warnings.append(f"Ignored an unreadable deadline for '{it.title}'.")
                else:
                    due = _fix_year(due, c["now"], tz)
                    due_iso = due.isoformat()
            old = carried.get(it.title.casefold())      # pins from earlier revisions stay unless changed again
            if due_iso and old is not None and not old.due_at:
                # The model often repeats a milestone's current time; that is an echo, not a request.
                n = next((k for k, x in enumerate(previous_items) if x is old), None)
                shown = Milestone(**previous_milestones[n]).due if n is not None and n < len(previous_milestones) else None
                if shown is not None and abs((_parse_due(due_iso, tz) - shown).total_seconds()) < 60:
                    due_iso = None
            if old is not None:
                due_iso = due_iso or old.due_at
                owner = owner or old.assignee_name
            items.append(it.model_copy(update={"domain": domain, "due_at": due_iso,
                                               "assignee_name": owner}).model_dump(mode="json"))
        if len(items) > MAX_MILESTONES:
            warnings.append(f"The plan had {len(items)} milestones; kept the first {MAX_MILESTONES}.")
            items = items[:MAX_MILESTONES]
        return {"items": items, "attempt": state.get("attempt", 0) + 1, "warnings": warnings, "notes": out.notes,
                "gaps": [], "milestones": [], "assignees": [], "plan_errors": [], "fatal_errors": []}

    # ------------------------------------------------------------------ 2. assign (code)
    def assign(state: PlanState, config) -> dict:
        c = _ctx(config)
        people: list[Person] = c["people"]
        project: Project = c["project"]
        items = [PlanItem(**i) for i in state["items"]]
        if not people or not items:
            return {"assignees": []}
        total = sum(i.weight for i in items) or 1
        load = {p.uid: 0 for p in people}
        leader = next((p for p in people if p.uid == project.leader_uid), None)
        chosen: dict[int, str] = {}
        gaps: list[Need] = []
        warnings = list(state.get("warnings", []))
        for idx in sorted(range(len(items)), key=lambda k: (-items[k].weight, k)):   # big pieces first
            item = items[idx]
            if item.assignee_name:
                pinned = _find_member(item.assignee_name, people)
                if pinned is not None:
                    chosen[idx] = pinned.uid
                    load[pinned.uid] += item.weight
                    continue
                warnings.append(f"There is no teammate called '{item.assignee_name}'; '{item.title}' was assigned automatically.")
            if item.domain in LEADER_DOMAINS and leader is not None:
                chosen[idx] = leader.uid
                load[leader.uid] += item.weight
                continue
            need = Need(role=item.title, domain=item.domain if item.domain in DOMAINS else None,
                        description=item.description)
            scored = []
            for p in people:
                m = scorer.score(p, need)
                scored.append((m.score - LOAD_PENALTY * (load[p.uid] / total), m, p))
            scored.sort(key=lambda t: (-t[0], t[2].uid))
            _, m, person = scored[0]
            chosen[idx] = person.uid
            load[person.uid] += item.weight
            if need.domain and (m.breakdown.get("domain") or 0) == 0:
                if not any(g.domain == need.domain for g in gaps):
                    gaps.append(Need(role=f"{need.domain} contributor", domain=need.domain,
                                     description=f"Needed for: {item.title}"))
                warnings.append(f"Nobody on the team lists {need.domain}; '{item.title}' went to "
                                f"{person.name}. Consider recruiting for it.")
        return {"assignees": [chosen[i] for i in range(len(items))],
                "gaps": [g.model_dump() for g in gaps], "warnings": warnings}

    # ------------------------------------------------------------------ 3. schedule (code)
    def schedule(state: PlanState, config) -> dict:
        c = _ctx(config)
        start, end = c["now"], c["end"]
        items = [PlanItem(**i) for i in state["items"]]
        window = end - start
        if window < timedelta(hours=2):
            return {"fatal_errors": ["There is less than 2 hours left, which is too little to plan milestones."],
                    "milestones": [], "end_eff": end.isoformat()}
        if not items or len(state.get("assignees", [])) != len(items):
            return {"milestones": [], "end_eff": end.isoformat()}
        pinned = {n: _parse_due(it.due_at, ZoneInfo("UTC")) for n, it in enumerate(items) if it.due_at}
        pinned = {n: d for n, d in pinned.items() if d is not None}

        # A pinned deadline after the project end means the leader wants a longer project: extend the end
        # (unless it is fixed, e.g. an event) and keep room for the milestones that come after it.
        buffer = min(max(window * 0.1, timedelta(minutes=15)), timedelta(hours=6))
        if pinned:
            last_n = max(pinned)
            last_pin = max(pinned.values())
            if last_pin > end - buffer:
                tail = TAIL_PER_WEIGHT * sum(it.weight for it in items[last_n + 1:])
                want = _round_up(last_pin + tail + buffer)
                cap = c.get("max_end")
                if (cap is None or want <= cap) and want - end <= MAX_EXTENSION and last_pin <= want:
                    end = want
                    window = end - start
                    buffer = min(max(window * 0.1, timedelta(minutes=15)), timedelta(hours=6))
        lead = min(max(window * 0.05, timedelta(minutes=15)), timedelta(hours=1))    # first task needs a head start
        first, last = start + lead, end - buffer
        project: Project = c["project"]
        # Between two pinned deadlines the milestones are spread by size; pinned ones stay exactly where asked.
        dues: dict[int, datetime] = dict(pinned)
        seg_start, seg = first, []
        for n in range(len(items)):
            seg.append(n)
            if n in pinned or n == len(items) - 1:
                seg_end = pinned[n] if n in pinned else last
                total, cum = sum(items[k].weight for k in seg), 0
                for k in seg:
                    cum += items[k].weight
                    if k not in pinned:
                        dues[k] = _round_up(seg_start + (seg_end - seg_start) * (cum / total))
                seg_start, seg = seg_end, []
        out, prev = [], start
        for n, (item, uid) in enumerate(zip(items, state["assignees"])):
            due = dues[n]
            if n not in pinned:
                due = min(max(due, prev + STEP), end)                  # strictly increasing, never past the end
            prev = due
            out.append(Milestone(id=f"{project.id}-m{n + 1}", title=item.title, domain=item.domain,
                                 assignee_uid=uid, due=due, weight=item.weight).model_dump(mode="json"))
        return {"milestones": out, "end_eff": end.isoformat()}

    # ------------------------------------------------------------------ 4. validate (code)
    def validate(state: PlanState, config) -> dict:
        c = _ctx(config)
        items = [PlanItem(**i) for i in state.get("items", [])]
        errors: list[str] = []
        if len(items) < MIN_MILESTONES:
            errors.append(f"Only {len(items)} milestones. Give at least {MIN_MILESTONES} concrete ones.")
        seen: set[str] = set()
        for it in items:
            if it.domain not in DOMAINS:
                errors.append(f"'{it.title}' has unknown domain '{it.domain}'. Use exactly one of: {', '.join(DOMAINS)}.")
            if it.title.lower() in seen:
                errors.append(f"Duplicate milestone title '{it.title}'. Every milestone needs a distinct title.")
            seen.add(it.title.lower())

        warnings = list(state.get("warnings", []))
        members = {p.uid for p in c["people"]}
        milestones = [Milestone(**m) for m in state.get("milestones", [])]
        end = _parse_due(state.get("end_eff"), ZoneInfo("UTC")) or c["end"]
        cap = c.get("max_end")
        for m in milestones:
            if m.assignee_uid not in members:
                errors.append(f"'{m.title}' is assigned to someone outside the team.")
            if not (c["now"] < m.due <= end):
                tz = _tz(c)
                why = (f" The project end is fixed at {_fmt(cap, tz)} and cannot be extended." if cap is not None
                       and m.due > cap else " The project can be extended up to 60 days, but not further.")
                errors.append(f"The deadline for '{m.title}' ({_fmt(m.due, tz)}) is outside the project window "
                              f"({_fmt(c['now'], tz)} to {_fmt(end, tz)}).{why} Keep every deadline inside it, "
                              f"or leave this milestone's deadline unpinned.")
        pins = [(it, _parse_due(it.due_at, ZoneInfo("UTC"))) for it in items if it.due_at]
        for (a, da), (b2, db) in zip(pins, pins[1:]):
            if da and db and db <= da:
                errors.append(f"'{b2.title}' comes after '{a.title}' in the plan, so its pinned deadline must be later.")
        if milestones:
            loads: dict[str, int] = {}
            for m in milestones:
                loads[m.assignee_uid] = loads.get(m.assignee_uid, 0) + m.weight
            idle = [p.name for p in c["people"] if p.uid not in loads]
            if idle and len(c["people"]) > 1:
                warnings.append(f"No milestone for {', '.join(idle)}.")
            total = sum(loads.values())
            if len(c["people"]) >= 3 and total and max(loads.values()) / total > 0.6:
                warnings.append("One person carries more than 60% of the work.")
        return {"plan_errors": errors, "warnings": warnings}

    def after_validate(state: PlanState) -> str:
        if state.get("fatal_errors"):
            return "done"
        if state.get("plan_errors"):
            return "retry" if state.get("attempt", 0) <= MAX_RETRIES else "done"
        return "review" if review else "done"

    def remember_feedback(state: PlanState) -> dict:
        return {"feedback": list(state.get("plan_errors", []))}

    # ------------------------------------------------------------------ 5. review (pauses for the leader)
    def review_node(state: PlanState) -> dict:
        good = {k: state.get(k) for k in ("items", "assignees", "milestones", "gaps", "warnings", "end_eff")}
        decision = interrupt({"milestones": state["milestones"], "gaps": state.get("gaps", []),
                              "warnings": state.get("warnings", []), "revisions": state.get("revisions", 0)})
        if isinstance(decision, dict) and decision.get("action") == "revise" and str(decision.get("feedback", "")).strip():
            return {"decision": "revise", "good": good,
                    "requirements": list(state.get("requirements", [])) + [str(decision["feedback"]).strip()[:600]],
                    "revisions": state.get("revisions", 0) + 1, "attempt": 0, "feedback": []}
        return {"decision": "confirm", "good": good}

    def after_review(state: PlanState) -> str:
        return "plan" if state.get("decision") == "revise" else "done"

    g = StateGraph(PlanState)
    g.add_node("plan", plan)
    g.add_node("assign", assign)
    g.add_node("schedule", schedule)
    g.add_node("validate", validate)
    g.add_node("feedback", remember_feedback)
    g.add_edge(START, "plan")
    g.add_edge("plan", "assign")
    g.add_edge("assign", "schedule")
    g.add_edge("schedule", "validate")
    targets = {"retry": "feedback", "done": END}
    if review:
        g.add_node("review", review_node)
        targets["review"] = "review"
        g.add_conditional_edges("review", after_review, {"plan": "plan", "done": END})
    g.add_conditional_edges("validate", after_validate, targets)
    g.add_edge("feedback", "plan")
    return g.compile(checkpointer=checkpointer if review else None)


def result_from_state(values: dict, project: Project, original_end: datetime | None = None) -> PlanResult:
    errors = list(values.get("fatal_errors", [])) or list(values.get("plan_errors", []))
    milestones = [Milestone(**m) for m in values.get("milestones", [])]
    end = _parse_due(values.get("end_eff"), ZoneInfo("UTC"))
    return PlanResult(ok=not errors and bool(milestones), milestones=milestones,
                      gaps=[Need(**g) for g in values.get("gaps", [])], warnings=list(values.get("warnings", [])),
                      errors=errors, attempts=values.get("attempt", 0), notes=values.get("notes", ""), end=end,
                      extended=bool(end and original_end and end > original_end))


def plan_project(project: Project, people: list[Person], brief: str | None = None, *,
                 now: datetime, end: datetime, llm: LLMFn | None = None, matcher: Matcher | None = None,
                 requirements: list[str] | None = None, max_end: datetime | None = None,
                 tz: str = DEFAULT_TZ) -> PlanResult:
    """One-shot planning without a review step. `people` = the team members. `end` = event end or deadline."""
    if not people:
        return PlanResult(ok=False, errors=["The project has no team members to plan for."])
    brief = (brief or project.description or project.title).strip()
    graph = build_graph(llm, matcher)
    final = graph.invoke(
        {"brief": brief, "requirements": list(requirements or []), "feedback": [], "attempt": 0, "revisions": 0,
         "warnings": [], "gaps": []},
        config={"configurable": {"project": project, "people": people, "now": now, "end": end,
                                 "max_end": max_end, "tz": tz}})
    return result_from_state(final, project, end)


def apply_plan(project: Project, result: PlanResult) -> None:
    """Puts a successful plan on the project. Refuses to overwrite work that has already been marked."""
    from ..reputation.service import RuleError
    if not result.ok:
        raise RuleError("This plan has errors, so it cannot be applied.")
    if any(m.status != "pending" for m in project.milestones):
        raise RuleError("Some tasks are already marked, so the plan cannot be replaced.")
    project.milestones = list(result.milestones)
    if result.end is not None:
        project.deadline = result.end
    for gap in result.gaps:
        if not any(n.domain == gap.domain and not n.filled_by for n in project.needs):
            project.needs.append(gap)

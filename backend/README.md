# SkillSwap backend

## Module 1: LLM wrapper (`app/llm`)

Every AI call in the project goes through one place:

```python
from app.llm import call_llm, call_llm_json

reply = call_llm("Summarise this in one line: ...")          # reply.text, reply.provider
data  = call_llm_json("Extract skills from: ...", MySchema)  # validated pydantic object
```

Fallback order: **Groq -> Gemini (every key, every model) -> Ollama**. A rate-limited
endpoint cools down and is skipped; a bad key is switched off; validated answers are
cached on disk (`.llm_cache.sqlite`) so seeded demo data costs nothing.

### Setup (your laptop)

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your keys into .env
python -m app.llm.check             # pings every key/model
python -m app.llm.check --models    # lists valid model names per key
python -m pytest                    # offline tests (193)
```

### Keys

- **Groq:** `GROQ_API_KEY1..GROQ_API_KEY30` (or a single `GROQ_API_KEY`). Keys are tried in order and rotated
  each call (`LLM_ROTATE_KEYS=1`), so load spreads out; a key that hits a limit cools down and the next one
  takes over, then Gemini, then Ollama. Groq limits are per *account/organization*, not per key, so extra keys
  only help if they belong to different accounts (e.g. one per teammate). Set `GROQ_MODELS=qwen/qwen3.8-27b`.
- **Gemini:** `GEMINI_API_KEY1..4`. Google limits by *project*, not key, so create each
  key in a **separate Google Cloud project** (one per teammate works well).
- **Ollama (offline backup):** install Ollama, run `ollama pull llama3.2:3b`.
- Never commit `.env`. It is already in `.gitignore`.

## Module 2: Resume to skill profile (`app/resume`)

```python
from app.resume import parse_resume_pdf, parse_resume_text

profile = parse_resume_pdf(pdf_bytes)     # or parse_resume_text(pasted_text)
profile.skills        # [Skill(name="Flask", level="intermediate", projects=4, evidence="...")]
profile.domains       # ["Backend", "DevOps & Cloud"]  (from a fixed list, used by people search)
profile.skill_summary # clean text that module 3 turns into an embedding
profile.warnings      # e.g. "Dropped 'Kubernetes': not found in the resume text."
```

How it stays trustworthy: the LLM reads the resume, then plain Python checks it. Every skill
must appear in the resume text (by name or by a quoted line), so skills cannot be invented;
names are normalised (`reactjs` -> `React`); domains are mapped to the fixed list; and the
resume is passed to the model as data, with an instruction to ignore commands inside it.

Errors you can show the student as-is: scanned/image PDF (offer the paste box), password-
protected PDF, file over 5 MB, corrupt file, text too short.

Try it on real resumes (needs keys in `.env`):

```bash
python scripts/make_sample_resumes.py           # 3 fictional PDFs -> samples/
python -m app.resume.cli samples/*.pdf          # readable summary per resume
python -m app.resume.cli my_resume.pdf --json   # full JSON
python -m app.resume.cli --text pasted.txt      # test the paste-text path
```

## Module 3: Matching, domain search and domain reputation

```python
from app.matching import Matcher
matcher = Matcher()
matcher.recommend_people(project, need, people, owner)        # who fits this open role
matcher.recommend_projects(person, projects, people_by_uid)   # which open roles fit this person
matcher.search_people(people, domains=["Mobile (iOS)", "UI/UX Design"], match="any")  # hackathon search
```

`score = fit - overlap with the owner + reputation in that domain`. Fit = skill coverage (weighted by
level) + semantic similarity (embeddings) + domain match. The "why this match" line is built from the
numbers, so it cannot invent anything and uses no API quota.

Try it with no keys and no resumes (fictional people): `python -m app.matching.demo`

Embeddings: the real model (`all-MiniLM-L6-v2`, ~90 MB) is used automatically when installed:
`pip install -r requirements-embeddings.txt`, then `python -m app.matching.check`. Without it a built-in
lighter embedder is used, so nothing breaks. Do the install once on the Mac while online.

### Domain reputation (`app/reputation`)

- Points and reliability are tracked **per domain** (Backend, Frontend, UI/UX ...), plus overall.
- Levels: Newcomer, Contributor (20+ points), Trusted (80+ points and 75%+ reliability).
- Reliability shows "New" until 3 reviewed tasks. Recent work counts more, so people recover.
- The **project leader marks** teammates' tasks (done / missed / abandoned / excused). Nobody marks their own.
- A leader's own overdue task counts as missed only when **all teammates report it**.
- Excused emergencies and proper handovers never lower reliability.
- **Leader tag**: the owner leads by default and can transfer it. It disappears when the project is
  completed, but "Led: <project>" stays in the profile history for good.

All rules are in `app/reputation/service.py` and are covered by tests.

### Gender and avatar

The student optionally picks a gender (female / male / other / prefer not to say) on the profile and
the avatar is chosen automatically (`app/avatars.py`, pictures in `static/avatars/`). Gender is used
only for the avatar and never for matching or search. The same three PNGs go into the Xcode asset
catalog under the names `avatar_female`, `avatar_male`, `avatar_neutral`.

## Module 4: AI milestone planner with leader review (`app/planner`)

A LangGraph graph: **plan** (the only LLM call) -> **assign** (code, uses the matching engine, avoids
overloading anyone; coordination and demo work goes to the leader) -> **schedule** (code, deadlines spread by
milestone size, slack before the end) -> **validate** (code) -> **review** (the graph pauses for the leader).
Problems the model can fix (too few milestones, unknown domain, duplicate titles) loop back to *plan* with
the problems as feedback, at most 2 times.

Human in the loop (`PlanSession`, uses LangGraph's `interrupt`):

```python
from app.planner import PlanSession, apply_plan
session = PlanSession()
draft = session.start(project, team, now=now, end=event_end, extra="optional note from the leader")
draft = session.revise(draft.session_id, project, team, now=now, end=event_end, feedback="drop the promo video")
result = session.confirm(draft.session_id, project, team, now=now, end=event_end,
                         edits=[{"id": "skillswap-m2", "assignee_uid": "meera"}])   # optional direct edits
apply_plan(project, result)       # nothing touches the project until this line
```

- The leader can ask in plain words for content changes ("add a testing task"), **owner changes**
  ("give the API to Meera") and **deadline changes** ("make the SwiftUI task due tomorrow 7 am; the rest follow").
  Pinned owners/deadlines are kept exactly, milestones before/after are re-spread around them, and pins stay
  through later revisions. Times are read in the display time zone (`tz`, default `Asia/Kolkata`).
- Every revision returns `draft.notes` (what the planner changed, or why it could not) and `draft.changed`.
  A request that cannot be done is never silently ignored: the model is told why and explains it in `notes`;
  if nothing changed the draft says so.
- Deadlines after the project end **extend the project** (up to 60 days): later milestones are scheduled after the
  pinned one, `draft.end` / `draft.extended` show the new end, and `apply_plan` saves it as `Project.deadline`.
  For a fixed end (the hackathon's 24 Oct 5pm), pass `max_end=end`; later deadlines are then refused with a
  reason (demo: `--fixed-end`).
- Up to 3 revisions. A revision that fails validation leaves the previous draft usable (`revision_failed`).
- If the LLM is down mid-revision the call raises; calling `revise` again with the same text continues it.
- Paused drafts are saved in `.planner_sessions.sqlite`, so a server restart does not lose them.
- Only the project leader may plan (`require_leader`). Direct edits (owner, deadline, title) are validated.
- One-shot planning with no review is still available: `plan_project(...)`.
- Domains nobody on the team has are returned as `gaps` (suggested open roles).

Try it: `python -m app.planner.demo` (canned plan, no keys), `--live` (real LLM), `--review` (you are the
leader: review the draft, ask for changes, confirm; combine with `--live`).

## Troubleshooting

Truncated JSON ("unterminated JSON in model reply"): Gemini 2.5 Flash used to spend its output tokens on hidden
"thinking" and cut long answers off. Thinking is now off for JSON calls (set `GEMINI_THINKING_BUDGET` in `.env` to
change it) and a truncated reply is retried with a bigger token budget (up to 8192).

Rate limits: free tiers allow only a few requests per minute. If every key is briefly limited,
the app waits up to `LLM_MAX_WAIT` seconds (default 25) for the first one to recover instead of
failing. More Gemini keys (each in its own Google Cloud project) means fewer waits.
Set `GROQ_MODELS` to only the model(s) your key can actually use to avoid "model not found" noise.

If a `FAIL ... model not found` line appears, run `--models` and put a valid name in
`GROQ_MODELS` / `GEMINI_MODELS` (comma separated, best first).

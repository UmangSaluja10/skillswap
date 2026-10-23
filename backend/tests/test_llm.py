import json

import pytest
import requests
from pydantic import BaseModel

from app.llm import LLMClient, LLMInvalidOutput, LLMUnavailable, extract_json
from app.llm import providers
from app.llm.cache import DiskCache
from app.llm.config import build_endpoints, load_dotenv


# ----------------------------------------------------------------- helpers
class Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.text = json.dumps(self._body)
        self.headers = headers or {}

    def json(self):
        return self._body


def groq_ok(text):
    return Resp(200, {"choices": [{"message": {"content": text}}]})


def gemini_ok(text):
    return Resp(200, {"candidates": [{"content": {"parts": [{"text": text}]}}]})


def ollama_ok(text):
    return Resp(200, {"message": {"content": text}})


def err(status, message="x", headers=None):
    return Resp(status, {"error": {"message": message}}, headers)


class Fake:
    """Replaces requests.post. Responses are scripted per API key."""

    def __init__(self):
        self.script = {}
        self.calls = []

    def add(self, key, *responses):
        self.script[key] = list(responses)

    def count(self, key):
        return sum(1 for k, _, _ in self.calls if k == key)

    def post(self, url, headers=None, json=None, timeout=None):
        headers = headers or {}
        key = headers.get("x-goog-api-key") or headers.get("Authorization", "").replace("Bearer ", "") or "ollama"
        self.calls.append((key, url, json))
        queue = self.script.get(key)
        assert queue, f"unscripted call for {key}"
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


ENV = {
    "LLM_PROVIDER_ORDER": "groq,gemini",
    "GROQ_API_KEY": "groq-key",
    "GROQ_MODELS": "gm",
    "GEMINI_API_KEY1": "gem1",
    "GEMINI_API_KEY2": "gem2",
    "GEMINI_API_KEY3": "gem3",
    "GEMINI_MODELS": "fm",
}


@pytest.fixture
def fake(monkeypatch):
    f = Fake()
    monkeypatch.setattr(providers.requests, "post", f.post)
    return f


@pytest.fixture
def clock():
    return Clock()


def client(clock=None, cache=None, env=None):
    return LLMClient(build_endpoints(env or ENV), cache or DiskCache(None), clock=clock or Clock())


class Person(BaseModel):
    name: str
    level: int


# ------------------------------------------------------------------ config
def test_endpoints_follow_order_models_then_keys():
    env = dict(ENV, GEMINI_MODELS="m1,m2")
    labels = [e.label for e in build_endpoints(env)]
    assert labels == [
        "groq:key:gm",
        "gemini:key1:m1", "gemini:key2:m1", "gemini:key3:m1",
        "gemini:key1:m2", "gemini:key2:m2", "gemini:key3:m2",
    ]


def test_provider_order_is_configurable():
    env = dict(ENV, LLM_PROVIDER_ORDER="gemini,groq")
    assert build_endpoints(env)[0].provider == "gemini"


def test_placeholder_and_blank_keys_are_ignored():
    env = dict(ENV, GEMINI_API_KEY1="your_key_here", GEMINI_API_KEY2="", GEMINI_API_KEY3="real")
    labels = [e.label for e in build_endpoints(env) if e.provider == "gemini"]
    assert labels == ["gemini:key3:fm"]


def test_dotenv_does_not_override_real_environment(tmp_path):
    f = tmp_path / ".env"
    f.write_text("A=from_file\nB='quoted'\n# c\n")
    env = {"A": "from_env"}
    load_dotenv(f, env)
    assert env == {"A": "from_env", "B": "quoted"}


# ---------------------------------------------------------------- fallback
def test_uses_first_provider_when_healthy(fake):
    fake.add("groq-key", groq_ok("hello"))
    res = client().call_llm("hi")
    assert (res.text, res.provider) == ("hello", "groq")
    assert fake.count("gem1") == 0


def test_rate_limited_groq_falls_back_to_gemini_and_is_skipped_next_time(fake):
    fake.add("groq-key", err(429, "slow down"))
    fake.add("gem1", gemini_ok("from gemini"))
    c = client()
    assert c.call_llm("a").provider == "gemini"
    assert c.call_llm("b").provider == "gemini"
    assert fake.count("groq-key") == 1  # second call skipped the cooling endpoint


def test_exhausted_gemini_key_rolls_to_next_key(fake):
    fake.add("groq-key", err(429))
    fake.add("gem1", err(429, "Quota exceeded GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
    fake.add("gem2", gemini_ok("key2 answered"))
    res = client().call_llm("hi")
    assert res.label == "gemini:key2:fm"
    assert ("gemini:key1:fm", "rate_limit") in res.failed_attempts


def test_daily_quota_cooldown_is_long_and_expires(fake, clock):
    fake.add("groq-key", err(429, "per day limit"))
    fake.add("gem1", gemini_ok("ok"))
    c = client(clock=clock)
    c.call_llm("a")
    clock.t += 1800
    c.call_llm("b")
    assert fake.count("groq-key") == 1  # still cooling after 30 min
    clock.t += 1900
    c.call_llm("c")
    assert fake.count("groq-key") == 2  # retried after the hour


def test_retry_after_header_is_respected(fake, clock):
    fake.add("groq-key", err(429, headers={"retry-after": "10"}), groq_ok("back"))
    fake.add("gem1", gemini_ok("fallback"))
    c = client(clock=clock)
    assert c.call_llm("a").provider == "gemini"
    clock.t += 11
    assert c.call_llm("b").provider == "groq"


def test_bad_key_is_disabled_for_good(fake, clock):
    fake.add("groq-key", err(401, "invalid api key"))
    fake.add("gem1", gemini_ok("ok"))
    c = client(clock=clock)
    c.call_llm("a")
    clock.t += 10_000
    c.call_llm("b")
    assert fake.count("groq-key") == 1
    assert "disabled" in [r for r in c.status() if r["label"] == "groq:key:gm"][0]["state"]


def test_gemini_invalid_key_400_counts_as_auth(fake):
    fake.add("groq-key", err(429))
    fake.add("gem1", err(400, "API key not valid. Please pass a valid API key."))
    fake.add("gem2", gemini_ok("ok"))
    c = client()
    assert c.call_llm("a").label == "gemini:key2:fm"
    assert "disabled" in [r for r in c.status() if r["label"] == "gemini:key1:fm"][0]["state"]


def test_unknown_model_disables_only_that_endpoint(fake):
    fake.add("groq-key", err(404, "model not found"))
    fake.add("gem1", gemini_ok("ok"))
    assert client().call_llm("a").provider == "gemini"


def test_network_errors_and_timeouts_fall_back(fake):
    fake.add("groq-key", requests.exceptions.ConnectionError("down"))
    fake.add("gem1", requests.exceptions.Timeout("slow"))
    fake.add("gem2", gemini_ok("ok"))
    assert client().call_llm("a").label == "gemini:key2:fm"


def test_empty_reply_moves_on(fake):
    fake.add("groq-key", groq_ok("   "))
    fake.add("gem1", gemini_ok("real answer"))
    assert client().call_llm("a").text == "real answer"


def test_all_failing_raises_with_retry_hint(fake, clock):
    fake.add("groq-key", err(429, headers={"retry-after": "30"}))
    for k in ("gem1", "gem2", "gem3"):
        fake.add(k, err(429, headers={"retry-after": "45"}))
    c = client(clock=clock)
    with pytest.raises(LLMUnavailable) as exc:
        c.call_llm("a")
    assert len(exc.value.attempts) == 4
    calls_before = len(fake.calls)
    with pytest.raises(LLMUnavailable) as exc2:
        c.call_llm("b")
    assert len(fake.calls) == calls_before  # everything cooling: no network
    assert exc2.value.retry_in == pytest.approx(30)


def test_no_endpoints_configured_raises():
    with pytest.raises(LLMUnavailable):
        LLMClient([]).call_llm("hi")


def test_ollama_is_used_last(fake):
    env = dict(ENV, LLM_PROVIDER_ORDER="groq,gemini,ollama", OLLAMA_MODEL="tiny")
    fake.add("groq-key", err(500))
    for k in ("gem1", "gem2", "gem3"):
        fake.add(k, err(500))
    fake.add("ollama", ollama_ok("local answer"))
    res = client(env=env).call_llm("hi")
    assert (res.provider, res.text) == ("ollama", "local answer")


# ------------------------------------------------------------ request shape
def test_gemini_request_shape(fake):
    fake.add("groq-key", err(429))
    fake.add("gem1", gemini_ok("{}"))
    client().call_llm("prompt", system="be brief", json_mode=True, max_tokens=99)
    key, url, body = fake.calls[-1]
    assert key == "gem1" and "gem1" not in url  # key travels in a header, never the URL
    assert url.endswith("/models/fm:generateContent")
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["maxOutputTokens"] == 99
    assert "be brief" in body["systemInstruction"]["parts"][0]["text"]


def test_groq_json_mode_request_shape(fake):
    fake.add("groq-key", groq_ok("{}"))
    client().call_llm("prompt", json_mode=True)
    body = fake.calls[-1][2]
    assert body["response_format"] == {"type": "json_object"}
    assert "JSON" in body["messages"][0]["content"]


# ----------------------------------------------------- brief rate limits
GROQ_ONLY = {"LLM_PROVIDER_ORDER": "groq", "GROQ_API_KEY": "groq-key", "GROQ_MODELS": "gm"}


def waiting_client(clock, max_wait):
    slept = []

    def fake_sleep(seconds):
        slept.append(seconds)
        clock.t += seconds

    c = LLMClient(build_endpoints(GROQ_ONLY), DiskCache(None), clock=clock, max_wait=max_wait, sleep=fake_sleep)
    return c, slept


def test_short_rate_limit_is_waited_out_instead_of_failing(fake, clock):
    fake.add("groq-key", err(429, headers={"retry-after": "5"}), groq_ok("worth the wait"))
    c, slept = waiting_client(clock, max_wait=10)
    res = c.call_llm("hi")
    assert res.text == "worth the wait"
    assert len(slept) == 1 and 5 <= slept[0] < 6
    assert fake.count("groq-key") == 2


def test_long_rate_limit_still_fails_fast(fake, clock):
    fake.add("groq-key", err(429, headers={"retry-after": "60"}))
    c, slept = waiting_client(clock, max_wait=10)
    with pytest.raises(LLMUnavailable) as exc:
        c.call_llm("hi")
    assert slept == [] and exc.value.retry_in == pytest.approx(60)


def test_waiting_is_capped_across_repeated_limits(fake, clock):
    fake.add("groq-key", err(429, headers={"retry-after": "4"}))  # limited again after every wait
    c, slept = waiting_client(clock, max_wait=10)
    with pytest.raises(LLMUnavailable):
        c.call_llm("hi")
    assert sum(slept) <= 10 and len(slept) == 2


# --------------------------------------------------------------------- JSON
def test_reasoning_blocks_are_ignored():
    assert extract_json('<think>maybe {"x": 0} is it?</think>\n{"a": 1}') == {"a": 1}


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here you go: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    assert extract_json("[1, 2, 3]") == [1, 2, 3]
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_json_validation_failure_is_retried_with_the_error(fake):
    fake.add("groq-key", groq_ok('{"name": 3}'), groq_ok('```json\n{"name": "Aditi", "level": 2}\n```'))
    out = client().call_llm_json("extract", Person)
    assert out == Person(name="Aditi", level=2)
    assert fake.count("groq-key") == 2
    assert "rejected" in fake.calls[-1][2]["messages"][-1]["content"]


def test_json_gives_up_after_retries(fake):
    fake.add("groq-key", groq_ok("not json at all"))
    with pytest.raises(LLMInvalidOutput):
        client().call_llm_json("extract", Person, retries=1)
    assert fake.count("groq-key") == 2


def test_schema_is_added_to_the_prompt(fake):
    fake.add("groq-key", groq_ok('{"name": "A", "level": 1}'))
    client().call_llm_json("extract", Person)
    assert '"level"' in fake.calls[0][2]["messages"][-1]["content"]


# -------------------------------------------------------------------- cache
def test_cache_avoids_second_network_call(fake, tmp_path):
    fake.add("groq-key", groq_ok("cached me"))
    c = client(cache=DiskCache(tmp_path / "c.sqlite"))
    first = c.call_llm("same prompt")
    second = c.call_llm("same prompt")
    assert fake.count("groq-key") == 1
    assert second.cached and second.text == first.text


def test_cache_is_skipped_for_creative_temperature(fake, tmp_path):
    fake.add("groq-key", groq_ok("a"), groq_ok("b"))
    c = client(cache=DiskCache(tmp_path / "c.sqlite"))
    c.call_llm("p", temperature=0.9)
    assert c.call_llm("p", temperature=0.9).text == "b"


def test_only_validated_json_is_cached(fake, tmp_path):
    cache = DiskCache(tmp_path / "c.sqlite")
    fake.add("groq-key", groq_ok("garbage"))
    c = client(cache=cache)
    with pytest.raises(LLMInvalidOutput):
        c.call_llm_json("x", Person, retries=0)
    fake.add("groq-key", groq_ok('{"name": "B", "level": 3}'))
    assert c.call_llm_json("x", Person).name == "B"  # bad answer was not remembered
    calls = len(fake.calls)
    assert c.call_llm_json("x", Person).name == "B"
    assert len(fake.calls) == calls  # now served from cache


# ------------------------------------------------------------------- status
def test_status_reports_ready_and_cooling(fake, clock):
    fake.add("groq-key", err(429, headers={"retry-after": "30"}))
    fake.add("gem1", gemini_ok("ok"))
    c = client(clock=clock)
    c.call_llm("a")
    states = {r["label"]: r["state"] for r in c.status()}
    assert states["groq:key:gm"].startswith("cooling")
    assert states["gemini:key1:fm"] == "ready"


# ------------------------------------------------------------------ Gemini thinking and truncated JSON
def _gemini_config(fake_calls):
    return fake_calls[0][2]["generationConfig"]


def test_gemini_flash_json_calls_turn_thinking_off_by_default(fake):
    fake.add("gem1", gemini_ok("{}"))
    client(env={"LLM_PROVIDER_ORDER": "gemini", "GEMINI_API_KEY1": "gem1", "GEMINI_MODELS": "gemini-2.5-flash"}) \
        .call_llm("x", json_mode=True)
    assert _gemini_config(fake.calls)["thinkingConfig"] == {"thinkingBudget": 0}


def test_gemini_thinking_stays_on_for_plain_text_pro_and_explicit_setting(fake):
    env = {"LLM_PROVIDER_ORDER": "gemini", "GEMINI_API_KEY1": "gem1", "GEMINI_MODELS": "gemini-2.5-flash"}
    fake.add("gem1", gemini_ok("hi"))
    client(env=env).call_llm("x", json_mode=False)
    assert "thinkingConfig" not in _gemini_config(fake.calls)

    fake.calls.clear(); fake.add("gem1", gemini_ok("{}"))
    client(env=dict(env, GEMINI_MODELS="gemini-2.5-pro")).call_llm("x", json_mode=True)
    assert "thinkingConfig" not in _gemini_config(fake.calls)

    fake.calls.clear(); fake.add("gem1", gemini_ok("{}"))
    client(env=dict(env, GEMINI_THINKING_BUDGET="512")).call_llm("x", json_mode=True)
    assert _gemini_config(fake.calls)["thinkingConfig"] == {"thinkingBudget": 512}


def test_truncated_json_is_retried_with_a_bigger_token_budget(fake):
    fake.add("groq-key", groq_ok('{"name": "Ana", "lev'), groq_ok('{"name": "Ana", "level": 3}'))
    c = client(env={"LLM_PROVIDER_ORDER": "groq", "GROQ_API_KEY": "groq-key", "GROQ_MODELS": "gm"})
    out = c.call_llm_json("who", Person, max_tokens=1000)
    assert out == Person(name="Ana", level=3)
    sent = [call[2]["max_tokens"] for call in fake.calls]
    assert sent == [1000, 2000]


def test_wrong_but_complete_json_does_not_raise_the_token_budget(fake):
    fake.add("groq-key", groq_ok('{"name": "Ana"}'), groq_ok('{"name": "Ana", "level": 3}'))
    c = client(env={"LLM_PROVIDER_ORDER": "groq", "GROQ_API_KEY": "groq-key", "GROQ_MODELS": "gm"})
    c.call_llm_json("who", Person, max_tokens=1000)
    assert [call[2]["max_tokens"] for call in fake.calls] == [1000, 1000]


def test_token_budget_growth_is_capped(fake):
    from app.llm.client import LLMInvalidOutput, MAX_RETRY_TOKENS
    fake.add("groq-key", groq_ok('{"name": "A'))
    c = client(env={"LLM_PROVIDER_ORDER": "groq", "GROQ_API_KEY": "groq-key", "GROQ_MODELS": "gm"})
    with pytest.raises(LLMInvalidOutput):
        c.call_llm_json("who", Person, max_tokens=5000, retries=3)
    assert max(call[2]["max_tokens"] for call in fake.calls) == MAX_RETRY_TOKENS


# ------------------------------------------------------------------ many keys and rotation
def many(n, provider="groq", **extra):
    key = "GROQ_API_KEY" if provider == "groq" else "GEMINI_API_KEY"
    env = {"LLM_PROVIDER_ORDER": provider, f"{provider.upper()}_MODELS": "m"}
    env.update({f"{key}{i}": f"k{i}" for i in range(1, n + 1)})
    env.update(extra)
    return env


def rotating(env, clock=None):
    return LLMClient(build_endpoints(env), DiskCache(None), clock=clock or Clock(), rotate=True)


def used(fake):
    return [call[0] for call in fake.calls]


def test_up_to_thirty_numbered_keys_are_collected():
    env = many(12)
    assert [e.label for e in build_endpoints(env)][-1] == "groq:key12:m"
    assert len(build_endpoints(many(30))) == 30


def test_keys_take_turns_round_robin(fake):
    for i in range(1, 4):
        fake.add(f"k{i}", groq_ok("ok"))
    c = rotating(many(3))
    for _ in range(7):
        c.call_llm("x", use_cache=False)
    assert used(fake) == ["k1", "k2", "k3", "k1", "k2", "k3", "k1"]


def test_without_rotation_the_first_key_is_used_until_it_fails(fake):
    fake.add("k1", groq_ok("ok")); fake.add("k2", groq_ok("ok"))
    c = client(env=many(2))
    for _ in range(3):
        c.call_llm("x", use_cache=False)
    assert used(fake) == ["k1", "k1", "k1"]


def test_rate_limited_key_is_skipped_and_the_next_one_answers(fake, clock):
    fake.add("k1", err(429, "rate limit", {"retry-after": "30"}))
    fake.add("k2", groq_ok("from 2")); fake.add("k3", groq_ok("from 3"))
    c = rotating(many(3), clock)
    assert c.call_llm("x", use_cache=False).label == "groq:key2:m"          # k1 failed, k2 answered
    assert c.call_llm("x", use_cache=False).label == "groq:key3:m"          # k1 cooling, rotation moves on
    assert c.call_llm("x", use_cache=False).label == "groq:key2:m"          # wraps around, still skipping k1
    assert used(fake).count("k1") == 1                                       # k1 was not hammered while cooling


def test_cooled_down_key_rejoins_the_rotation(fake, clock):
    fake.add("k1", err(429, "rate limit", {"retry-after": "30"}), groq_ok("back"))
    fake.add("k2", groq_ok("2"))
    c = rotating(many(2), clock)
    c.call_llm("x", use_cache=False)          # k1 limited, k2 answers
    clock.t += 31
    labels = [c.call_llm("x", use_cache=False).label for _ in range(2)]
    assert "groq:key1:m" in labels


def test_bad_key_is_switched_off_for_good_and_others_continue(fake):
    fake.add("k1", err(401, "invalid api key")); fake.add("k2", groq_ok("ok"))
    c = rotating(many(2))
    for _ in range(3):
        assert c.call_llm("x", use_cache=False).label == "groq:key2:m"
    assert used(fake).count("k1") == 1


def test_all_groq_keys_limited_falls_through_to_gemini(fake, clock):
    env = many(3, LLM_PROVIDER_ORDER="groq,gemini", GEMINI_API_KEY1="g1", GEMINI_MODELS="fm")
    for i in range(1, 4):
        fake.add(f"k{i}", err(429, "rate limit", {"retry-after": "60"}))
    fake.add("g1", gemini_ok("from gemini"))
    c = rotating(env, clock)
    r = c.call_llm("x", use_cache=False)
    assert r.label == "gemini:key1:fm" and [a[1] for a in r.failed_attempts] == ["rate_limit"] * 3


def test_rotation_keeps_provider_and_model_order(fake):
    env = many(2, GROQ_MODELS="m1,m2")
    for k in ("k1", "k2"):
        fake.add(k, groq_ok("ok"))
    c = rotating(env)
    c._advance(c.endpoints[0])                                   # pretend key1/m1 just answered
    labels = [e.label for e in c._ordered()]
    assert labels == ["groq:key2:m1", "groq:key1:m1", "groq:key1:m2", "groq:key2:m2"]


def test_rotate_is_on_by_default_in_from_env_and_can_be_turned_off():
    assert LLMClient.from_env(many(2), load_env_file=False).rotate is True
    assert LLMClient.from_env(many(2, LLM_ROTATE_KEYS="0"), load_env_file=False).rotate is False

import asyncio

import pytest

from spendcap import Budget, BudgetExceededError, Meter
from spendcap.wrappers import extract_usage

from conftest import (
    FakeAnthropic,
    FakeAnthropicUsage,
    FakeOpenAI,
    FakeOpenAIUsage,
    FakePromptDetails,
)


def test_wrap_anthropic_meters_calls():
    meter = Meter()
    client = meter.wrap(FakeAnthropic())
    resp = client.messages.create(
        model="claude-haiku-4-5", messages=[{"role": "user", "content": "hi"}]
    )
    assert resp.content == "ok"  # response passes through unwrapped
    assert meter.calls == 1
    assert meter.spent == pytest.approx((1000 * 1.00 + 200 * 5.00) / 1_000_000)


def test_wrap_openai_meters_calls_with_cache():
    meter = Meter()
    fake = FakeOpenAI()
    fake.next_usage = FakeOpenAIUsage(
        prompt_tokens=1000,
        completion_tokens=100,
        prompt_tokens_details=FakePromptDetails(cached_tokens=600),
    )
    client = meter.wrap(fake)
    client.chat.completions.create(model="gpt-5.4-mini", messages=[])
    rec = meter.records[0]
    assert rec.input_tokens == 400  # prompt minus cached
    assert rec.cached_input_tokens == 600


def test_wrap_anthropic_cache_fields():
    meter = Meter()
    fake = FakeAnthropic()
    fake.next_usage = FakeAnthropicUsage(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=2000,
        cache_creation_input_tokens=500,
    )
    client = meter.wrap(fake)
    client.messages.create(model="claude-haiku-4-5", messages=[])
    rec = meter.records[0]
    # Anthropic input_tokens already excludes cache traffic; no subtraction.
    assert rec.input_tokens == 100
    assert rec.cached_input_tokens == 2000
    assert rec.cache_write_tokens == 500


def test_circuit_breaker_stops_next_call():
    meter = Meter(budget=Budget(usd=0.001))
    fake = FakeAnthropic()
    client = meter.wrap(fake)
    client.messages.create(model="claude-haiku-4-5", messages=[])  # goes through
    assert fake.calls_made == 1
    with pytest.raises(BudgetExceededError):
        client.messages.create(model="claude-haiku-4-5", messages=[])
    assert fake.calls_made == 1  # second call never reached the API


def test_wrapped_response_not_lost_on_crossing_call():
    # The call that crosses the cap still returns its response;
    # only the NEXT call raises.
    meter = Meter(budget=Budget(usd=0.0001))
    client = meter.wrap(FakeAnthropic())
    resp = client.messages.create(model="claude-haiku-4-5", messages=[])
    assert resp is not None


def test_non_llm_calls_pass_through_unmetered():
    meter = Meter()

    class Thing:
        def ping(self):
            return "pong"

    proxied = meter.wrap(Thing())
    assert proxied.ping() == "pong"
    assert meter.calls == 0


def test_primitive_attributes_unwrapped():
    meter = Meter()
    client = meter.wrap(FakeAnthropic())
    assert client.api_key == "sk-fake"
    assert isinstance(client.api_key, str)


def test_wrapped_dunder_exposes_original():
    meter = Meter()
    fake = FakeAnthropic()
    assert meter.wrap(fake).__wrapped__ is fake


def test_async_client_metered():
    meter = Meter()

    class AsyncMessages:
        async def create(self, *, model, messages):
            return {
                "model": model,
                "usage": {"input_tokens": 500, "output_tokens": 100},
            }

    class AsyncClient:
        messages = AsyncMessages()

    client = meter.wrap(AsyncClient())

    async def main():
        return await client.messages.create(model="claude-haiku-4-5", messages=[])

    resp = asyncio.run(main())
    assert resp["usage"]["input_tokens"] == 500
    assert meter.calls == 1


def test_dict_shaped_response():
    usage = extract_usage(
        {"model": "gpt-5.5", "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        {},
    )
    assert usage == dict(
        model="gpt-5.5",
        input_tokens=10,
        output_tokens=5,
        cached_input_tokens=0,
        cache_write_tokens=0,
    )


def test_stream_without_usage_warns():
    meter = Meter()

    class Streamy:
        def create(self, *, model, stream=False, **kw):
            return iter([])  # no usage anywhere

    client = meter.wrap(Streamy())
    with pytest.warns(UserWarning, match="streaming"):
        client.create(model="claude-haiku-4-5", stream=True)
    assert meter.calls == 0

import pytest

import spendcap
from spendcap.pricing import _normalize


def test_exact_lookup():
    p = spendcap.get_price("claude-haiku-4-5")
    assert p is not None
    assert p.input_per_m == 1.00
    assert p.output_per_m == 5.00


def test_dated_model_id_resolves():
    assert spendcap.resolve_model("claude-haiku-4-5-20251001") == "claude-haiku-4-5"
    assert spendcap.resolve_model("gpt-5.4-mini-2026-01-15") == "gpt-5.4-mini"


def test_provider_prefix_and_case():
    assert spendcap.resolve_model("Anthropic/Claude-Sonnet-5") == "claude-sonnet-5"
    assert spendcap.resolve_model("models/gemini-3.5-flash") == "gemini-3.5-flash"


def test_latest_suffix():
    assert spendcap.resolve_model("claude-sonnet-5-latest") == "claude-sonnet-5"


def test_longest_prefix_wins():
    # gpt-5.4-mini-XYZ should hit gpt-5.4-mini, not gpt-5.4
    assert spendcap.resolve_model("gpt-5.4-mini-preview") == "gpt-5.4-mini"
    assert spendcap.resolve_model("gemini-3-flash-lite-001") == "gemini-3-flash-lite"


def test_no_cross_family_guessing():
    # 'gpt-5.6' alone must NOT resolve to sol/terra/luna
    assert spendcap.resolve_model("gpt-5.6") is None
    assert spendcap.get_price("totally-unknown-model") is None


def test_cost_math_exact():
    p = spendcap.get_price("claude-haiku-4-5")  # $1 in / $5 out / $0.10 cached
    cost = p.cost(input_tokens=1_000_000, output_tokens=200_000)
    assert cost == pytest.approx(1.00 + 1.00)
    cost = p.cost(cached_input_tokens=1_000_000)
    assert cost == pytest.approx(0.10)
    # cache writes bill at 1.25x input
    cost = p.cost(cache_write_tokens=1_000_000)
    assert cost == pytest.approx(1.25)


def test_register_and_load(tmp_path):
    spendcap.register_model("my-local-model", 0.0, 0.0)
    assert spendcap.get_price("my-local-model").input_per_m == 0.0

    pricing_file = tmp_path / "prices.json"
    pricing_file.write_text(
        '{"groq-llama-4-70b": {"input": 0.59, "output": 0.79}}'
    )
    assert spendcap.load_pricing(str(pricing_file)) == 1
    assert spendcap.get_price("groq-llama-4-70b").output_per_m == 0.79


def test_normalize():
    assert _normalize("  OpenAI/GPT-5.5-latest ") == "gpt-5.5"

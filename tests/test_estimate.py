import pytest

import spendcap
from spendcap import UnknownModelError, compare_models, estimate_loop


def brute_force_input(turns, s, n, o):
    total = 0
    history = 0
    for _ in range(turns):
        total += s + history + n
        history += n + o
    return total


def test_closed_form_matches_brute_force():
    for turns in (1, 2, 5, 37, 100):
        est = estimate_loop(
            "claude-haiku-4-5",
            turns=turns,
            new_tokens_per_turn=800,
            output_tokens_per_turn=300,
            system_tokens=1500,
        )
        assert est.total_input_tokens == brute_force_input(turns, 1500, 800, 300)
        assert est.total_output_tokens == turns * 300


def test_cost_matches_price_table():
    est = estimate_loop("claude-haiku-4-5", turns=10, new_tokens_per_turn=1000,
                        output_tokens_per_turn=0, system_tokens=0)
    # input only: 10*1000 + 1000*10*9/2 = 55,000 tokens at $1/M
    assert est.total_input_tokens == 55_000
    assert est.cost_usd == pytest.approx(0.055)


def test_quadratic_growth_is_visible():
    small = estimate_loop("claude-haiku-4-5", turns=10)
    big = estimate_loop("claude-haiku-4-5", turns=100)
    # 10x the turns should be much more than 10x the cost
    assert big.cost_usd > 5 * 10 * small.cost_usd / 10


def test_cache_hit_rate_reduces_cost():
    plain = estimate_loop("claude-fable-5", turns=50)
    cached = estimate_loop("claude-fable-5", turns=50, cache_hit_rate=0.9)
    assert cached.cost_usd < plain.cost_usd * 0.35


def test_growth_factor_and_summary():
    est = estimate_loop("claude-haiku-4-5", turns=100)
    assert est.growth_factor > 10
    text = est.summary()
    assert "100 turns" in text and "estimated cost" in text
    assert "with 90% prompt-cache hits" in text


def test_compare_models_sorted_cheapest_first():
    ranked = compare_models(
        ["claude-fable-5", "claude-haiku-4-5", "gpt-5.4-mini"], turns=50
    )
    costs = [c for _, c in ranked]
    assert costs == sorted(costs)
    assert ranked[-1][0] == "claude-fable-5"


def test_validation():
    with pytest.raises(ValueError):
        estimate_loop("claude-haiku-4-5", turns=0)
    with pytest.raises(ValueError):
        estimate_loop("claude-haiku-4-5", turns=5, cache_hit_rate=1.5)
    with pytest.raises(UnknownModelError):
        estimate_loop("mystery-model", turns=5)

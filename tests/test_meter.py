import threading

import pytest

import spendcap
from spendcap import Budget, BudgetExceededError, Meter, UnknownModelError


def test_record_accumulates():
    m = Meter()
    m.record("claude-haiku-4-5", input_tokens=1_000_000, output_tokens=0)
    m.record("claude-haiku-4-5", input_tokens=0, output_tokens=1_000_000)
    assert m.spent == pytest.approx(1.00 + 5.00)
    assert m.calls == 2


def test_hard_cap_raises_on_record():
    m = Meter(budget=Budget(usd=1.00, warn_at=0.5))
    with pytest.raises(BudgetExceededError) as exc:
        for _ in range(100):
            m.record("claude-haiku-4-5", input_tokens=300_000)  # $0.30 each
    assert exc.value.spent >= 1.00
    assert exc.value.cap == 1.00
    assert m.spent >= 1.00  # the crossing call was still recorded


def test_check_is_the_circuit_breaker():
    m = Meter(budget=Budget(usd=0.50))
    m.record("claude-haiku-4-5", input_tokens=600_000, enforce=False)  # $0.60
    with pytest.raises(BudgetExceededError):
        m.check()


def test_soft_budget_never_raises():
    warned = []
    m = Meter(budget=Budget(usd=0.10, hard=False, on_warn=lambda s, c: warned.append(s)))
    for _ in range(5):
        m.record("claude-haiku-4-5", input_tokens=100_000)  # $0.10 each
    assert m.spent == pytest.approx(0.50)
    assert len(warned) == 1  # warn fires exactly once


def test_warn_threshold_fires_once():
    warned = []
    m = Meter(budget=Budget(usd=1.00, warn_at=0.5, on_warn=lambda s, c: warned.append((s, c))))
    m.record("claude-haiku-4-5", input_tokens=400_000)  # $0.40 — below
    assert warned == []
    m.record("claude-haiku-4-5", input_tokens=200_000)  # $0.60 — crosses 50%
    m.record("claude-haiku-4-5", input_tokens=100_000)  # $0.70
    assert len(warned) == 1
    assert warned[0][1] == 1.00


def test_remaining():
    m = Meter(budget=Budget(usd=1.00))
    assert m.remaining == 1.00
    m.record("claude-haiku-4-5", input_tokens=250_000)
    assert m.remaining == pytest.approx(0.75)
    assert Meter().remaining is None


def test_unknown_model_warns_and_books_zero():
    m = Meter()
    with pytest.warns(UserWarning, match="unknown model"):
        m.record("mystery-model-9000", input_tokens=1_000_000)
    assert m.spent == 0.0
    assert m.unknown_models == {"mystery-model-9000": 1}


def test_unknown_model_strict_raises():
    m = Meter(strict_pricing=True)
    with pytest.raises(UnknownModelError):
        m.record("mystery-model-9000", input_tokens=10)


def test_task_scoping_and_caps():
    m = Meter()
    with m.task("research"):
        m.record("claude-haiku-4-5", input_tokens=100_000)
    m.record("claude-haiku-4-5", input_tokens=100_000)
    assert m.task_spent("research") == pytest.approx(0.10)

    with pytest.raises(BudgetExceededError) as exc:
        with m.task("cheap-job", cap_usd=0.05):
            m.record("claude-haiku-4-5", input_tokens=100_000)  # $0.10 > cap
    assert exc.value.task == "cheap-job"


def test_thread_safety_smoke():
    m = Meter()

    def work():
        for _ in range(200):
            m.record("claude-haiku-4-5", input_tokens=1_000)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.calls == 1600
    assert m.spent == pytest.approx(1600 * 1_000 * 1.00 / 1_000_000)


def test_report_aggregation():
    m = Meter(budget=Budget(usd=10.0))
    with m.task("a"):
        m.record("claude-haiku-4-5", input_tokens=1_000_000)
        m.record("gpt-5.4-mini", input_tokens=1_000_000)
    rep = m.report()
    assert rep.spent_usd == pytest.approx(1.00 + 0.75)
    assert rep.by_model["claude-haiku-4-5"].cost_usd == pytest.approx(1.00)
    assert rep.by_task["a"].calls == 2
    text = str(rep)
    assert "spendcap report" in text and "claude-haiku-4-5" in text
    assert rep.to_json()  # serializes

"""Demo: a runaway agent loop, caught by spendcap.

Uses a fake client (no API key, no network, $0 actually spent) that bills
exactly like the real Anthropic API would. Run it:

    python examples/runaway_agent.py
"""

import sys
from dataclasses import dataclass

sys.path.insert(0, ".")  # run from repo root without installing

import spendcap


# --- a fake SDK that behaves (and would bill) like the real one -------------


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class Response:
    model: str
    usage: Usage
    content: str = "…model output…"


class Messages:
    def create(self, *, model, messages, **kw):
        # input grows with the conversation we send — like a real agent loop
        input_tokens = sum(len(m["content"]) // 4 for m in messages)
        return Response(model=model, usage=Usage(input_tokens, 300))


class FakeAnthropic:
    messages = Messages()


# --- the demo ---------------------------------------------------------------

MODEL = "claude-haiku-4-5"
CAP = 1.00  # dollars

print("1) Predict the damage BEFORE running the loop:\n")
est = spendcap.estimate_loop(MODEL, turns=200, new_tokens_per_turn=1200,
                             output_tokens_per_turn=300, system_tokens=1500)
print(est.summary())

print(f"\n2) Run the 'agent' under a ${CAP:.2f} hard cap:\n")

meter = spendcap.Meter(
    budget=spendcap.Budget(
        usd=CAP,
        warn_at=0.8,
        on_warn=lambda spent, cap: print(f"   ⚠ warn: ${spent:.3f} spent — 80% of ${cap:.2f} cap"),
    )
)
client = meter.wrap(FakeAnthropic())

history = [{"role": "user", "content": "Refactor my codebase." * 100}]
turn = 0
try:
    while True:  # oops: no exit condition — a classic runaway loop
        turn += 1
        resp = client.messages.create(model=MODEL, messages=history)
        history.append({"role": "assistant", "content": "x" * 1200})
        history.append({"role": "user", "content": "continue " * 150})
        if turn % 10 == 0:
            print(f"   turn {turn:>3}: spent ${meter.spent:.4f}  (remaining ${meter.remaining:.4f})")
except spendcap.BudgetExceededError as e:
    print(f"\n   🛑 {e}")
    print(f"   loop stopped at turn {turn} — the API was never called again.\n")

print("3) Where did the money go?\n")
print(meter.report())

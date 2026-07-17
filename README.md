# spendcap

**Real-time spend tracking, cost prediction, and runaway-loop circuit breakers for LLM API calls.**

Your agent loop just spent $40 while you got coffee. Provider dashboards tell you *after* the money is gone. `spendcap` stops the loop *before* the next call — and tells you what a loop will cost before you run it at all.

- 🧮 **Exact metering** — costs computed from the token counts your provider returns, not tokenizer guesses
- 🛑 **Circuit breaker** — set a hard USD cap; the call that would exceed it raises instead of hitting the API
- 🔮 **Cost prediction** — closed-form estimate of an agent loop's cost (input grows *quadratically* with turns; most people underestimate this by 10–50x)
- 🏷️ **Task scoping** — attribute spend to named tasks, with optional per-task caps
- 📊 **Reports** — by model, by task, as text or JSON
- **Zero dependencies.** Works with the Anthropic and OpenAI SDKs (sync + async) and any duck-typed client whose responses carry `usage`.

```
pip install spendcap
```

## Quickstart

Wrap your client. Nothing else changes.

```python
import anthropic
import spendcap

meter = spendcap.Meter(budget=spendcap.Budget(usd=5.00))
client = meter.wrap(anthropic.Anthropic())

# ... your agent loop, exactly as before ...
resp = client.messages.create(model="claude-haiku-4-5", max_tokens=1024,
                              messages=[{"role": "user", "content": "hi"}])

print(meter.spent)       # 0.0023  (USD, exact)
print(meter.remaining)   # 4.9977
```

When the cap is spent, the **next** call raises instead of reaching the API:

```python
try:
    while True:
        resp = client.messages.create(...)   # metered every call
        ...
except spendcap.BudgetExceededError as e:
    print(e)  # spendcap: meter budget exceeded — spent $5.0031 of $5.00 cap
```

The call that *crosses* the cap still returns its response (you paid for it); the breaker refuses the one after. A warning fires once at 80% of the cap (configurable: `Budget(usd=5, warn_at=0.5, on_warn=my_callback)`). Set `Budget(hard=False)` for observe-only mode.

## Predict a loop's cost before running it

Agent loops resend the whole conversation every turn, so input tokens grow quadratically:

```python
est = spendcap.estimate_loop("claude-haiku-4-5", turns=200,
                             new_tokens_per_turn=1200,
                             output_tokens_per_turn=300,
                             system_tokens=1500)
print(est.summary())
```

```
Loop estimate — claude-haiku-4-5, 200 turns
  history growth: 1,200 new + 300 output tokens/turn, 1,500 system tokens
  total input: 30,390,000 tok   total output: 60,000 tok
  estimated cost: $30.69   (turn 1: $0.0042 -> turn 200: $0.3027, 72x growth)
  with 90% prompt-cache hits: $6.07
```

Compare models for the same loop:

```python
spendcap.compare_models(["claude-haiku-4-5", "gpt-5.4-mini", "gemini-3-flash"], turns=100)
# [('gpt-5.4-mini', ...), ('gemini-3-flash', ...), ...]  cheapest first
```

## Task scoping and per-task caps

```python
with meter.task("research", cap_usd=1.50):
    ...  # calls here are tagged 'research' and capped at $1.50

with meter.task("summarize"):
    ...

print(meter.report())
```

```
spendcap report — spent $2.4312 of $5.00 cap (48.6%)
  calls: 41   input: 1,912,340 tok   output: 96,200 tok   cached: 210,000 tok
  by model:
    claude-haiku-4-5            38 calls   $2.1201
    gpt-5.4-mini                 3 calls   $0.3111
  by task:
    research                    30 calls   $1.4890
    summarize                   11 calls   $0.9422
```

`meter.report().to_json()` for machines.

## No wrapper? Record manually

Works with any provider, any framework — just feed it the usage numbers:

```python
meter.record("gpt-5.4-mini", input_tokens=1200, output_tokens=340)
meter.record("claude-haiku-4-5", input_tokens=100, cached_input_tokens=2000)
```

## Pricing data

Built-in prices (USD per 1M tokens) for current Anthropic, OpenAI, and Google models, verified **2026-07-17** (`spendcap.PRICING_AS_OF`). Model IDs resolve fuzzily: `anthropic/claude-haiku-4-5-20251001` → `claude-haiku-4-5`. Unknown models warn once and book at $0 (or raise, with `Meter(strict_pricing=True)`).

Prices change — override anything at runtime, no fork needed:

```python
spendcap.register_model("groq-llama-4-70b", input_per_m=0.59, output_per_m=0.79)
spendcap.load_pricing("my_prices.json")   # {"model": {"input": .., "output": .., "cached": ..}}
```

> Note: Claude Sonnet 5 is listed at its introductory $2/$10 rate, which runs through 2026-08-31 ($3/$15 after).

Cache accounting mirrors the providers: Anthropic cache reads bill at the cached rate and cache writes at 1.25x input (`input_tokens` already excludes both); OpenAI `prompt_tokens` includes cached tokens, so spendcap splits them out.

## Demo

No API key needed — a fake client that bills like the real thing:

```
python examples/runaway_agent.py
```

## Limitations (v0.1)

- **Streaming** responses that don't return usage aren't metered (spendcap warns once). Use `stream_options={"include_usage": True}` (OpenAI) or `meter.record()` manually.
- Wrapping is duck-typed; `isinstance` checks against the SDK's client class won't see through the proxy (`client.__wrapped__` gives the original).
- Budgets live in memory, per process. Persistence across restarts is on the roadmap.

## Roadmap

Streaming usage capture, per-provider price auto-refresh, persistent ledgers (SQLite), a CLI (`spendcap report`), LangChain/agent-framework callbacks.

## Contributing

Issues and PRs welcome — especially price-table updates and new provider usage shapes. Run `pip install -e ".[dev]" && pytest`.

## License

MIT © Shreyansh Goyal

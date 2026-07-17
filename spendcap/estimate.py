"""Predict what an agent loop will cost BEFORE you run it.

Agent loops resend the whole conversation every turn, so input tokens grow
quadratically with turn count. This module puts a number on that before you
spend the money.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .core import UnknownModelError
from .pricing import get_price


@dataclass(frozen=True)
class LoopEstimate:
    model: str
    turns: int
    system_tokens: int
    new_tokens_per_turn: int
    output_tokens_per_turn: int
    cache_hit_rate: float
    total_input_tokens: int
    total_output_tokens: int
    cost_usd: float
    first_turn_cost_usd: float
    final_turn_cost_usd: float

    @property
    def growth_factor(self) -> float:
        """How much more the last turn costs than the first."""
        if self.first_turn_cost_usd == 0:
            return float("inf")
        return self.final_turn_cost_usd / self.first_turn_cost_usd

    def summary(self) -> str:
        lines = [
            f"Loop estimate: {self.model}, {self.turns} turns",
            f"  history growth: {self.new_tokens_per_turn:,} new + "
            f"{self.output_tokens_per_turn:,} output tokens/turn, "
            f"{self.system_tokens:,} system tokens",
            f"  total input: {self.total_input_tokens:,} tok   "
            f"total output: {self.total_output_tokens:,} tok",
            f"  estimated cost: ${self.cost_usd:.2f}   "
            f"(turn 1: ${self.first_turn_cost_usd:.4f} -> "
            f"turn {self.turns}: ${self.final_turn_cost_usd:.4f}, "
            f"{self.growth_factor:.0f}x growth)",
        ]
        if self.cache_hit_rate == 0.0:
            cached = estimate_loop(
                self.model,
                turns=self.turns,
                new_tokens_per_turn=self.new_tokens_per_turn,
                output_tokens_per_turn=self.output_tokens_per_turn,
                system_tokens=self.system_tokens,
                cache_hit_rate=0.9,
            )
            lines.append(f"  with 90% prompt-cache hits: ${cached.cost_usd:.2f}")
        return "\n".join(lines)


def estimate_loop(
    model: str,
    turns: int,
    new_tokens_per_turn: int = 800,
    output_tokens_per_turn: int = 300,
    system_tokens: int = 1500,
    cache_hit_rate: float = 0.0,
) -> LoopEstimate:
    """Closed-form cost estimate for a history-resending agent loop.

    Turn t sends: system + full history so far + this turn's new tokens,
    where history grows by (new + output) tokens each turn. That makes total
    input quadratic in ``turns``::

        total_input = T*system + T*new + (new + output) * T*(T-1)/2

    ``cache_hit_rate`` is the fraction of input tokens billed at the
    provider's cached rate (0.0 = no caching, 0.9 = a well-cached loop).
    """
    if turns < 1:
        raise ValueError("turns must be >= 1")
    if not 0.0 <= cache_hit_rate <= 1.0:
        raise ValueError("cache_hit_rate must be in [0, 1]")
    price = get_price(model)
    if price is None:
        raise UnknownModelError(model)

    T, s, n, o = turns, system_tokens, new_tokens_per_turn, output_tokens_per_turn
    total_input = T * s + T * n + (n + o) * T * (T - 1) // 2
    total_output = T * o

    def _cost(inp: int, out: int) -> float:
        billed_cached = int(inp * cache_hit_rate)
        billed_full = inp - billed_cached
        return price.cost(
            input_tokens=billed_full,
            output_tokens=out,
            cached_input_tokens=billed_cached,
        )

    first_in = s + n
    final_in = s + n + (n + o) * (T - 1)
    return LoopEstimate(
        model=model,
        turns=T,
        system_tokens=s,
        new_tokens_per_turn=n,
        output_tokens_per_turn=o,
        cache_hit_rate=cache_hit_rate,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        cost_usd=_cost(total_input, total_output),
        first_turn_cost_usd=_cost(first_in, o),
        final_turn_cost_usd=_cost(final_in, o),
    )


def compare_models(
    models: Sequence[str],
    turns: int,
    new_tokens_per_turn: int = 800,
    output_tokens_per_turn: int = 300,
    system_tokens: int = 1500,
    cache_hit_rate: float = 0.0,
) -> List[Tuple[str, float]]:
    """Estimate the same loop across models. Returns [(model, usd)] cheapest first."""
    out: List[Tuple[str, float]] = []
    for m in models:
        est = estimate_loop(
            m,
            turns=turns,
            new_tokens_per_turn=new_tokens_per_turn,
            output_tokens_per_turn=output_tokens_per_turn,
            system_tokens=system_tokens,
            cache_hit_rate=cache_hit_rate,
        )
        out.append((m, est.cost_usd))
    return sorted(out, key=lambda kv: kv[1])

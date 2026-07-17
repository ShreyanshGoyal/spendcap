"""Call records and aggregated spend reports."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

UNTAGGED = "(untagged)"


@dataclass(frozen=True)
class CallRecord:
    """One metered API call."""

    ts: float
    model: str
    resolved_model: Optional[str]
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cache_write_tokens: int
    cost_usd: float
    task: Optional[str]


@dataclass
class _Bucket:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, r: CallRecord) -> None:
        self.calls += 1
        self.input_tokens += r.input_tokens
        self.output_tokens += r.output_tokens
        self.cached_input_tokens += r.cached_input_tokens
        self.cost_usd += r.cost_usd


@dataclass
class Report:
    """Aggregated view of everything a Meter has recorded."""

    spent_usd: float
    cap_usd: Optional[float]
    calls: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    by_model: Dict[str, _Bucket] = field(default_factory=dict)
    by_task: Dict[str, _Bucket] = field(default_factory=dict)
    unknown_models: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        records: List[CallRecord],
        cap_usd: Optional[float] = None,
        unknown_models: Optional[Dict[str, int]] = None,
    ) -> "Report":
        rep = cls(
            spent_usd=0.0,
            cap_usd=cap_usd,
            calls=0,
            input_tokens=0,
            output_tokens=0,
            cached_input_tokens=0,
            unknown_models=unknown_models or {},
        )
        for r in records:
            rep.spent_usd += r.cost_usd
            rep.calls += 1
            rep.input_tokens += r.input_tokens
            rep.output_tokens += r.output_tokens
            rep.cached_input_tokens += r.cached_input_tokens
            model_key = r.resolved_model or r.model
            rep.by_model.setdefault(model_key, _Bucket()).add(r)
            rep.by_task.setdefault(r.task or UNTAGGED, _Bucket()).add(r)
        return rep

    def as_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, **kwargs) -> str:
        kwargs.setdefault("indent", 2)
        return json.dumps(self.as_dict(), **kwargs)

    def __str__(self) -> str:
        cap = ""
        if self.cap_usd is not None:
            pct = (self.spent_usd / self.cap_usd * 100) if self.cap_usd else 0.0
            cap = f" of ${self.cap_usd:.2f} cap ({pct:.1f}%)"
        lines = [
            f"spendcap report — spent ${self.spent_usd:.4f}{cap}",
            f"  calls: {self.calls}   input: {self.input_tokens:,} tok   "
            f"output: {self.output_tokens:,} tok   cached: {self.cached_input_tokens:,} tok",
        ]
        if self.by_model:
            lines.append("  by model:")
            for name, b in sorted(self.by_model.items(), key=lambda kv: -kv[1].cost_usd):
                lines.append(f"    {name:<24} {b.calls:>5} calls   ${b.cost_usd:.4f}")
        if self.by_task and (len(self.by_task) > 1 or UNTAGGED not in self.by_task):
            lines.append("  by task:")
            for name, b in sorted(self.by_task.items(), key=lambda kv: -kv[1].cost_usd):
                lines.append(f"    {name:<24} {b.calls:>5} calls   ${b.cost_usd:.4f}")
        if self.unknown_models:
            names = ", ".join(sorted(self.unknown_models))
            lines.append(f"  ⚠ unpriced models (booked at $0): {names}")
        return "\n".join(lines)

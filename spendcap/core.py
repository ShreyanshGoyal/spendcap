"""Meter, Budget, and the circuit breaker."""

from __future__ import annotations

import contextvars
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from .pricing import ModelPrice, get_price, resolve_model
from .report import CallRecord, Report

__all__ = [
    "Budget",
    "Meter",
    "BudgetExceededError",
    "UnknownModelError",
]


class BudgetExceededError(RuntimeError):
    """Raised when a spend cap is hit. Carries the numbers for logging."""

    def __init__(self, spent: float, cap: float, task: Optional[str] = None):
        self.spent = spent
        self.cap = cap
        self.task = task
        scope = f"task '{task}'" if task else "meter"
        super().__init__(
            f"spendcap: {scope} budget exceeded (spent ${spent:.4f} of ${cap:.2f} cap)"
        )


class UnknownModelError(KeyError):
    """Raised in strict mode when a model has no known price."""

    def __init__(self, model: str):
        self.model = model
        super().__init__(
            f"spendcap: no price known for model '{model}'. "
            f"Add one with spendcap.register_model('{model}', input_per_m, output_per_m)."
        )


@dataclass
class Budget:
    """A hard or soft USD spending cap.

    Args:
        usd: the cap.
        warn_at: fraction of the cap at which the warn callback fires (once).
        hard: if True (default), crossing the cap raises BudgetExceededError.
            If False, spendcap only warns ("observe mode").
        on_warn: optional callback ``(spent, cap) -> None``; defaults to
            ``warnings.warn``.
    """

    usd: float
    warn_at: float = 0.8
    hard: bool = True
    on_warn: Optional[Callable[[float, float], None]] = None
    _warned: bool = field(default=False, repr=False, compare=False)

    def _fire_warn(self, spent: float) -> None:
        if self._warned:
            return
        self._warned = True
        if self.on_warn is not None:
            self.on_warn(spent, self.usd)
        else:
            warnings.warn(
                f"spendcap: spent ${spent:.4f} "
                f"({spent / self.usd:.0%} of the ${self.usd:.2f} budget)",
                stacklevel=3,
            )


_current_task: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "spendcap_task", default=None
)


class Meter:
    """Tracks LLM spend in real time and enforces budgets.

    Use it one of two ways (or both):

    1. Wrap a provider client; every call is metered automatically::

        meter = Meter(budget=Budget(usd=5.00))
        client = meter.wrap(anthropic.Anthropic())

    2. Record manually from any response's usage numbers::

        meter.record("claude-haiku-4-5", input_tokens=1200, output_tokens=340)

    The circuit breaker: once ``spent`` >= the cap, the next metered call (or
    explicit :meth:`check`) raises :class:`BudgetExceededError` instead of
    hitting the API.
    """

    def __init__(
        self,
        budget: Optional[Budget] = None,
        strict_pricing: bool = False,
        clock: Callable[[], float] = time.time,
    ):
        self.budget = budget
        self.strict_pricing = strict_pricing
        self._clock = clock
        self._lock = threading.Lock()
        self._records: List[CallRecord] = []
        self._spent = 0.0
        self._task_spent: Dict[str, float] = {}
        self._task_caps: Dict[str, float] = {}
        self._unknown_models: Dict[str, int] = {}
        self._stream_warned = False

    # ------------------------------------------------------------------ state

    @property
    def spent(self) -> float:
        """Total USD spent so far."""
        return self._spent

    @property
    def remaining(self) -> Optional[float]:
        """USD left before the cap, or None if no budget is set."""
        if self.budget is None:
            return None
        return max(self.budget.usd - self._spent, 0.0)

    @property
    def calls(self) -> int:
        return len(self._records)

    @property
    def records(self) -> List[CallRecord]:
        return list(self._records)

    @property
    def unknown_models(self) -> Dict[str, int]:
        """Models seen without a known price -> call count (cost booked as $0)."""
        return dict(self._unknown_models)

    # ------------------------------------------------------------- enforcement

    def check(self, task: Optional[str] = None) -> None:
        """Raise BudgetExceededError if the cap (or a task cap) is spent.

        Wrapped clients call this before every API call; that is the
        circuit breaker.
        """
        if self.budget is not None and self.budget.hard and self._spent >= self.budget.usd:
            raise BudgetExceededError(self._spent, self.budget.usd)
        task = task if task is not None else _current_task.get()
        if task is not None and task in self._task_caps:
            if self._task_spent.get(task, 0.0) >= self._task_caps[task]:
                raise BudgetExceededError(
                    self._task_spent.get(task, 0.0), self._task_caps[task], task=task
                )

    # --------------------------------------------------------------- recording

    def record(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        cache_write_tokens: int = 0,
        task: Optional[str] = None,
        enforce: bool = True,
    ) -> CallRecord:
        """Record one completed call and return its CallRecord (with cost).

        If ``enforce`` and this record crosses a hard cap, raises
        BudgetExceededError *after* recording (the tokens were already
        consumed; the point is to stop the loop).
        """
        price = get_price(model)
        resolved = resolve_model(model)
        if price is None:
            if self.strict_pricing:
                raise UnknownModelError(model)
            with self._lock:
                first_time = model not in self._unknown_models
                self._unknown_models[model] = self._unknown_models.get(model, 0) + 1
            if first_time:
                warnings.warn(
                    f"spendcap: unknown model '{model}'; cost recorded as $0. "
                    f"Register a price with spendcap.register_model().",
                    stacklevel=2,
                )
            price = ModelPrice(0.0, 0.0, 0.0)

        cost = price.cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        task = task if task is not None else _current_task.get()
        rec = CallRecord(
            ts=self._clock(),
            model=model,
            resolved_model=resolved,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_tokens=cache_write_tokens,
            cost_usd=cost,
            task=task,
        )
        with self._lock:
            self._records.append(rec)
            self._spent += cost
            if task is not None:
                self._task_spent[task] = self._task_spent.get(task, 0.0) + cost
            spent = self._spent

        if self.budget is not None:
            if spent >= self.budget.warn_at * self.budget.usd:
                self.budget._fire_warn(spent)
            if enforce and self.budget.hard and spent >= self.budget.usd:
                raise BudgetExceededError(spent, self.budget.usd)
        if enforce and task is not None and task in self._task_caps:
            if self._task_spent.get(task, 0.0) >= self._task_caps[task]:
                raise BudgetExceededError(
                    self._task_spent[task], self._task_caps[task], task=task
                )
        return rec

    # ------------------------------------------------------------------- tasks

    @contextmanager
    def task(self, name: str, cap_usd: Optional[float] = None) -> Iterator[None]:
        """Attribute all calls inside the block to ``name``.

        Optionally give the task its own hard cap::

            with meter.task("research", cap_usd=1.50):
                ...
        """
        if cap_usd is not None:
            self._task_caps[name] = cap_usd
        token = _current_task.set(name)
        try:
            yield
        finally:
            _current_task.reset(token)

    def task_spent(self, name: str) -> float:
        return self._task_spent.get(name, 0.0)

    # ---------------------------------------------------------------- wrapping

    def wrap(self, client: Any) -> Any:
        """Return a transparent proxy of ``client`` that meters every call.

        Works with the official Anthropic and OpenAI SDKs (sync and async)
        and any duck-typed client whose responses carry a ``usage`` object.
        """
        from .wrappers import MeteredProxy

        return MeteredProxy(client, self)

    # --------------------------------------------------------------- reporting

    def report(self) -> Report:
        """Aggregated spend report (by model, by task)."""
        with self._lock:
            return Report.build(
                self._records,
                cap_usd=self.budget.usd if self.budget else None,
                unknown_models=dict(self._unknown_models),
            )

    def _warn_stream_once(self) -> None:
        if not self._stream_warned:
            self._stream_warned = True
            warnings.warn(
                "spendcap: a streaming call returned no usage data and was not "
                "metered. Meter streams manually with meter.record(), or use "
                "stream_options={'include_usage': True} (OpenAI).",
                stacklevel=3,
            )

    def __repr__(self) -> str:  # pragma: no cover
        cap = f" / cap ${self.budget.usd:.2f}" if self.budget else ""
        return f"<spendcap.Meter spent ${self._spent:.4f}{cap}, {self.calls} calls>"

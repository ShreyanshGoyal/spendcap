"""spendcap: real-time spend tracking, cost prediction, and runaway-loop
circuit breakers for LLM API calls.

Quickstart::

    import spendcap

    meter = spendcap.Meter(budget=spendcap.Budget(usd=5.00))
    client = meter.wrap(anthropic.Anthropic())   # use exactly as normal

    # ... your agent loop ...
    # raises spendcap.BudgetExceededError before the call that would
    # exceed the cap.

    print(meter.report())
"""

from .core import Budget, BudgetExceededError, Meter, UnknownModelError
from .estimate import LoopEstimate, compare_models, estimate_loop
from .pricing import (
    PRICING_AS_OF,
    ModelPrice,
    get_price,
    known_models,
    load_pricing,
    register_model,
    resolve_model,
)
from .report import CallRecord, Report

__version__ = "0.1.0"

__all__ = [
    "Meter",
    "Budget",
    "BudgetExceededError",
    "UnknownModelError",
    "estimate_loop",
    "compare_models",
    "LoopEstimate",
    "register_model",
    "load_pricing",
    "get_price",
    "resolve_model",
    "known_models",
    "ModelPrice",
    "CallRecord",
    "Report",
    "PRICING_AS_OF",
    "__version__",
]

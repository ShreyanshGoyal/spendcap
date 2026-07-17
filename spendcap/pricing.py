"""Model pricing tables and model-name resolution.

All prices are **USD per million tokens**, current as of :data:`PRICING_AS_OF`.
Prices change. Override or extend at runtime with :func:`register_model` or
:func:`load_pricing`; never fork the library just to fix a price.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Optional, Union

#: Date the built-in price table was last verified against provider docs.
PRICING_AS_OF = "2026-07-17"

#: Multiplier applied to prompt-cache *writes* (Anthropic bills 1.25x input).
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token prices for one model."""

    input_per_m: float
    output_per_m: float
    cached_input_per_m: Optional[float] = None  # None -> assume 10% of input

    @property
    def cached_rate(self) -> float:
        if self.cached_input_per_m is not None:
            return self.cached_input_per_m
        return self.input_per_m * 0.10

    def cost(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """Exact USD cost of one call given token counts from the API response."""
        return (
            input_tokens * self.input_per_m
            + output_tokens * self.output_per_m
            + cached_input_tokens * self.cached_rate
            + cache_write_tokens * self.input_per_m * CACHE_WRITE_MULTIPLIER
        ) / 1_000_000


# ---------------------------------------------------------------------------
# Built-in table (USD per 1M tokens): (input, output, cached_input)
# Verified 2026-07-17. Sonnet 5 is introductory pricing through 2026-08-31;
# standard pricing from 2026-09-01 is 3.00 / 15.00.
# ---------------------------------------------------------------------------
_BUILTIN: Dict[str, ModelPrice] = {
    # --- Anthropic (current) ---
    "claude-fable-5": ModelPrice(10.00, 50.00, 1.00),
    "claude-opus-4-8": ModelPrice(5.00, 25.00, 0.50),
    "claude-sonnet-5": ModelPrice(2.00, 10.00, 0.20),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00, 0.10),
    # --- Anthropic (legacy, still commonly pinned) ---
    "claude-3-5-sonnet": ModelPrice(3.00, 15.00, 0.30),
    "claude-3-5-haiku": ModelPrice(0.80, 4.00, 0.08),
    # --- OpenAI ---
    "gpt-5.6-sol": ModelPrice(5.00, 30.00, 0.50),
    "gpt-5.6-terra": ModelPrice(2.50, 15.00, 0.25),
    "gpt-5.6-luna": ModelPrice(1.00, 6.00, 0.10),
    "gpt-5.5": ModelPrice(5.00, 30.00, 0.50),
    "gpt-5.4": ModelPrice(2.50, 15.00, 0.25),
    "gpt-5.4-mini": ModelPrice(0.75, 4.50, 0.075),
    "gpt-5.4-nano": ModelPrice(0.20, 1.25, 0.02),
    "gpt-4.1": ModelPrice(2.00, 8.00, 0.50),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60, 0.10),
    "gpt-4.1-nano": ModelPrice(0.10, 0.40, 0.025),
    "gpt-4o": ModelPrice(2.50, 10.00, 1.25),
    # --- Google ---
    "gemini-3.5-flash": ModelPrice(1.50, 9.00),
    "gemini-3.1-pro": ModelPrice(2.00, 12.00),
    "gemini-3-flash": ModelPrice(0.50, 3.00),
    "gemini-3-flash-lite": ModelPrice(0.25, 1.50),
    "gemini-2.5-pro": ModelPrice(1.25, 10.00),
    "gemini-2.5-flash": ModelPrice(0.30, 2.50),
}

# User-registered prices take priority over built-ins.
_REGISTRY: Dict[str, ModelPrice] = dict(_BUILTIN)

_DATE_SUFFIX = re.compile(r"-(?:19|20)\d{6}$|-(?:19|20)\d{2}-\d{2}-\d{2}$")


def _normalize(model: str) -> str:
    """Lowercase, drop provider prefixes and date/latest suffixes.

    'anthropic/Claude-Haiku-4-5-20251001' -> 'claude-haiku-4-5'
    'models/gemini-3.5-flash'             -> 'gemini-3.5-flash'
    """
    name = model.strip().lower().split("/")[-1]
    if name.endswith("-latest"):
        name = name[: -len("-latest")]
    name = _DATE_SUFFIX.sub("", name)
    return name


def resolve_model(model: str) -> Optional[str]:
    """Return the canonical registry key for ``model``, or None if unknown.

    Matching order: exact (after normalization), then the *longest* registry
    key that is a dash/dot-boundary prefix of the name. We never guess across
    families: 'gpt-5.6' alone will NOT silently resolve to some 'gpt-5.6-x'.
    """
    name = _normalize(model)
    if name in _REGISTRY:
        return name
    best: Optional[str] = None
    for key in _REGISTRY:
        if name.startswith(key) and len(name) > len(key) and name[len(key)] in "-_.:":
            if best is None or len(key) > len(best):
                best = key
    return best


def get_price(model: str) -> Optional[ModelPrice]:
    """Price for a model name (raw API id is fine), or None if unknown."""
    key = resolve_model(model)
    return _REGISTRY.get(key) if key else None


def register_model(
    name: str,
    input_per_m: float,
    output_per_m: float,
    cached_input_per_m: Optional[float] = None,
) -> None:
    """Add or override a model price at runtime.

    >>> register_model("groq-llama-4-70b", 0.59, 0.79)
    """
    _REGISTRY[_normalize(name)] = ModelPrice(input_per_m, output_per_m, cached_input_per_m)


def load_pricing(source: Union[str, Dict[str, dict]]) -> int:
    """Bulk-load prices from a dict or a JSON file path. Returns count loaded.

    Format: {"model-name": {"input": 1.0, "output": 5.0, "cached": 0.1}, ...}
    """
    if isinstance(source, str):
        with open(source, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = source
    for name, spec in data.items():
        register_model(
            name,
            float(spec["input"]),
            float(spec["output"]),
            float(spec["cached"]) if spec.get("cached") is not None else None,
        )
    return len(data)


def known_models() -> Dict[str, ModelPrice]:
    """Copy of the current model->price registry."""
    return dict(_REGISTRY)

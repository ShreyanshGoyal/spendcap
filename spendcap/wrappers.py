"""Transparent client proxy: meter any SDK whose responses carry usage data.

No provider SDK is imported here; everything is duck-typed, so the same
proxy handles Anthropic (input_tokens/output_tokens), OpenAI chat completions
(prompt_tokens/completion_tokens), OpenAI responses (input_tokens +
input_tokens_details.cached_tokens), and dict-shaped responses.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

_PRIMITIVES = (str, bytes, int, float, bool, type(None), list, tuple, dict, set)


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def extract_usage(result: Any, kwargs: dict) -> Optional[dict]:
    """Pull (model, token counts) out of an API response, or None."""
    usage = _get(result, "usage")
    if usage is None:
        return None
    model = _get(result, "model") or kwargs.get("model")
    if not model:
        return None

    it = _get(usage, "input_tokens")
    ot = _get(usage, "output_tokens")
    if it is not None or ot is not None:
        cache_read = _get(usage, "cache_read_input_tokens")
        cache_write = _get(usage, "cache_creation_input_tokens") or 0
        it = it or 0
        if cache_read is not None:
            # Anthropic semantics: input_tokens EXCLUDES cache reads/writes.
            cached = cache_read
        else:
            # OpenAI responses-API semantics: input_tokens INCLUDES cached.
            details = _get(usage, "input_tokens_details")
            cached = (_get(details, "cached_tokens") or 0) if details is not None else 0
            it = max(it - cached, 0)
        return dict(
            model=model,
            input_tokens=it,
            output_tokens=ot or 0,
            cached_input_tokens=cached,
            cache_write_tokens=cache_write,
        )

    pt = _get(usage, "prompt_tokens")
    ct = _get(usage, "completion_tokens")
    if pt is not None:
        # OpenAI chat-completions semantics: prompt_tokens INCLUDES cached.
        details = _get(usage, "prompt_tokens_details")
        cached = (_get(details, "cached_tokens") or 0) if details is not None else 0
        return dict(
            model=model,
            input_tokens=max(pt - cached, 0),
            output_tokens=ct or 0,
            cached_input_tokens=cached,
            cache_write_tokens=0,
        )
    return None


class MeteredProxy:
    """Wraps a client object; attribute access returns further proxies and
    every call is budget-checked before and metered after."""

    def __init__(self, obj: Any, meter: Any):
        object.__setattr__(self, "_sc_obj", obj)
        object.__setattr__(self, "_sc_meter", meter)

    # -- attribute plumbing -------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, "_sc_obj")
        attr = getattr(target, name)
        if name.startswith("_") or isinstance(attr, _PRIMITIVES):
            return attr
        return MeteredProxy(attr, object.__getattribute__(self, "_sc_meter"))

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_sc_obj"), name, value)

    def __repr__(self) -> str:
        return f"MeteredProxy({object.__getattribute__(self, '_sc_obj')!r})"

    def __dir__(self):
        return dir(object.__getattribute__(self, "_sc_obj"))

    @property
    def __wrapped__(self) -> Any:
        """The original, unmetered object."""
        return object.__getattribute__(self, "_sc_obj")

    # -- the interesting part ----------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        meter = object.__getattribute__(self, "_sc_meter")
        target = object.__getattribute__(self, "_sc_obj")

        meter.check()  # circuit breaker: refuse the call if the cap is spent
        result = target(*args, **kwargs)

        if inspect.isawaitable(result):
            return self._sc_await(result, kwargs)

        self._sc_record(result, kwargs)
        return result

    async def _sc_await(self, awaitable: Any, kwargs: dict) -> Any:
        result = await awaitable
        self._sc_record(result, kwargs)
        return result

    def _sc_record(self, result: Any, kwargs: dict) -> None:
        meter = object.__getattribute__(self, "_sc_meter")
        usage = extract_usage(result, kwargs)
        if usage is not None:
            # Don't raise mid-flight and eat the response the caller paid for;
            # the pre-call check() on the NEXT call is the breaker.
            meter.record(**usage, enforce=False)
        elif kwargs.get("stream"):
            meter._warn_stream_once()

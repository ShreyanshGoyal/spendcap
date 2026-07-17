"""Fake provider SDKs — no network, no dependencies."""

from dataclasses import dataclass, field
from typing import Optional


# ---- Anthropic-shaped ------------------------------------------------------


@dataclass
class FakeAnthropicUsage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class FakeAnthropicResponse:
    model: str
    usage: FakeAnthropicUsage
    content: str = "ok"


class _AnthropicMessages:
    def __init__(self, outer):
        self._outer = outer

    def create(self, *, model, messages, max_tokens=1024, **kw):
        self._outer.calls_made += 1
        n = self._outer.next_usage or FakeAnthropicUsage(1000, 200)
        return FakeAnthropicResponse(model=model, usage=n)


class FakeAnthropic:
    """Mimics anthropic.Anthropic() closely enough for the proxy."""

    def __init__(self):
        self.calls_made = 0
        self.next_usage: Optional[FakeAnthropicUsage] = None
        self.messages = _AnthropicMessages(self)
        self.api_key = "sk-fake"


# ---- OpenAI-shaped ---------------------------------------------------------


@dataclass
class FakePromptDetails:
    cached_tokens: int = 0


@dataclass
class FakeOpenAIUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int = 0
    prompt_tokens_details: FakePromptDetails = field(default_factory=FakePromptDetails)


@dataclass
class FakeOpenAIResponse:
    model: str
    usage: FakeOpenAIUsage


class _Completions:
    def __init__(self, outer):
        self._outer = outer

    def create(self, *, model, messages, **kw):
        self._outer.calls_made += 1
        u = self._outer.next_usage or FakeOpenAIUsage(1000, 200)
        return FakeOpenAIResponse(model=model, usage=u)


class _Chat:
    def __init__(self, outer):
        self.completions = _Completions(outer)


class FakeOpenAI:
    def __init__(self):
        self.calls_made = 0
        self.next_usage: Optional[FakeOpenAIUsage] = None
        self.chat = _Chat(self)

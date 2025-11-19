from pydantic import BaseModel, Field
from enum import Enum


class LLMRole(str, Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """Represents a single message in a conversation."""
    role: LLMRole
    content: str
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict:
        """Convert to LiteLLM-compatible dictionary."""
        msg = {
            "role": self.role.value,
            "content": self.content
        }
        if self.name:
            msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


class LLMResponse(BaseModel):
    """Standardized response from any LLM call."""
    content: str | None = None
    tool_calls: list[dict] | None = None
    finish_reason: str | None = None
    model: str | None = None
    usage: dict | None = None
    raw_response: dict | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response includes tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0


class LLMConfig(BaseModel):
    """Configuration for LLM invocation."""
    model: str
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    top_p: float = Field(default=1.0, ge=0, le=1)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    timeout: int | None = Field(default=None, gt=0)
    api_key: str | None = Field(default=None)
    api_base: str | None = Field(default=None)
    extra_params: dict = Field(default_factory=dict)

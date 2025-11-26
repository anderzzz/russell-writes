from pydantic import BaseModel, Field
from enum import Enum
from typing import Any, Literal


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
    model_config = {"arbitrary_types_allowed": True}

    content: str | None = None
    tool_calls: list[dict] | None = None
    finish_reason: str | None = None
    model: str | None = None
    usage: dict | None = None
    raw_response: Any = None

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
    api_key: str
    api_base: str | None = Field(default=None)
    response_format: dict | None = Field(
        default=None,
        description="Structured output format. Use {'type': 'json_object'} for JSON mode."
    )
    extra_params: dict = Field(default_factory=dict)


# =============================================================================
# Structured Output Models
# =============================================================================

class StyleJudgment(BaseModel):
    """
    Structured output format for style similarity judgments.

    Used by the style judge to compare a reconstruction against the original
    gold standard text.
    """

    ranking: Literal["original_better", "reconstruction_better", "roughly_equal"] = Field(
        ...,
        description="Which text better matches the gold standard style"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Judge's confidence level in the ranking"
    )
    reasoning: str = Field(
        ...,
        min_length=10,
        description="Chain-of-thought explanation for the ranking decision"
    )


class MethodMapping(BaseModel):
    """
    Maps anonymous labels to reconstruction methods for blind evaluation.

    Used to track which text (A, B, C, D) corresponds to which method
    (generic, fewshot, author, instructions) during comparative judging.
    """

    text_a: str = Field(..., description="Method assigned to Text A")
    text_b: str = Field(..., description="Method assigned to Text B")
    text_c: str = Field(..., description="Method assigned to Text C")
    text_d: str = Field(..., description="Method assigned to Text D")

    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate that method is one of the four allowed values."""
        valid = {'generic', 'fewshot', 'author', 'instructions'}
        if v not in valid:
            raise ValueError(f"Method must be one of {valid}, got '{v}'")
        return v

    def model_post_init(self, __context):
        """Ensure all 4 methods appear exactly once."""
        # Validate each method
        for field in ['text_a', 'text_b', 'text_c', 'text_d']:
            value = getattr(self, field)
            self.validate_method(value)

        # Ensure uniqueness
        methods = {self.text_a, self.text_b, self.text_c, self.text_d}
        expected = {'generic', 'fewshot', 'author', 'instructions'}
        if methods != expected:
            raise ValueError(
                f"Must map all 4 methods exactly once. "
                f"Got {methods}, expected {expected}"
            )


class StyleJudgmentComparative(BaseModel):
    """
    Structured output for comparative blind ranking of 4 reconstructions.

    Judge ranks 4 reconstructions (labeled A, B, C, D) from 1-4 based on
    stylistic similarity to the original gold standard text.
    """

    ranking_text_a: int = Field(..., ge=1, le=4, description="Rank for Text A (1-4)")
    ranking_text_b: int = Field(..., ge=1, le=4, description="Rank for Text B (1-4)")
    ranking_text_c: int = Field(..., ge=1, le=4, description="Rank for Text C (1-4)")
    ranking_text_d: int = Field(..., ge=1, le=4, description="Rank for Text D (1-4)")
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Judge's confidence level in the rankings"
    )
    reasoning: str = Field(
        ...,
        min_length=10,
        description="Chain-of-thought explanation for the ranking decisions"
    )

    def model_post_init(self, __context):
        """Ensure rankings are 1, 2, 3, 4 exactly (no ties, no duplicates)."""
        rankings = {
            self.ranking_text_a,
            self.ranking_text_b,
            self.ranking_text_c,
            self.ranking_text_d
        }
        if rankings != {1, 2, 3, 4}:
            raise ValueError(
                f"Rankings must be exactly [1, 2, 3, 4] with no duplicates. "
                f"Got {sorted([self.ranking_text_a, self.ranking_text_b, self.ranking_text_c, self.ranking_text_d])}"
            )

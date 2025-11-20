from pydantic import BaseModel, Field
from .base import Tool, ToolConfig

class WordCountTool(Tool):
    """Tool for counting words in text."""

    class Parameters(BaseModel):
        text: str = Field(..., description="Text to count words in")

    def __init__(self):
        super().__init__(ToolConfig(
            name="word_count",
            description="Count the number of words in a text"
        ))

    def execute(self, text: str) -> str:
        """Count words in the provided text."""
        word_count = len(text.split())
        return f"The text contains {word_count} words."

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.config.name,
                "description": self.config.description,
                "parameters": self.Parameters.model_json_schema()
            }
        }

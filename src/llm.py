"""
Simple LLM wrapper for text completion.

Provides a clean interface for text-in, text-out interactions with any
LiteLLM-supported model.
"""
from typing import Optional, Union
import json
import litellm

from models.llm_config_models import LLMConfig, Message, LLMRole, LLMResponse
from tools import Tool


class LLM:
    """
    Simple LLM wrapper for text completion.

    Handles basic text-in, text-out interactions with any LiteLLM-supported model.
    Focuses on simplicity and clarity over feature completeness.
    """

    def __init__(self, config: Union[str, LLMConfig], **kwargs):
        """
        Initialize the LLM with configuration.

        Args:
            config: Either an LLMConfig object or a model string
            **kwargs: If config is a string, these become config parameters

        Example:
            # Explicit config (recommended, matches PromptMaker pattern)
            llm = LLM(LLMConfig(model="gpt-4", temperature=0.7))

            # Shorthand for notebooks
            llm = LLM("gpt-4", temperature=0.7)
        """
        if isinstance(config, str):
            # Shorthand: model string + kwargs
            self.config = LLMConfig(model=config, **kwargs)
        else:
            # Explicit: LLMConfig object (preferred)
            self.config = config

    def complete(
            self,
            prompt: Union[str, list[Message]],
            system: Optional[str] = None,
            **kwargs
    ) -> LLMResponse:
        """
        Execute a single completion.

        Args:
            prompt: Either a string (converted to user message) or list of Messages
            system: Optional system prompt to prepend
            **kwargs: Override config parameters for this call

        Returns:
            LLMResponse containing the model's output

        Example:
            llm = LLM("gpt-4")
            response = llm.complete("What is the capital of France?")
            print(response.content)
        """
        # Convert string prompt to message list
        if isinstance(prompt, str):
            messages = [Message(role=LLMRole.USER, content=prompt)]
        else:
            messages = list(prompt)

        # Prepend system message if provided
        if system:
            messages.insert(0, Message(role=LLMRole.SYSTEM, content=system))

        # Build the request
        request_params = self._build_request_params(messages, **kwargs)

        # Execute the completion
        raw_response = litellm.completion(**request_params)

        # Parse and return the response
        return self._parse_response(raw_response)

    def _build_request_params(self, messages: list[Message], **overrides) -> dict:
        """
        Build parameters for the LiteLLM completion call.

        Args:
            messages: List of messages to send
            **overrides: Parameters to override from config

        Returns:
            Dictionary of parameters for litellm.completion()
        """
        # Start with config defaults
        params = {
            "model": self.config.model,
            "messages": [msg.model_dump(exclude_none=True) for msg in messages],
            "temperature": self.config.temperature,
        }

        # Add optional config parameters if set
        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens
        if self.config.top_p != 1.0:
            params["top_p"] = self.config.top_p
        if self.config.frequency_penalty != 0.0:
            params["frequency_penalty"] = self.config.frequency_penalty
        if self.config.presence_penalty != 0.0:
            params["presence_penalty"] = self.config.presence_penalty
        if self.config.timeout:
            params["timeout"] = self.config.timeout
        if self.config.api_key:
            params["api_key"] = self.config.api_key
        if self.config.api_base:
            params["api_base"] = self.config.api_base

        # Apply any overrides for this specific call
        params.update(overrides)

        return params

    def _parse_response(self, raw_response) -> LLMResponse:
        """
        Parse raw LiteLLM response into our standardized format.

        Args:
            raw_response: Raw response from litellm.completion()

        Returns:
            Standardized LLMResponse object
        """
        choice = raw_response.choices[0]
        message = choice.message

        # Extract usage information if available
        usage = None
        if hasattr(raw_response, 'usage'):
            usage = {
                "prompt_tokens": raw_response.usage.prompt_tokens,
                "completion_tokens": raw_response.usage.completion_tokens,
                "total_tokens": raw_response.usage.total_tokens
            }

        return LLMResponse(
            content=getattr(message, 'content', None),
            tool_calls=getattr(message, 'tool_calls', None),
            finish_reason=choice.finish_reason,
            model=raw_response.model,
            usage=usage,
            raw_response=raw_response
        )


class ToolLLM(LLM):
    """
    LLM with tool execution capabilities.

    Manages tool registration, execution, and the control flow between
    LLM responses and tool calls.
    """

    def __init__(self, config: Union[str, LLMConfig], **kwargs):
        """
        Initialize the tool-enabled LLM.

        Args:
            config: Either an LLMConfig object or a model string
            **kwargs: If config is a string, these become config parameters

        Example:
            # Explicit config
            llm = ToolLLM(LLMConfig(model="gpt-4", temperature=0.7))

            # Shorthand
            llm = ToolLLM("gpt-4", temperature=0.7)
        """
        super().__init__(config, **kwargs)
        self.tools: dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """
        Register a tool for use by the LLM.

        Args:
            tool: Tool instance to register

        Example:
            llm = ToolLLM("gpt-4")
            llm.register_tool(WordCountTool())
        """
        self.tools[tool.config.name] = tool

    def complete_with_tools(
            self,
            prompt: str,
            system: Optional[str] = None,
            max_iterations: int = 5,
            **kwargs
    ) -> str:
        """
        Execute completion with automatic tool handling.

        The LLM will automatically invoke registered tools as needed to
        answer the prompt, handling multiple rounds of tool calls if necessary.

        Args:
            prompt: User's question or request
            system: Optional system prompt
            max_iterations: Maximum rounds of tool execution
            **kwargs: Override config parameters

        Returns:
            Final text response after all tool executions

        """
        # Initialize conversation with user prompt
        messages = []
        if system:
            messages.append(Message(role=LLMRole.SYSTEM, content=system))
        messages.append(Message(role=LLMRole.USER, content=prompt))

        # Get tool schemas for the LLM
        tool_schemas = [tool.to_openai_schema() for tool in self.tools.values()]

        # Tool execution loop
        for iteration in range(max_iterations):
            # Get LLM response with tool schemas
            response = self._call_llm_with_tools(messages, tool_schemas, **kwargs)

            # If no tool calls, we have our final answer
            if not response.tool_calls:
                return response.content or "No response generated."

            # Add assistant's message (with tool calls) to history
            messages.append(
                Message(
                    role=LLMRole.ASSISTANT,
                    content=response.content or "",
                    tool_calls=response.tool_calls
                )
            )

            # Execute each tool call and add results
            for tool_call in response.tool_calls:
                tool_result = self._execute_tool_call(tool_call)
                messages.append(
                    Message(
                        role=LLMRole.TOOL,
                        content=tool_result['output'],
                        tool_call_id=tool_call['id'],
                        name=tool_result['name']
                    )
                )

        # If we've exhausted iterations, return last response
        return response.content or "Maximum iterations reached without final answer."

    def _call_llm_with_tools(
            self,
            messages: list[Message],
            tool_schemas: list[dict],
            **kwargs
    ) -> LLMResponse:
        """
        Call the LLM with tool schemas enabled.

        Args:
            messages: Conversation history
            tool_schemas: OpenAI-format tool definitions
            **kwargs: Override config parameters

        Returns:
            LLMResponse potentially containing tool calls
        """
        # Build request parameters
        params = self._build_request_params(messages, **kwargs)

        # Add tools if available
        if tool_schemas:
            params['tools'] = tool_schemas
            params['tool_choice'] = 'auto'  # Let the model decide when to use tools

        # Execute the completion
        raw_response = litellm.completion(**params)

        # Parse and return
        return self._parse_response(raw_response)

    def _execute_tool_call(self, tool_call: dict) -> dict:
        """
        Execute a single tool call.

        Args:
            tool_call: Tool call from LLM response

        Returns:
            Dictionary with tool execution results
        """
        function_name = tool_call['function']['name']
        arguments_str = tool_call['function']['arguments']

        # Parse arguments (they come as JSON string)
        try:
            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
        except json.JSONDecodeError:
            return {
                'name': function_name,
                'output': f"Error: Failed to parse arguments for {function_name}"
            }

        # Execute the tool
        if function_name in self.tools:
            try:
                tool = self.tools[function_name]
                output = tool.execute(**arguments)
            except Exception as e:
                output = f"Error executing {function_name}: {str(e)}"
        else:
            output = f"Error: Tool '{function_name}' not found"

        return {
            'name': function_name,
            'output': str(output)
        }

    def list_tools(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self.tools.keys())

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a registered tool by name."""
        return self.tools.get(name)

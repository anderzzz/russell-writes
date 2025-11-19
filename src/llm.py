"""
LLM wrapper and abstraction layer over LiteLLM.

Provides unified interface for:
- Pure text completion (text in, text out)
- Tool/function calling (text in, function execution, text out)
"""

from typing import Any, Dict, List, Optional, Callable, Union
import litellm

from models.llm_config_models import (
    LLMRole,
    Message,
    LLMResponse,
    LLMConfig
)


class BaseLLM:
    """
    Base LLM wrapper for pure text completion.

    Handles basic text-in, text-out interactions with any LiteLLM-supported model.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.conversation_history: List[Message] = []

    def _build_litellm_kwargs(self, messages: List[Message], **kwargs) -> Dict[str, Any]:
        """Build kwargs dict for litellm.completion()."""
        litellm_kwargs = {
            "model": self.config.model,
            "messages": [msg.to_dict() for msg in messages],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
        }

        # Add optional parameters
        if self.config.timeout:
            litellm_kwargs["timeout"] = self.config.timeout
        if self.config.api_key:
            litellm_kwargs["api_key"] = self.config.api_key
        if self.config.api_base:
            litellm_kwargs["api_base"] = self.config.api_base

        # Merge any extra params
        litellm_kwargs.update(self.config.extra_params)
        litellm_kwargs.update(kwargs)

        return litellm_kwargs

    def _parse_response(self, raw_response: Any) -> LLMResponse:
        """Parse raw LiteLLM response into standardized format."""
        choice = raw_response.choices[0]
        message = choice.message

        return LLMResponse(
            content=getattr(message, "content", None),
            tool_calls=getattr(message, "tool_calls", None),
            finish_reason=choice.finish_reason,
            model=raw_response.model,
            usage=raw_response.usage.dict() if hasattr(raw_response, "usage") else None,
            raw_response=raw_response
        )

    def complete(
        self,
        messages: Union[str, List[Message]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Execute a completion request.

        Args:
            messages: Either a string (converted to user message) or list of Message objects
            system_prompt: Optional system prompt to prepend
            **kwargs: Override config parameters for this call

        Returns:
            LLMResponse with the model's output
        """
        # Convert string to message list
        if isinstance(messages, str):
            msg_list = [Message(role=LLMRole.USER, content=messages)]
        else:
            msg_list = list(messages)

        # Prepend system prompt if provided
        if system_prompt:
            msg_list.insert(0, Message(role=LLMRole.SYSTEM, content=system_prompt))

        # Build and execute request
        litellm_kwargs = self._build_litellm_kwargs(msg_list, **kwargs)
        raw_response = litellm.completion(**litellm_kwargs)

        return self._parse_response(raw_response)

    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        reset_history: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Stateful chat interface maintaining conversation history.

        Args:
            user_message: The user's message
            system_prompt: Optional system prompt (added once at start)
            reset_history: Clear history before this message
            **kwargs: Override config parameters

        Returns:
            LLMResponse with the model's output
        """
        if reset_history:
            self.conversation_history.clear()

        # Add system prompt if this is first message
        if system_prompt and not self.conversation_history:
            self.conversation_history.append(
                Message(role=LLMRole.SYSTEM, content=system_prompt)
            )

        # Add user message
        self.conversation_history.append(
            Message(role=LLMRole.USER, content=user_message)
        )

        # Get response
        response = self.complete(self.conversation_history, **kwargs)

        # Add assistant response to history
        if response.content:
            self.conversation_history.append(
                Message(
                    role=LLMRole.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls
                )
            )

        return response

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()


class ToolCallingLLM(BaseLLM):
    """
    Extended LLM wrapper with tool/function calling capabilities.

    Handles tool definitions, tool call execution, and result integration.
    """

    def __init__(self, config: LLMConfig, tools: Optional[List[Dict[str, Any]]] = None):
        super().__init__(config)
        self.tools = tools or []
        self.tool_handlers: Dict[str, Callable] = {}

    def register_tool(self, tool_definition: Dict[str, Any], handler: Callable):
        """
        Register a tool with its handler function.

        Args:
            tool_definition: OpenAI-style tool definition dict
            handler: Callable that executes the tool
        """
        tool_name = tool_definition["function"]["name"]
        self.tools.append(tool_definition)
        self.tool_handlers[tool_name] = handler

    def complete_with_tools(
        self,
        messages: Union[str, List[Message]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Execute completion with tool calling enabled.

        Args:
            messages: User messages
            tools: Tool definitions (defaults to registered tools)
            tool_choice: "auto", "none", or specific tool name
            system_prompt: Optional system prompt
            **kwargs: Override config parameters

        Returns:
            LLMResponse potentially containing tool calls
        """
        # Convert string to message list
        if isinstance(messages, str):
            msg_list = [Message(role=LLMRole.USER, content=messages)]
        else:
            msg_list = list(messages)

        # Prepend system prompt if provided
        if system_prompt:
            msg_list.insert(0, Message(role=LLMRole.SYSTEM, content=system_prompt))

        # Use provided tools or registered tools
        active_tools = tools or self.tools

        # Build request with tools
        litellm_kwargs = self._build_litellm_kwargs(msg_list, **kwargs)
        if active_tools:
            litellm_kwargs["tools"] = active_tools
            litellm_kwargs["tool_choice"] = tool_choice

        raw_response = litellm.completion(**litellm_kwargs)
        return self._parse_response(raw_response)

    def execute_tool_calls(self, response: LLMResponse) -> List[Dict[str, Any]]:
        """
        Execute tool calls from a response.

        Args:
            response: LLMResponse containing tool calls

        Returns:
            List of tool results with id, name, and output
        """
        if not response.has_tool_calls:
            return []

        results = []
        for tool_call in response.tool_calls:
            tool_id = tool_call["id"]
            function_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]

            # Execute the tool
            if function_name in self.tool_handlers:
                handler = self.tool_handlers[function_name]
                # Parse arguments (usually JSON string)
                import json
                args = json.loads(arguments) if isinstance(arguments, str) else arguments
                output = handler(**args)
            else:
                output = f"Error: Tool '{function_name}' not registered"

            results.append({
                "tool_call_id": tool_id,
                "name": function_name,
                "output": str(output)
            })

        return results

    def complete_with_tool_execution(
        self,
        messages: Union[str, List[Message]],
        max_iterations: int = 5,
        **kwargs
    ) -> LLMResponse:
        """
        Execute completion with automatic tool execution loop.

        Handles the full cycle: request -> tool calls -> execution -> next request
        until the model returns a final text response or max iterations reached.

        Args:
            messages: Initial messages
            max_iterations: Maximum tool execution rounds
            **kwargs: Override config parameters

        Returns:
            Final LLMResponse after all tool executions
        """
        # Convert to message list if needed
        if isinstance(messages, str):
            msg_list = [Message(role=LLMRole.USER, content=messages)]
        else:
            msg_list = list(messages)

        iteration = 0
        while iteration < max_iterations:
            # Get response
            response = self.complete_with_tools(msg_list, **kwargs)

            # If no tool calls, we're done
            if not response.has_tool_calls:
                return response

            # Add assistant's tool call message to conversation
            msg_list.append(Message(
                role=LLMRole.ASSISTANT,
                content=response.content or "",
                tool_calls=response.tool_calls
            ))

            # Execute tools and add results
            tool_results = self.execute_tool_calls(response)
            for result in tool_results:
                msg_list.append(Message(
                    role=LLMRole.TOOL,
                    content=result["output"],
                    tool_call_id=result["tool_call_id"],
                    name=result["name"]
                ))

            iteration += 1

        # Max iterations reached, return last response
        return response


# Convenience factory functions
def create_text_llm(model: str, **config_kwargs) -> BaseLLM:
    """Create a basic text completion LLM."""
    config = LLMConfig(model=model, **config_kwargs)
    return BaseLLM(config)


def create_tool_llm(model: str, tools: Optional[List[Dict[str, Any]]] = None, **config_kwargs) -> ToolCallingLLM:
    """Create a tool-calling LLM."""
    config = LLMConfig(model=model, **config_kwargs)
    return ToolCallingLLM(config, tools=tools)

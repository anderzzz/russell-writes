"""
Jinja template loader to read prompt templates and build them

Supports both Pydantic models (type-safe, recommended) and kwargs (flexible).
"""
from pathlib import Path
from typing import Any
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel


PROMPTS_PATH = (Path(__file__).parent.parent / "prompts").resolve()

class TemplateLoader:
    """Load and render Jinja2 templates from the prompts directory."""

    def __init__(self):
        """
        Initialize the template loader.

        """
        self.env = Environment(
            loader=FileSystemLoader(PROMPTS_PATH),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def render(
        self,
        template_name: str,
        config: BaseModel | None = None,
        **kwargs: Any
    ) -> str:
        """
        Load and render a template with the given variables.

        Two usage patterns:
        1. Type-safe with Pydantic models (recommended):
           config = SyntacticianConfig(include_sentence_structures=True)
           loader.render("syntactician", config=config)

        2. Flexible with kwargs (backwards compatible):
           loader.render("syntactician", include_sentence_structures=True)

        Args:
            template_name: Name of the template file (with or without .jinja extension)
            config: Optional Pydantic model containing validated template variables
            **kwargs: Variables to pass to template (alternative to config)

        Returns:
            Rendered template as a string

        Raises:
            ValueError: If both config and kwargs are provided, or if template_name
                       doesn't match config.template_name()
            pydantic.ValidationError: If config has invalid/missing fields

        """
        if not template_name.endswith(".jinja"):
            template_name += ".jinja"

        # Determine which variables to use
        if config is not None:
            if kwargs:
                raise ValueError(
                    "Cannot provide both 'config' and '**kwargs'. "
                    "Use either Pydantic model or kwargs, not both."
                )

            # Validate template name matches config
            expected_template = config.template_name() + ".jinja"
            if template_name != expected_template:
                raise ValueError(
                    f"Template mismatch: config is for '{expected_template}' "
                    f"but rendering '{template_name}'"
                )

            # Use Pydantic model's validated data
            template_vars = config.model_dump()
        else:
            # Use kwargs directly
            template_vars = kwargs

        template = self.env.get_template(template_name)
        return template.render(**template_vars)


if __name__ == "__main__":
    from models.prompt_models import (
        PreambleTextConfig,
        SyntacticianConfig,
        BasicPromptConfig,
    )
    from pydantic import ValidationError

    loader = TemplateLoader()

    print("=" * 70)
    print("EXAMPLE 1: Using Pydantic Models (Type-Safe, Recommended)")
    print("=" * 70)

    # Example 1a: PreambleTextConfig
    print("\n1a. PreambleTextConfig with valid data:")
    config = PreambleTextConfig(
        text_to_analyze="The quick brown fox jumps over the lazy dog."
    )
    result = loader.render(config.template_name(), config=config)
    print(result)

    # Example 1b: Missing required field
    print("\n1b. PreambleTextConfig with missing required field:")
    try:
        config = PreambleTextConfig()  # Missing text_to_analyze
    except ValidationError as e:
        print(f"✓ Pydantic validation error:\n{e}")

    # Example 1c: SyntacticianConfig with selective sections
    print("\n1c. SyntacticianConfig with selective sections:")
    config = SyntacticianConfig(
        include_sentence_structures=True,
        include_clause_architecture=True,
        include_grammatical_features=False,  # Disable this section
        include_functional_observations=False  # Disable this section
    )
    result = loader.render(config.template_name(), config=config)
    print(f"Rendered template length: {len(result)} characters")
    print(f"Sections enabled: sentence_structures, clause_architecture")

    # Example 1d: BasicPromptConfig with optional fields
    print("\n1d. BasicPromptConfig with optional instructions:")
    config = BasicPromptConfig(
        role="a Python expert",
        question="How do I use Pydantic for data validation?",
        context="I'm building a web API",
        instructions=["Be concise", "Provide code examples"]
    )
    result = loader.render(config.template_name(), config=config)
    print(result[:200] + "...")

    print("\n" + "=" * 70)
    print("EXAMPLE 2: Using kwargs (Flexible, Backwards Compatible)")
    print("=" * 70)

    # Example 2a: Using kwargs directly
    print("\n2a. Rendering with kwargs:")
    result = loader.render(
        "preamble_text",
        text_to_analyze="A different text passage."
    )
    print(result)

    # Example 2b: Template name mismatch detection
    print("\n2b. Template name mismatch error:")
    try:
        config = PreambleTextConfig(text_to_analyze="Some text")
        loader.render("syntactician", config=config)  # Wrong template!
    except ValueError as e:
        print(f"✓ Template mismatch error: {e}")

    # Example 2c: Cannot mix config and kwargs
    print("\n2c. Cannot mix config and kwargs:")
    try:
        config = PreambleTextConfig(text_to_analyze="Some text")
        loader.render("preamble_text", config=config, extra_arg="oops")
    except ValueError as e:
        print(f"✓ Mixed usage error: {e}")

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)

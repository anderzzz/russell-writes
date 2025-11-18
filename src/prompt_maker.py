"""
Prompt maker that constructs prompt snippets from Jinja templates.

Uses Pydantic models for type-safe, validated prompt construction.
"""
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from models.prompt_models import BasePromptConfig


PROMPTS_PATH = (Path(__file__).parent.parent / "prompts").resolve()

class PromptMaker:
    """Constructs prompt snippets from Jinja templates using Pydantic models."""

    def __init__(self):
        """Initialize the prompt maker with Jinja environment."""
        self.env = Environment(
            loader=FileSystemLoader(PROMPTS_PATH),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def render(self, prompt_model: BasePromptConfig) -> str:
        """
        Render a prompt snippet from a Pydantic model.

        Args:
            prompt_model: Pydantic model containing validated template variables

        Returns:
            Rendered prompt snippet as a string

        Raises:
            pydantic.ValidationError: If model has invalid/missing fields
            jinja2.TemplateNotFound: If template file doesn't exist

        Example:
            maker = PromptMaker()
            config = PreambleTextConfig(text_to_analyze="Sample text")
            prompt = maker.render(config)
        """
        template_name = prompt_model.template_name() + ".jinja"
        template_vars = prompt_model.model_dump()

        template = self.env.get_template(template_name)
        return template.render(**template_vars)


if __name__ == "__main__":
    from models.prompt_models import (
        PreambleTextConfig,
        SyntacticianConfig,
        BasicPromptConfig,
        EfficiencyAuditorConfig,
    )
    from pydantic import ValidationError

    maker = PromptMaker()

    print("=" * 70)
    print("PromptMaker Examples - Type-Safe Prompt Construction")
    print("=" * 70)

    # Example 1: Basic usage with required field
    print("\n1. PreambleTextConfig with required field:")
    model = PreambleTextConfig(
        text_to_analyze="The quick brown fox jumps over the lazy dog."
    )
    prompt = maker.render(model)
    print(prompt)

    # Example 2: Validation - missing required field
    print("\n2. Validation catches missing required field:")
    try:
        model = PreambleTextConfig()  # Missing text_to_analyze
    except ValidationError as e:
        print(f"✓ Pydantic validation error:")
        print(f"  Field 'text_to_analyze' is required\n")

    # Example 3: Specialist with selective sections
    print("3. SyntacticianConfig with selective sections:")
    model = SyntacticianConfig(
        include_sentence_structures=True,
        include_clause_architecture=True,
        include_grammatical_features=False,  # Disabled
        include_functional_observations=False  # Disabled
    )
    prompt = maker.render(model)
    print(f"  Rendered {len(prompt)} characters")
    print(f"  Sections: 1. Sentence Structures, 2. Clause Architecture")
    print(f"  (grammatical_features and functional_observations omitted)")

    # Example 4: Complex model with optional fields
    print("\n4. BasicPromptConfig with optional context and instructions:")
    model = BasicPromptConfig(
        role="a Python expert",
        question="How do I use Pydantic for data validation?",
        context="I'm building a web API",
        instructions=["Be concise", "Provide code examples", "Include error handling"]
    )
    prompt = maker.render(model)
    print(prompt[:250] + "...")

    # Example 5: All sections enabled (default behavior)
    print("\n5. EfficiencyAuditorConfig with all sections (default=True):")
    model = EfficiencyAuditorConfig()  # All includes default to True
    prompt = maker.render(model)
    print(f"  Rendered {len(prompt)} characters")
    print(f"  All 4 analytical sections included by default")

    # Example 6: Type safety demonstration
    print("\n6. Type safety - IDE autocomplete and validation:")
    print("  model = PreambleTextConfig(")
    print("      text_to_analyze=\"...\"  # ← IDE knows this field exists")
    print("  )")
    print("  prompt = maker.render(model)  # ← Type-checked by mypy")

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("Simple API: maker.render(prompt_model)")
    print("=" * 70)

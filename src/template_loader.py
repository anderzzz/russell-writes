"""
Simple Jinja2 template loader for prompts.
"""
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template


class TemplateLoader:
    """Load and render Jinja2 templates from the prompts directory."""

    def __init__(self, template_dir: str = "prompts"):
        """
        Initialize the template loader.

        Args:
            template_dir: Directory containing template files
        """
        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def load(self, template_name: str) -> Template:
        """
        Load a template by name.

        Args:
            template_name: Name of the template file (e.g., 'basic_prompt.jinja')

        Returns:
            Jinja2 Template object
        """
        return self.env.get_template(template_name)

    def render(self, template_name: str, **kwargs) -> str:
        """
        Load and render a template with the given variables.

        Args:
            template_name: Name of the template file
            **kwargs: Variables to pass to the template

        Returns:
            Rendered template as a string
        """
        template = self.load(template_name)
        return template.render(**kwargs)


# Example usage
if __name__ == "__main__":
    loader = TemplateLoader()

    # Example 1: Simple rendering
    result = loader.render(
        "basic_prompt.jinja",
        role="a helpful AI assistant",
        question="What is the capital of France?",
        context="This is a geography quiz.",
        instructions=["Be concise", "Provide only factual information"]
    )

    print(result)
    print("\n" + "="*50 + "\n")

    # Example 2: Without optional fields
    result2 = loader.render(
        "basic_prompt.jinja",
        role="a coding expert",
        question="How do I use Jinja templates in Python?"
    )

    print(result2)

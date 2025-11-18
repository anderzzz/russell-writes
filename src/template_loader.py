"""
Jinja template loader to read prompt templates and build them

"""
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template


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

    def render(self, template_name: str, **kwargs) -> str:
        """
        Load and render a template with the given variables.

        Args:
            template_name: Name of the template file
            **kwargs: Variables to pass to the template

        Returns:
            Rendered template as a string

        """
        if not template_name.endswith(".jinja"):
            template_name += ".jinja"
        template = self.env.get_template(template_name)
        return template.render(**kwargs)


if __name__ == "__main__":
    loader = TemplateLoader()
    x =loader.render("basic_prompt.jinja")
    print(x)

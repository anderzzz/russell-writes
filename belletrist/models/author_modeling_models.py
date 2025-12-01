"""
Pydantic models for author-modeling workflow.

This module implements a fundamentally different approach from the traditional
style decomposition pipeline. Rather than extracting techniques to codify in rules,
this workflow:

1. Models the author's decision-making patterns and sensibility
2. Curates exemplary passages based on density of characteristic qualities
3. Transmits style tacitly through few-shot demonstration

The analytical work serves curation, not prescription. The generating LLM never
sees explicit rules - only carefully selected examples.

Workflow:
    Stage 1: Analytical Mining (per-text)
        - 1A: Implied Author Construction
        - 1B: Decision Pattern Analysis
        - 1C: Functional Texture Analysis

    Stage 2: Cross-Text Synthesis
        - 2A: Implied Author Synthesis
        - 2B: Decision Pattern Synthesis
        - 2C: Textural Synthesis

    Stage 3: Integration & Field Guide Construction
        - Unified sensibility description
        - Recognition criteria & density rubric
        - Master passage index

    Stage 4: Corpus Mining
        - Evaluate passages for density (1-5)
        - Assess compositional task coverage

    Stage 5: Example Set Construction
        - Curate 3-4 high-density passages
        - Ensure diversity of tasks/topics

    [Stage 6: Generation & Calibration - future work]
"""
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from .prompt_models import BasePromptConfig


# =============================================================================
# Stage 1: Analytical Mining (Per-Text)
# =============================================================================

class ImpliedAuthorConfig(BasePromptConfig):
    """
    Configuration for implied_author.jinja - Stage 1A.

    Constructs a portrait of the implied author from a single text. Analyzes:
    - Relationship to material (curiosity, mastery, struggle)
    - Relationship to reader (collegial, pedagogical, adversarial)
    - Relationship to uncertainty (qualification, confidence)
    - Relationship to writing itself (pleasure, transparency, craft)

    Output: Structured portrait with quoted evidence, signature moments,
            and synthetic paragraph capturing the sensibility.
    """

    text: str = Field(
        ...,
        min_length=1,
        description="The text to analyze for implied author"
    )

    @classmethod
    def template_name(cls) -> str:
        return "implied_author"

    @classmethod
    def analyst_name(cls) -> str:
        """Return the analyst identifier for storing in ResultStore."""
        return "implied_author"


class DecisionPatternConfig(BasePromptConfig):
    """
    Configuration for decision_pattern.jinja - Stage 1B.

    Analyzes compositional decisions visible in the text. For each decision:
    - What problem the author faced
    - What choice they made
    - What alternatives were available
    - What the choice achieves

    Output: Sequential decision analysis, recurring patterns, and
            high-density passages where multiple decisions converge.
    """

    text: str = Field(
        ...,
        min_length=1,
        description="The text to analyze for decision patterns"
    )

    @classmethod
    def template_name(cls) -> str:
        return "decision_pattern"

    @classmethod
    def analyst_name(cls) -> str:
        """Return the analyst identifier for storing in ResultStore."""
        return "decision_pattern"


class FunctionalTextureConfig(BasePromptConfig):
    """
    Configuration for functional_texture.jinja - Stage 1C.

    Analyzes how surface features serve communicative purposes:
    - Sentence-level patterns and their functions
    - Lexical patterns and precision/accessibility balance
    - Rhythmic patterns and emphasis
    - Paragraph-level structure

    Output: Functional analysis of formal features with texturally
            distinctive passages flagged.
    """

    text: str = Field(
        ...,
        min_length=1,
        description="The text to analyze for functional texture"
    )

    @classmethod
    def template_name(cls) -> str:
        return "functional_texture"

    @classmethod
    def analyst_name(cls) -> str:
        """Return the analyst identifier for storing in ResultStore."""
        return "functional_texture"


# =============================================================================
# Stage 2: Cross-Text Synthesis
# =============================================================================

class ImpliedAuthorSynthesisConfig(BasePromptConfig):
    """
    Configuration for implied_author_synthesis.jinja - Stage 2A.

    Synthesizes implied author portraits from multiple texts into:
    - The stable core (constant across texts)
    - Productive tensions (dimensions of variation)
    - Synthetic portrait (300-400 word unified description)
    - Recognition markers (telltale signs of this author)

    Requires at least 2 implied author analyses.
    """

    implied_author_analyses: dict[str, str] = Field(
        ...,
        description="Mapping of sample IDs to their implied author analyses"
    )

    @field_validator('implied_author_analyses')
    @classmethod
    def validate_analyses(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate at least 2 analyses provided."""
        if len(v) < 2:
            raise ValueError(
                "At least 2 implied author analyses required for synthesis. "
                f"Got {len(v)}."
            )

        for sample_id, analysis in v.items():
            if not analysis.strip():
                raise ValueError(f"Implied author analysis for '{sample_id}' is empty")

        return v

    @classmethod
    def template_name(cls) -> str:
        return "implied_author_synthesis"

    @classmethod
    def synthesis_type(cls) -> str:
        """Return synthesis type identifier for ResultStore."""
        return "implied_author_synthesis"


class DecisionPatternSynthesisConfig(BasePromptConfig):
    """
    Configuration for decision_pattern_synthesis.jinja - Stage 2B.

    Synthesizes decision patterns from multiple texts into:
    - Compositional problem types taxonomy
    - Characteristic solutions for each problem type
    - Signature moves (4-6 fingerprint techniques)
    - Anti-patterns (what author avoids)
    - High-density passage index

    Requires at least 2 decision pattern analyses.
    """

    decision_pattern_analyses: dict[str, str] = Field(
        ...,
        description="Mapping of sample IDs to their decision pattern analyses"
    )

    @field_validator('decision_pattern_analyses')
    @classmethod
    def validate_analyses(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate at least 2 analyses provided."""
        if len(v) < 2:
            raise ValueError(
                "At least 2 decision pattern analyses required for synthesis. "
                f"Got {len(v)}."
            )

        for sample_id, analysis in v.items():
            if not analysis.strip():
                raise ValueError(f"Decision pattern analysis for '{sample_id}' is empty")

        return v

    @classmethod
    def template_name(cls) -> str:
        return "decision_pattern_synthesis"

    @classmethod
    def synthesis_type(cls) -> str:
        """Return synthesis type identifier for ResultStore."""
        return "decision_pattern_synthesis"


class TexturalSynthesisConfig(BasePromptConfig):
    """
    Configuration for textural_synthesis.jinja - Stage 2C.

    Synthesizes textural analyses from multiple texts into:
    - Characteristic sentence architecture
    - Lexical character and word choice patterns
    - Paragraph shape and sequence patterns
    - Texture-function integration
    - Texturally exemplary passages

    Requires at least 2 functional texture analyses.
    """

    textural_analyses: dict[str, str] = Field(
        ...,
        description="Mapping of sample IDs to their functional texture analyses"
    )

    @field_validator('textural_analyses')
    @classmethod
    def validate_analyses(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate at least 2 analyses provided."""
        if len(v) < 2:
            raise ValueError(
                "At least 2 textural analyses required for synthesis. "
                f"Got {len(v)}."
            )

        for sample_id, analysis in v.items():
            if not analysis.strip():
                raise ValueError(f"Textural analysis for '{sample_id}' is empty")

        return v

    @classmethod
    def template_name(cls) -> str:
        return "textural_synthesis"

    @classmethod
    def synthesis_type(cls) -> str:
        """Return synthesis type identifier for ResultStore."""
        return "textural_synthesis"


# =============================================================================
# Stage 3: Integration & Field Guide Construction
# =============================================================================

class FieldGuideConstructionConfig(BasePromptConfig):
    """
    Configuration for field_guide_construction.jinja - Stage 3.

    Integrates the three Stage 2 syntheses into a unified recognition field guide:
    - Part 1: Unified sensibility description (400-500 words)
    - Part 2: Recognition criteria (questions for evaluation)
    - Part 3: Density evaluation rubric (1-5 scale)
    - Part 4: Master passage index (all flagged passages)

    This field guide enables passage evaluation in Stage 4.
    """

    implied_author_synthesis: str = Field(
        ...,
        min_length=1,
        description="Output from Stage 2A (implied author synthesis)"
    )
    decision_pattern_synthesis: str = Field(
        ...,
        min_length=1,
        description="Output from Stage 2B (decision pattern synthesis)"
    )
    textural_synthesis: str = Field(
        ...,
        min_length=1,
        description="Output from Stage 2C (textural synthesis)"
    )

    @classmethod
    def template_name(cls) -> str:
        return "field_guide_construction"

    @classmethod
    def synthesis_type(cls) -> str:
        """Return synthesis type identifier for ResultStore."""
        return "field_guide"


class AuthorModelDefinitionConfig(BasePromptConfig):
    """
    Configuration for author_model_definition.jinja - Stage 3 (Generation-Oriented).

    Integrates the three Stage 2 syntheses into a unified Author Model Definition:
    - Part 1: Unified author model (500-700 words, prescriptive)
    - Part 2: Generative guidelines (actionable instructions by task)
    - Part 3: Few-shot exemplars (8-12 annotated passages)
    - Part 4: Implementation guidance (intensity modulation, pitfalls)

    This definition enables direct style generation without further processing.
    It replaces the recognition-oriented field guide for generation use cases.
    """

    implied_author_synthesis: str = Field(
        ...,
        min_length=1,
        description="Output from Stage 2A (implied author synthesis)"
    )
    decision_pattern_synthesis: str = Field(
        ...,
        min_length=1,
        description="Output from Stage 2B (decision pattern synthesis)"
    )
    textural_synthesis: str = Field(
        ...,
        min_length=1,
        description="Output from Stage 2C (textural synthesis)"
    )

    @classmethod
    def template_name(cls) -> str:
        return "author_model_definition"

    @classmethod
    def synthesis_type(cls) -> str:
        """Return synthesis type identifier for ResultStore."""
        return "author_model_definition"


# =============================================================================
# Stage 4: Corpus Mining
# =============================================================================

class PassageEvaluationConfig(BasePromptConfig):
    """
    Configuration for passage_evaluation.jinja - Stage 4.

    Evaluates a single passage for suitability as a few-shot example:
    - Recognition criteria assessment (sensibility, decisions, texture)
    - Density rating (1-5 using rubric from field guide)
    - Compositional task coverage
    - Teaching value assessment
    - Summary recommendation

    Used to evaluate every candidate passage in the corpus.
    """

    field_guide: str = Field(
        ...,
        min_length=1,
        description="The field guide from Stage 3"
    )
    passage: str = Field(
        ...,
        min_length=1,
        description="The passage to evaluate"
    )
    source: str = Field(
        ...,
        min_length=1,
        description="Source text identifier (e.g., 'sample_003, paragraphs 5-7')"
    )

    @classmethod
    def template_name(cls) -> str:
        return "passage_evaluation"


# =============================================================================
# Stage 5: Example Set Construction
# =============================================================================

class ExampleSetConstructionConfig(BasePromptConfig):
    """
    Configuration for example_set_construction.jinja - Stage 5.

    Curates optimal few-shot example set from evaluated passages:
    - Selection criteria (density, diversity, task coverage)
    - Primary example set (3-4 passages with justifications)
    - Set coherence assessment
    - Alternative/supplementary passages
    - Set variants for different purposes

    Requires field guide and all passage evaluations from Stage 4.
    """

    field_guide: str = Field(
        ...,
        min_length=1,
        description="The field guide from Stage 3"
    )
    passage_evaluations: list[str] = Field(
        ...,
        min_length=5,
        description="All passage evaluations from Stage 4 (minimum 5 for selection)"
    )

    @field_validator('passage_evaluations')
    @classmethod
    def validate_evaluations(cls, v: list[str]) -> list[str]:
        """Validate minimum number of evaluations."""
        if len(v) < 5:
            raise ValueError(
                "At least 5 passage evaluations required for meaningful example set construction. "
                f"Got {len(v)}."
            )
        return v

    @classmethod
    def template_name(cls) -> str:
        return "example_set_construction"

    @classmethod
    def synthesis_type(cls) -> str:
        """Return synthesis type identifier for ResultStore."""
        return "example_set"


# =============================================================================
# Quote Extraction for Nominated Passages
# =============================================================================

class ExtractedQuote(BaseModel):
    """A single quote extracted from an analysis."""

    quote_text: str = Field(
        ...,
        min_length=10,
        description="The quoted passage text (1-3 sentences typical)"
    )
    context_type: Literal[
        "signature_moment",
        "high_density_decision",
        "textural_exemplar",
        "evidence_quote"
    ] = Field(
        ...,
        description="Type of quote based on its role in the analysis"
    )
    rationale: str = Field(
        ...,
        min_length=10,
        description="Why this passage was highlighted in the analysis"
    )


class QuoteExtractionResponse(BaseModel):
    """Response from quote extraction LLM call."""

    quotes: list[ExtractedQuote] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="Quotes extracted from the analysis"
    )
    analysis_type: str = Field(
        ...,
        description="Type of analysis (implied_author, decision_pattern, functional_texture)"
    )


class QuoteExtractionConfig(BasePromptConfig):
    """
    Configuration for extract_quotes.jinja.

    Extracts quoted passages from Stage 1-2 analyses using LLM with JSON mode.
    Uses quotes rather than paragraph indices since LLMs reliably quote text
    but unreliably provide accurate indices.
    """

    analysis_text: str = Field(
        ...,
        min_length=100,
        description="The analysis text containing quotes to extract"
    )
    analysis_type: Literal[
        "implied_author",
        "decision_pattern",
        "functional_texture"
    ] = Field(
        ...,
        description="Type of analysis being processed"
    )

    @classmethod
    def template_name(cls) -> str:
        return "extract_quotes"

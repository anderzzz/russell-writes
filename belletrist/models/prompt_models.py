"""
Pydantic models for prompt templates.

Each model corresponds to a Jinja template in the prompts/ directory,
providing type-safe validation and clear documentation of required variables.
"""
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Base Models
# =============================================================================

class BasePromptConfig(BaseModel, ABC):
    """Abstract base class for all prompt configurations."""

    @classmethod
    @abstractmethod
    def template_name(cls) -> str:
        """Return the name of the Jinja template file (without .jinja extension)."""
        pass

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Prevent accidental extra fields


class SpecialistAnalystConfig(BasePromptConfig, ABC):
    """
    Base class for specialist analyst templates.

    All specialist analysts have 4 optional boolean sections that can be
    enabled/disabled. By default, all sections are enabled.
    """

    @classmethod
    @abstractmethod
    def description(cls) -> str:
        """Return a brief description of this analyst's focus area."""
        pass

    @classmethod
    @abstractmethod
    def display_name(cls) -> str:
        """Return the formatted display name for this analyst."""
        pass


# =============================================================================
# Foundational Template Models
# =============================================================================

class BasicPromptConfig(BasePromptConfig):
    """Configuration for basic_prompt.jinja - generic role-based prompts."""

    role: str = Field(
        ...,
        min_length=1,
        description="The role or persona for the AI to adopt"
    )
    context: str | None = Field(
        None,
        description="Optional context information to provide background"
    )
    question: str = Field(
        ...,
        min_length=1,
        description="The main question or task to address"
    )
    instructions: list[str] = Field(
        default_factory=list,
        description="Optional list of specific instructions to follow"
    )

    @classmethod
    def template_name(cls) -> str:
        return "basic_prompt"


class PreambleTextConfig(BasePromptConfig):
    """Configuration for preamble_text.jinja - text passage container."""

    text_to_analyze: str = Field(
        ...,
        min_length=1,
        description="The prose text to be analyzed"
    )

    @classmethod
    def template_name(cls) -> str:
        return "preamble_text"


class PreambleInstructionConfig(BasePromptConfig):
    """
    Configuration for preamble_instruction.jinja - static preamble.

    This template has no variables; it's a static instructional preamble
    explaining the analytical task.
    """

    @classmethod
    def template_name(cls) -> str:
        return "preamble_instruction"


# =============================================================================
# Specialist Analyst Models
# =============================================================================

class SyntacticianConfig(SpecialistAnalystConfig):
    """Configuration for syntactician.jinja - syntax and sentence structure analysis."""

    include_sentence_structures: bool = Field(
        True,
        description="Analyze sentence length, types, and variety"
    )
    include_clause_architecture: bool = Field(
        True,
        description="Analyze clause relationships and dependencies"
    )
    include_grammatical_features: bool = Field(
        True,
        description="Analyze voice, mood, tense, and aspect"
    )
    include_functional_observations: bool = Field(
        True,
        description="Analyze how syntax serves meaning and effect"
    )

    @classmethod
    def template_name(cls) -> str:
        return "syntactician"

    @classmethod
    def description(cls) -> str:
        return "Sentence structure, clause architecture, grammatical patterns"

    @classmethod
    def display_name(cls) -> str:
        return "Syntactician"


class LexicologistConfig(SpecialistAnalystConfig):
    """Configuration for lexicologist.jinja - vocabulary and diction analysis."""

    include_lexical_register: bool = Field(
        True,
        description="Analyze formality and word choice patterns"
    )
    include_semantic_fields: bool = Field(
        True,
        description="Analyze word meanings and conceptual groupings"
    )
    include_precision_analysis: bool = Field(
        True,
        description="Analyze specificity and exactness of word choice"
    )
    include_clarity_mechanisms: bool = Field(
        True,
        description="Analyze how vocabulary achieves clarity"
    )

    @classmethod
    def template_name(cls) -> str:
        return "lexicologist"

    @classmethod
    def description(cls) -> str:
        return "Word choice, register, etymology, semantic fields"

    @classmethod
    def display_name(cls) -> str:
        return "Lexicologist"


class InformationArchitectConfig(SpecialistAnalystConfig):
    """Configuration for information_architect.jinja - information structure analysis."""

    include_paragraph_architecture: bool = Field(
        True,
        description="Analyze paragraph structure and organization"
    )
    include_coherence_mechanisms: bool = Field(
        True,
        description="Analyze how ideas connect and relate"
    )
    include_logical_progression: bool = Field(
        True,
        description="Analyze the sequence and development of ideas"
    )
    include_transitions: bool = Field(
        True,
        description="Analyze transition techniques and flow"
    )

    @classmethod
    def template_name(cls) -> str:
        return "information_architect"

    @classmethod
    def description(cls) -> str:
        return "Logical flow, coherence, information structure"

    @classmethod
    def display_name(cls) -> str:
        return "Information Architect"


class RhetoricianConfig(SpecialistAnalystConfig):
    """Configuration for rhetorician.jinja - rhetorical strategy analysis."""

    include_writer_position: bool = Field(
        True,
        description="Analyze the writer's stance and voice"
    )
    include_reader_positioning: bool = Field(
        True,
        description="Analyze how the text positions the reader"
    )
    include_persuasive_techniques: bool = Field(
        True,
        description="Analyze persuasive devices and appeals"
    )
    include_argumentative_moves: bool = Field(
        True,
        description="Analyze argumentative structure and logic"
    )

    @classmethod
    def template_name(cls) -> str:
        return "rhetorician"

    @classmethod
    def description(cls) -> str:
        return "Persuasive strategy, tone, reader positioning"

    @classmethod
    def display_name(cls) -> str:
        return "Rhetorician"


class EfficiencyAuditorConfig(SpecialistAnalystConfig):
    """Configuration for efficiency_auditor.jinja - economy and compression analysis."""

    include_word_economy: bool = Field(
        True,
        description="Analyze conciseness and word efficiency"
    )
    include_structural_efficiency: bool = Field(
        True,
        description="Analyze sentence and paragraph efficiency"
    )
    include_density_analysis: bool = Field(
        True,
        description="Analyze information density and payload"
    )
    include_subtraction_test: bool = Field(
        True,
        description="Analyze necessity of each element"
    )

    @classmethod
    def template_name(cls) -> str:
        return "efficiency_auditor"

    @classmethod
    def description(cls) -> str:
        return "Economy, necessity, compression"

    @classmethod
    def display_name(cls) -> str:
        return "Efficiency Auditor"


# =============================================================================
# Integration & Synthesis Models
# =============================================================================

class CrossPerspectiveIntegratorConfig(BasePromptConfig):
    """
    Configuration for cross_perspective_integrator.jinja - cross-perspective integration.

    This template integrates multiple specialist analyses of a single text
    to identify unified patterns. Requires at least 2 analysts.

    The analysts dict should have structure:
    {
        'analyst_key': {
            'analysis': 'The full analysis text from this analyst',
            'analyst_descr_short': 'Brief description of analyst focus'
        },
        ...
    }
    """

    original_text: str = Field(
        ...,
        min_length=1,
        description="The original text being analyzed"
    )
    analysts: dict[str, dict[str, str]] = Field(
        ...,
        description="Dictionary mapping analyst keys to their analysis and description"
    )

    @field_validator('analysts')
    @classmethod
    def validate_analysts(cls, v: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
        """Validate that at least 2 analysts are provided with required keys."""
        if len(v) < 2:
            raise ValueError("At least 2 analysts are required for cross-perspective integration")

        required_keys = {'analysis', 'analyst_descr_short'}
        for analyst_key, analyst_data in v.items():
            missing_keys = required_keys - set(analyst_data.keys())
            if missing_keys:
                raise ValueError(
                    f"Analyst '{analyst_key}' missing required keys: {missing_keys}. "
                    f"Each analyst must have 'analysis' and 'analyst_descr_short'."
                )
            if not analyst_data['analysis'].strip():
                raise ValueError(f"Analyst '{analyst_key}' has empty analysis")
            if not analyst_data['analyst_descr_short'].strip():
                raise ValueError(f"Analyst '{analyst_key}' has empty description")

        return v

    @classmethod
    def template_name(cls) -> str:
        return "cross_perspective_integrator"

    @classmethod
    def analyst_name(cls) -> str:
        """Return the analyst identifier for storing results in ResultStore."""
        return "cross_perspective_integrator"


class CrossTextSynthesizerConfig(BasePromptConfig):
    """
    Configuration for cross_text_synthesizer.jinja - cross-text synthesis.

    This template synthesizes patterns across multiple text analyses
    to extract generalizable principles. Requires at least 2 integrated analyses.

    The integrated_analyses dict should map sample IDs to their cross-perspective
    integration outputs:
    {
        'sample_001': 'The integrated analysis for sample 001...',
        'sample_002': 'The integrated analysis for sample 002...',
        ...
    }
    """

    integrated_analyses: dict[str, str] = Field(
        ...,
        description="Mapping of sample IDs to their cross-perspective integration analyses"
    )

    @field_validator('integrated_analyses')
    @classmethod
    def validate_integrated_analyses(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate that at least 2 integrated analyses are provided."""
        if len(v) < 2:
            raise ValueError(
                "At least 2 integrated analyses are required for cross-text synthesis. "
                f"Got {len(v)}."
            )

        for sample_id, analysis in v.items():
            if not analysis.strip():
                raise ValueError(f"Integrated analysis for '{sample_id}' is empty")

        return v

    @classmethod
    def template_name(cls) -> str:
        return "cross_text_synthesizer"


class SynthesizerOfPrinciplesConfig(BasePromptConfig):
    """
    Configuration for synthesizer_of_principles.jinja - prescriptive guide generation.

    This template converts descriptive pattern analyses into actionable
    writing principles and style guidelines.
    """

    synthesis_document: str = Field(
        ...,
        min_length=1,
        description="The complete Stage 2 cross-text synthesis document"
    )

    @classmethod
    def template_name(cls) -> str:
        return "synthesizer_of_principles"

"""
Pydantic models for prompt templates.

Each model corresponds to a Jinja template in the prompts/ directory,
providing type-safe validation and clear documentation of required variables.
"""
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


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


# =============================================================================
# Integration & Synthesis Models
# =============================================================================

class PatternRecognizerTextConfig(BasePromptConfig):
    """
    Configuration for pattern_recognizer_text.jinja - cross-perspective integration.

    This template integrates multiple specialist analyses of a single text
    to identify unified patterns.
    """

    original_text: str = Field(
        ...,
        min_length=1,
        description="The original text being analyzed"
    )
    specialist_analyses: str = Field(
        ...,
        min_length=1,
        description="Combined output from all 6 specialist analyses"
    )

    @classmethod
    def template_name(cls) -> str:
        return "pattern_recognizer_text"


class PatternRecognizerCrossAnalystConfig(BasePromptConfig):
    """
    Configuration for pattern_recognizer_cross_analyst.jinja - cross-text synthesis.

    This template synthesizes patterns across multiple text analyses
    to extract generalizable principles.
    """

    text_count: int = Field(
        ...,
        ge=1,
        description="Number of texts analyzed"
    )
    integrated_analyses: str = Field(
        ...,
        min_length=1,
        description="Combined pattern recognition analyses from all texts"
    )

    @classmethod
    def template_name(cls) -> str:
        return "pattern_recognizer_cross_analyst"


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

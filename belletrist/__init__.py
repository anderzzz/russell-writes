from belletrist.models import (
    # LLM core
    LLMConfig,
    Message,
    LLMRole,
    LLMResponse,

    # Author modeling models
    ImpliedAuthorConfig,
    DecisionPatternConfig,
    FunctionalTextureConfig,
    ImpliedAuthorSynthesisConfig,
    DecisionPatternSynthesisConfig,
    TexturalSynthesisConfig,
    FieldGuideConstructionConfig,
    PassageEvaluationConfig,
    ExampleSetConstructionConfig,
)
from belletrist.tools import (
    Tool,
    WordCountTool,
)
from belletrist.utils import (
    # Passage extraction
    extract_paragraph_windows,
    extract_logical_sections,
    get_full_sample_as_passage,
    extract_passages_by_indices,
    # Evaluation parsing
    parse_passage_evaluation,
    parse_example_set_selection,
    extract_selection_criteria,
    extract_coherence_assessment,
    extract_alternative_passages,
    validate_passage_evaluation,
    validate_example_set_selection,
)

# Expose commonly-used classes at top level for convenience
from belletrist.llm import LLM, ToolLLM
from belletrist.prompt_maker import PromptMaker
from belletrist.data_sampler import DataSampler
from belletrist.result_store import ResultStore
from belletrist.style_evaluation_store import StyleEvaluationStore

__all__ = [
    # LLM core
    "LLMConfig", "Message", "LLMRole", "LLMResponse",
    "LLM", "ToolLLM",

    # Tools
    "Tool", "WordCountTool",

    # Core utilities
    "PromptMaker", "DataSampler", "ResultStore", "StyleEvaluationStore",

    # Passage extraction utilities
    "extract_paragraph_windows",
    "extract_logical_sections",
    "get_full_sample_as_passage",
    "extract_passages_by_indices",

    # Evaluation parsing utilities
    "parse_passage_evaluation",
    "parse_example_set_selection",
    "extract_selection_criteria",
    "extract_coherence_assessment",
    "extract_alternative_passages",
    "validate_passage_evaluation",
    "validate_example_set_selection",

    # Author modeling models
    "ImpliedAuthorConfig",
    "DecisionPatternConfig",
    "FunctionalTextureConfig",
    "ImpliedAuthorSynthesisConfig",
    "DecisionPatternSynthesisConfig",
    "TexturalSynthesisConfig",
    "FieldGuideConstructionConfig",
    "PassageEvaluationConfig",
    "ExampleSetConstructionConfig",
]

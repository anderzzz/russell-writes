from belletrist.models.llm_config_models import (
    LLMConfig,
    Message,
    LLMRole,
    LLMResponse
)
from belletrist.models.style_evaluation_models import (
    MethodMapping,
    StyleJudgmentComparative
)
from belletrist.models.prompt_models import (
    # Base classes
    BasePromptConfig,
    SpecialistAnalystConfig,

    # Foundational templates
    BasicPromptConfig,
    PreambleTextConfig,
    PreambleInstructionConfig,

    # Specialist analysts
    SyntacticianConfig,
    LexicologistConfig,
    InformationArchitectConfig,
    RhetoricianConfig,
    EfficiencyAuditorConfig,

    # Integration & synthesis
    CrossPerspectiveIntegratorConfig,
    CrossTextSynthesizerConfig,
    SynthesizerOfPrinciplesConfig,

    # Style evaluation
    StyleFlatteningConfig,
    StyleFlatteningAggressiveConfig,
    StyleReconstructionGenericConfig,
    StyleReconstructionFewShotConfig,
    StyleReconstructionAuthorConfig,
    StyleReconstructionInstructionsConfig,
    StyleJudgeComparativeConfig,
)
from belletrist.models.author_modeling_models import (
    # Stage 1: Analytical Mining
    ImpliedAuthorConfig,
    DecisionPatternConfig,
    FunctionalTextureConfig,

    # Stage 2: Cross-Text Synthesis
    ImpliedAuthorSynthesisConfig,
    DecisionPatternSynthesisConfig,
    TexturalSynthesisConfig,

    # Stage 3: Field Guide Construction
    FieldGuideConstructionConfig,

    # Stage 4: Corpus Mining
    PassageEvaluationConfig,

    # Stage 5: Example Set Construction
    ExampleSetConstructionConfig,
)

__all__ = [
    # LLM core
    "LLMConfig", "Message", "LLMRole", "LLMResponse",

    # Style evaluation
    "MethodMapping", "StyleJudgmentComparative",

    # Base classes
    "BasePromptConfig", "SpecialistAnalystConfig",

    # Foundational
    "BasicPromptConfig", "PreambleTextConfig", "PreambleInstructionConfig",

    # Specialists
    "SyntacticianConfig", "LexicologistConfig", "InformationArchitectConfig",
    "RhetoricianConfig", "EfficiencyAuditorConfig",

    # Synthesis
    "CrossPerspectiveIntegratorConfig", "CrossTextSynthesizerConfig",
    "SynthesizerOfPrinciplesConfig",

    # Style evaluation
    "StyleFlatteningConfig", "StyleFlatteningAggressiveConfig",
    "StyleReconstructionGenericConfig", "StyleReconstructionFewShotConfig",
    "StyleReconstructionAuthorConfig", "StyleReconstructionInstructionsConfig",
    "StyleJudgeComparativeConfig",

    # Author modeling - Stage 1
    "ImpliedAuthorConfig", "DecisionPatternConfig", "FunctionalTextureConfig",

    # Author modeling - Stage 2
    "ImpliedAuthorSynthesisConfig", "DecisionPatternSynthesisConfig",
    "TexturalSynthesisConfig",

    # Author modeling - Stage 3
    "FieldGuideConstructionConfig",

    # Author modeling - Stage 4
    "PassageEvaluationConfig",

    # Author modeling - Stage 5
    "ExampleSetConstructionConfig",
]
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Russell Writes** is a multi-stage text analysis framework that analyzes literary texts (primarily Bertrand Russell's essays) through multiple specialist analytical lenses, synthesizes findings, and evaluates style reconstruction capabilities.

Core architecture: Decompose text analysis into specialized perspectives → run independently via LLM prompts → integrate for holistic understanding → synthesize cross-text patterns → evaluate style reconstruction.

## Development Commands

### Testing
```bash
# Run the main analysis pipeline
jupyter notebook test.ipynb

# Run style evaluation
jupyter notebook style_evaluation.ipynb

# Test LLM functionality directly
python belletrist/llm.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

Required environment variables:
- `MISTRAL_API_KEY` or `OPENAI_API_KEY` (depending on model choice)
- Any other LiteLLM-supported provider API key

## High-Level Architecture

### Four-Layer Design

```
┌─────────────────────────────────────────┐
│  Data Layer                             │
│  DataSampler + ResultStore (SQLite)     │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  LLM Interface                          │
│  LLM/ToolLLM → LiteLLM abstraction      │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Prompt Engineering                     │
│  PromptMaker + Pydantic + Jinja         │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Analysis Pipeline (Jupyter)            │
│  Multi-stage specialist → synthesis     │
└─────────────────────────────────────────┘
```

### Multi-Stage Analysis Pipeline

**Main Pipeline (test.ipynb):**

1. **Data Sampling**: Use `DataSampler` to extract text segments with full provenance (file index, paragraph range)

2. **Specialist Analyses** (5 parallel analysts):
   - Syntactician, Lexicologist, Rhetorician, Information Architect, Efficiency Auditor
   - Each receives: `preamble_instruction + analyst_prompt + preamble_text`
   - Prompt structure optimized for LLM caching (static + static + dynamic)
   - All results stored in `ResultStore` with sample_id + analyst_name keys

3. **Cross-Perspective Integration** (per-sample):
   - `CrossPerspectiveIntegratorConfig` takes all 5 specialist analyses
   - Identifies interactions, tensions, load-bearing features
   - Output stored as another "analyst" in ResultStore

4. **Cross-Text Synthesis** (multi-sample):
   - `CrossTextSynthesizerConfig` takes multiple integration analyses
   - Extracts recurring patterns and hierarchies
   - Output is standalone (not stored in ResultStore)

5. **Principles Synthesis**:
   - `SynthesizerOfPrinciplesConfig` converts patterns to prescriptive guide

**Style Evaluation Pipeline (style_evaluation.ipynb):**

1. **Flatten**: Extract content while removing style (`StyleFlatteningConfig`)
2. **Reconstruct** (4 methods × M runs):
   - Generic baseline
   - Few-shot learning
   - Author name prompting
   - Derived style instructions
3. **Judge**: Compare each reconstruction vs. original using structured JSON output (`StyleJudgeConfig` → `StyleJudgment`)
4. **Analyze**: Aggregate judgments in DataFrame, export to CSV

## Core Architectural Patterns

### 1. Pydantic-First Configuration

All prompts and LLM calls use type-safe Pydantic models:

```python
# Bad: string building
prompt = f"Analyze this: {text}"

# Good: type-safe config
config = RhetoricianConfig(
    include_writer_position=True,
    include_reader_positioning=True,
    include_persuasive_techniques=True,
    include_argumentative_moves=True
)
prompt = prompt_maker.render(config)
```

Every `*Config` class maps to a `.jinja` template via `template_name()` classmethod.

### 2. Template-Driven Prompts

**Never build prompts with string concatenation.** Use Jinja templates in `/prompts/`:

```python
# Prompt logic lives in prompts/rhetorician.jinja
# Python code is declarative:
config = RhetoricianConfig()
prompt = prompt_maker.render(config)
response = llm.complete(prompt)
```

To add a new prompt type:
1. Create `MyAnalystConfig(BasePromptConfig)` in `prompt_models.py`
2. Create `my_analyst.jinja` in `prompts/`
3. Export in `models/__init__.py`

### 3. Provenance Tracking

All text segments carry full provenance:

```python
segment = sampler.sample_segment(p_length=5)
# segment has: .text, .file_index, .paragraph_start, .paragraph_end, .file_path
store.save_segment(sample_id, segment)  # Preserves provenance in DB
```

This enables:
- Reproducible sampling
- Source attribution
- Debugging which text produced which analysis

### 4. ResultStore as Analysis Cache

`ResultStore` is sample-centric:
- Samples table: Original text + provenance
- Analyses table: (sample_id, analyst_name) → output

Pattern for new analysts:
```python
# Check if analysis exists
if store.is_complete(sample_id, ['new_analyst']):
    analysis = store.get_analysis(sample_id, 'new_analyst')
else:
    # Run analysis
    response = llm.complete(prompt)
    store.save_analysis(sample_id, 'new_analyst', response.content, response.model)
```

### 5. Structured Output with JSON Mode

For structured responses, use `complete_json()`:

```python
# Define Pydantic model for validation
class StyleJudgment(BaseModel):
    ranking: Literal["original_better", "reconstruction_better", "roughly_equal"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str

# LLM call with JSON mode
response = llm.complete_json(judge_prompt)
judgment_data = json.loads(response.content)
judgment = StyleJudgment(**judgment_data)  # Validates structure
```

## Important Implementation Details

### Specialist Analyst Metadata

All `SpecialistAnalystConfig` subclasses must implement:
- `template_name()` → maps to `.jinja` file
- `description()` → short description for UI/metadata
- `display_name()` → formatted name (e.g., "Syntactician")

This enables dynamic analyst selection without hardcoding.

### Dynamic Analysts in Integration

`CrossPerspectiveIntegratorConfig` accepts arbitrary analysts:

```python
# Supports runtime analyst selection
analysts_dict = {
    'rhetorician': {
        'analysis': '...',
        'analyst_descr_short': RhetoricianConfig.description()
    },
    'custom_analyst': {
        'analysis': '...',
        'analyst_descr_short': 'Custom description'
    }
}

config = CrossPerspectiveIntegratorConfig(
    original_text=text,
    analysts=analysts_dict  # Validates ≥2 analysts
)
```

Template loops over `analysts.items()` dynamically.

### Prompt Caching Optimization

Specialist analysis prompts use 3-part structure for caching:

```python
# Part 1: Static preamble (cacheable)
preamble_instruction = prompt_maker.render(PreambleInstructionConfig())

# Part 2: Static per-analyst (cacheable)
analyst_prompt = prompt_maker.render(SyntacticianConfig())

# Part 3: Dynamic text (not cacheable)
preamble_text = prompt_maker.render(PreambleTextConfig(text_to_analyze=text))

# Combine in order for optimal cache hits
full_prompt = f"{preamble_instruction}\n\n{analyst_prompt}\n\n{preamble_text}"
```

This structure minimizes token costs with providers that support prompt caching.

### LiteLLM Provider Flexibility

The `LLM` class wraps LiteLLM, supporting 50+ providers:

```python
# OpenAI
llm = LLM(LLMConfig(model="openai/gpt-4o", api_key=os.environ['OPENAI_API_KEY']))

# Mistral
llm = LLM(LLMConfig(model="mistral/mistral-large-2411", api_key=os.environ['MISTRAL_API_KEY']))

# Anthropic
llm = LLM(LLMConfig(model="anthropic/claude-3-5-sonnet-20241022", api_key=os.environ['ANTHROPIC_API_KEY']))
```

All prompts are provider-agnostic.

## File Organization Logic

### `/belletrist/models/`

Two model files with distinct purposes:

- **`llm_config_models.py`**: LLM interface models (LLMConfig, Message, LLMResponse, StyleJudgment)
- **`prompt_models.py`**: Prompt configuration models (all `*Config` classes that map to templates)

When adding structured output schemas, add to `llm_config_models.py`. When adding new prompt types, add to `prompt_models.py`.

### `/prompts/` Template Naming

Templates are named after their config's `template_name()`:

```python
class SyntacticianConfig(SpecialistAnalystConfig):
    @classmethod
    def template_name(cls) -> str:
        return "syntactician"  # Maps to prompts/syntactician.jinja
```

Multi-file templates (e.g., cross-perspective integrator) use `_framework`, `_guidelines`, `_output_structure` suffixes:
- `cross_perspective_integrator.jinja` (main)
- `cross_perspective_integrator_framework.jinja` (included)
- `cross_perspective_integrator_guidelines.jinja` (included)
- `cross_perspective_integrator_output_structure.jinja` (included)

### Module Exports

Public API exposed via `/belletrist/__init__.py`:
```python
from belletrist import (
    LLM, ToolLLM, LLMConfig,
    PromptMaker, DataSampler, ResultStore,
    RhetoricianConfig, SyntacticianConfig, ...
)
```

Internal imports use full paths: `from belletrist.models import ...`

## Data Sources

Text corpus in `/data/russell/`:
- 8 Bertrand Russell essays from Project Gutenberg
- Each ~460-514KB (prose paragraphs)
- License in `/data/GUTENBERG_LICENSE.md`

`DataSampler` loads all files at init, provides:
- `sample_segment(p_length=N)` - Random weighted by file size
- `get_paragraph_chunk(file_index, start_para, length)` - Deterministic selection
- `iter_paragraph_chunks(...)` - Streaming iterator

## Common Workflows

### Adding a New Specialist Analyst

1. Create config class in `prompt_models.py`:
```python
class NewAnalystConfig(SpecialistAnalystConfig):
    # Add 4 boolean sections (optional)
    include_section_one: bool = True
    include_section_two: bool = True
    include_section_three: bool = True
    include_section_four: bool = True

    @classmethod
    def template_name(cls) -> str:
        return "new_analyst"

    @classmethod
    def description(cls) -> str:
        return "Brief description of focus area"

    @classmethod
    def display_name(cls) -> str:
        return "New Analyst"
```

2. Create template `prompts/new_analyst.jinja`:
```jinja
You are a specialist analyst focusing on [domain].

{% if include_section_one %}
## Section One
Instructions for first analytical dimension...
{% endif %}

{# Repeat for other sections #}
```

3. Export in `models/__init__.py`

4. Add to notebook's `ANALYST_CONFIGS` dict

### Running Partial Analysis

Skip stages by using existing ResultStore data:

```python
# Load existing analyses
sample, analyses = store.get_sample_with_analyses('sample_001')

# Skip to cross-perspective integration
pattern_config = CrossPerspectiveIntegratorConfig(
    original_text=sample['text'],
    analysts={k: {'analysis': v, 'analyst_descr_short': '...'} for k, v in analyses.items()}
)
```

### Debugging LLM Calls

`LLMResponse` preserves full context:

```python
response = llm.complete(prompt)
print(response.content)        # Parsed text
print(response.model)          # Actual model used (may differ from requested)
print(response.finish_reason)  # 'stop', 'length', 'tool_calls', etc.
print(response.usage)          # Token counts
print(response.raw_response)   # Full LiteLLM response object
```

## Architecture Rationale

### Why Specialist Decomposition?

Breaking analysis into 5 independent specialists enables:
- **Parallel execution**: Run all 5 simultaneously
- **Comparative study**: Isolate syntax vs. rhetoric vs. efficiency patterns
- **Iterative refinement**: Improve one specialist without touching others
- **Clear boundaries**: Each analyst has well-defined responsibility

### Why Two-Level Synthesis?

**Per-sample (Cross-Perspective)**: Compresses 5 specialist viewpoints into unified understanding of *this text*

**Multi-sample (Cross-Text)**: Identifies recurring patterns across *multiple texts* to extract generalizable principles

This mirrors how humans learn style: observe specific instances, then abstract patterns.

### Why SQLite for ResultStore?

- **Relational integrity**: Foreign keys ensure analyses link to valid samples
- **Query flexibility**: Check completeness, retrieve by sample/analyst
- **Persistence**: Survives kernel restarts, enables incremental processing
- **Lightweight**: No external database server needed

### Why Template-First Prompts?

Alternatives considered:
- **String concatenation**: Hard to maintain, no validation, poor separation of concerns
- **Python-only**: Harder for non-coders to iterate on prompts
- **JSON configs**: Less expressive than templates for complex prompt logic

Jinja provides:
- Conditional sections (`{% if include_X %}`)
- Loops (`{% for analyst in analysts %}`)
- Includes (`{% include 'subtemplate.jinja' %}`)
- Clear separation: prompt logic in `.jinja`, validation in Python

## Known Constraints

- **Sample Selection**: Currently manual indices in notebooks. For production, implement automated train/test/fewshot splits.

- **Style Instructions Loading**: `style_evaluation.ipynb` assumes `derived_style_instructions.txt` exists. Generate via `SynthesizerOfPrinciplesConfig` first.

- **Token Costs**: Multi-stage pipeline can be expensive. Consider:
  - Using cheaper models for earlier stages
  - Caching static prompt components
  - Batching samples when possible

- **Judgment Consistency**: Style judge uses LLM, subject to stochastic variation. Increase `M_RUNS` for statistical significance.

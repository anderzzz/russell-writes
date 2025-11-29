"""
Utility modules for author modeling pipeline.

This package provides helper functions for passage extraction and evaluation parsing.
"""

from belletrist.utils.passage_extraction import (
    extract_paragraph_windows,
    extract_logical_sections,
    get_full_sample_as_passage,
    extract_passages_by_indices
)

from belletrist.utils.evaluation_parsing import (
    parse_passage_evaluation,
    parse_example_set_selection,
    extract_selection_criteria,
    extract_coherence_assessment,
    extract_alternative_passages,
    validate_passage_evaluation,
    validate_example_set_selection
)

__all__ = [
    # Passage extraction
    'extract_paragraph_windows',
    'extract_logical_sections',
    'get_full_sample_as_passage',
    'extract_passages_by_indices',
    # Evaluation parsing
    'parse_passage_evaluation',
    'parse_example_set_selection',
    'extract_selection_criteria',
    'extract_coherence_assessment',
    'extract_alternative_passages',
    'validate_passage_evaluation',
    'validate_example_set_selection'
]

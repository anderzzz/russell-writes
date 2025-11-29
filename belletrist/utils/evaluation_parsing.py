"""
Utilities for parsing LLM evaluation responses into structured data.

This module provides functions to extract structured information from LLM outputs
for Stage 4 (passage evaluation) and Stage 5 (example set construction).
"""

import re
from typing import Dict, Any, List, Optional


def parse_passage_evaluation(response_text: str) -> Dict[str, Any]:
    """
    Parse a passage evaluation response into structured fields.

    Extracts:
        - density_rating: Integer 1-5
        - task_coverage: Text describing compositional tasks demonstrated
        - teaching_value: Text assessing pedagogical factors
        - recommendation: Boolean (YES=True, MAYBE/NO=False)

    Args:
        response_text: LLM response from passage_evaluation.jinja template

    Returns:
        Dict with keys: density_rating, task_coverage, teaching_value, recommendation

    Raises:
        ValueError: If critical fields cannot be parsed

    Example:
        >>> response = "### DENSITY RATING\\n**Rating:** 4\\n..."
        >>> result = parse_passage_evaluation(response)
        >>> print(result['density_rating'])
        4
    """
    result = {
        'density_rating': None,
        'task_coverage': None,
        'teaching_value': None,
        'recommendation': None
    }

    # Extract density rating (1-5)
    # Look for patterns like "Rating: 4" or "**Rating:** 4" or "RATING: 4"
    rating_pattern = r'\*?\*?[Rr]ating\*?\*?:?\s*(\d)'
    rating_match = re.search(rating_pattern, response_text)
    if rating_match:
        rating = int(rating_match.group(1))
        if 1 <= rating <= 5:
            result['density_rating'] = rating
        else:
            raise ValueError(f"Invalid density rating: {rating} (must be 1-5)")
    else:
        raise ValueError("Could not find density rating in response")

    # Extract task coverage section
    # Look for "### COMPOSITIONAL TASK COVERAGE" or similar heading
    task_coverage_pattern = r'###\s*COMPOSITIONAL TASK COVERAGE\s*\n(.*?)(?=\n###|\Z)'
    task_match = re.search(task_coverage_pattern, response_text, re.DOTALL | re.IGNORECASE)
    if task_match:
        result['task_coverage'] = task_match.group(1).strip()
    else:
        # Fallback: try to find any section with "task" in heading
        task_fallback = r'###\s*.*?TASK.*?\n(.*?)(?=\n###|\Z)'
        task_match = re.search(task_fallback, response_text, re.DOTALL | re.IGNORECASE)
        if task_match:
            result['task_coverage'] = task_match.group(1).strip()

    # Extract teaching value section
    teaching_pattern = r'###\s*TEACHING VALUE ASSESSMENT\s*\n(.*?)(?=\n###|\Z)'
    teaching_match = re.search(teaching_pattern, response_text, re.DOTALL | re.IGNORECASE)
    if teaching_match:
        result['teaching_value'] = teaching_match.group(1).strip()
    else:
        # Fallback: try to find section with "teaching" or "value"
        teaching_fallback = r'###\s*.*?TEACHING.*?\n(.*?)(?=\n###|\Z)'
        teaching_match = re.search(teaching_fallback, response_text, re.DOTALL | re.IGNORECASE)
        if teaching_match:
            result['teaching_value'] = teaching_match.group(1).strip()

    # Extract recommendation (YES/MAYBE/NO)
    # Look for "Suitable as Few-Shot Example: YES" or similar
    rec_pattern = r'(?:Suitable|Recommendation).*?:\s*\*?\*?([Yy][Ee][Ss]|[Mm][Aa][Yy][Bb][Ee]|[Nn][Oo])\*?\*?'
    rec_match = re.search(rec_pattern, response_text)
    if rec_match:
        rec_value = rec_match.group(1).upper()
        result['recommendation'] = (rec_value == 'YES')
    else:
        # Default to False if not found
        result['recommendation'] = False

    return result


def parse_example_set_selection(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse example set selection response into list of selected passages.

    Extracts passage identifiers and metadata from Stage 5 output.

    Args:
        response_text: LLM response from example_set_construction.jinja template

    Returns:
        List of dicts with keys:
            - passage_id: Identifier (evaluation ID or sample ID)
            - density_rating: Rating if mentioned
            - unique_contribution: What this passage adds to the set
            - compositional_tasks: Tasks demonstrated

    Example:
        >>> response = "PASSAGE #1: passage_eval_042\\n**Density Rating:** 5\\n..."
        >>> passages = parse_example_set_selection(response)
        >>> print(passages[0]['passage_id'])
        'passage_eval_042'
    """
    passages = []

    # Find all PASSAGE sections
    # Pattern: "PASSAGE #N: [ID]" followed by metadata
    passage_pattern = r'PASSAGE #(\d+):\s*([^\n]+)\s*\n(.*?)(?=PASSAGE #|\n###|\Z)'
    passage_matches = re.finditer(passage_pattern, response_text, re.DOTALL | re.IGNORECASE)

    for match in passage_matches:
        passage_num = int(match.group(1))
        passage_id = match.group(2).strip()
        passage_content = match.group(3)

        passage_info = {
            'passage_id': passage_id,
            'passage_number': passage_num,
            'density_rating': None,
            'unique_contribution': None,
            'compositional_tasks': None
        }

        # Extract density rating from this passage's section
        rating_match = re.search(r'\*?\*?Density Rating\*?\*?:?\s*(\d)', passage_content)
        if rating_match:
            passage_info['density_rating'] = int(rating_match.group(1))

        # Extract unique contribution
        contrib_match = re.search(
            r'\*?\*?Unique Contribution\*?\*?:?\s*\n(.*?)(?=\n\*?\*?[A-Z]|\Z)',
            passage_content,
            re.DOTALL
        )
        if contrib_match:
            passage_info['unique_contribution'] = contrib_match.group(1).strip()

        # Extract compositional tasks
        tasks_match = re.search(
            r'\*?\*?Compositional Task\(?s?\)?\*?\*?:?\s*\n(.*?)(?=\n\*?\*?[A-Z]|\Z)',
            passage_content,
            re.DOTALL
        )
        if tasks_match:
            passage_info['compositional_tasks'] = tasks_match.group(1).strip()

        passages.append(passage_info)

    return passages


def extract_selection_criteria(response_text: str) -> Optional[str]:
    """
    Extract selection criteria section from example set construction response.

    Args:
        response_text: LLM response from example_set_construction.jinja template

    Returns:
        Selection criteria text or None if not found

    Example:
        >>> criteria = extract_selection_criteria(response)
        >>> print(criteria[:50])
        '1. **Minimum density**: Only 4-5 rated passages...'
    """
    criteria_pattern = r'###\s*SELECTION CRITERIA\s*\n(.*?)(?=\n###|\Z)'
    match = re.search(criteria_pattern, response_text, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()
    return None


def extract_coherence_assessment(response_text: str) -> Optional[str]:
    """
    Extract set coherence assessment from example set construction response.

    Args:
        response_text: LLM response from example_set_construction.jinja template

    Returns:
        Coherence assessment text or None if not found
    """
    coherence_pattern = r'###\s*SET COHERENCE ASSESSMENT\s*\n(.*?)(?=\n###|\Z)'
    match = re.search(coherence_pattern, response_text, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()
    return None


def extract_alternative_passages(response_text: str) -> List[Dict[str, Any]]:
    """
    Extract alternative passage recommendations from example set construction.

    Args:
        response_text: LLM response from example_set_construction.jinja template

    Returns:
        List of dicts with keys:
            - passage_id: Identifier
            - rating: Density rating
            - when_to_use: Use case description
            - trade_off: Trade-off analysis
    """
    alternatives = []

    # Find ALTERNATIVE sections
    alt_pattern = r'ALTERNATIVE:\s*([^\n]+)\s*\n(.*?)(?=ALTERNATIVE:|\n###|\Z)'
    alt_matches = re.finditer(alt_pattern, response_text, re.DOTALL | re.IGNORECASE)

    for match in alt_matches:
        passage_id = match.group(1).strip()
        alt_content = match.group(2)

        alt_info = {
            'passage_id': passage_id,
            'rating': None,
            'when_to_use': None,
            'trade_off': None
        }

        # Extract rating
        rating_match = re.search(r'\*?\*?Rating\*?\*?:?\s*(\d)', alt_content)
        if rating_match:
            alt_info['rating'] = int(rating_match.group(1))

        # Extract when to use
        when_match = re.search(
            r'\*?\*?When to Use\*?\*?:?\s*\n(.*?)(?=\n\*?\*?[A-Z]|\Z)',
            alt_content,
            re.DOTALL
        )
        if when_match:
            alt_info['when_to_use'] = when_match.group(1).strip()

        # Extract trade-off
        trade_match = re.search(
            r'\*?\*?Trade-?Off\*?\*?:?\s*\n(.*?)(?=\n\*?\*?[A-Z]|\Z)',
            alt_content,
            re.DOTALL
        )
        if trade_match:
            alt_info['trade_off'] = trade_match.group(1).strip()

        alternatives.append(alt_info)

    return alternatives


def validate_passage_evaluation(parsed: Dict[str, Any]) -> bool:
    """
    Validate that a parsed passage evaluation has all required fields.

    Args:
        parsed: Dict returned by parse_passage_evaluation()

    Returns:
        True if valid, False otherwise
    """
    required_fields = ['density_rating', 'recommendation']

    # Check required fields are present and not None
    for field in required_fields:
        if field not in parsed or parsed[field] is None:
            return False

    # Check density_rating is in valid range
    if not (1 <= parsed['density_rating'] <= 5):
        return False

    # Check recommendation is boolean
    if not isinstance(parsed['recommendation'], bool):
        return False

    return True


def validate_example_set_selection(parsed: List[Dict[str, Any]], min_passages: int = 3, max_passages: int = 4) -> bool:
    """
    Validate that a parsed example set selection has the right number of passages.

    Args:
        parsed: List of dicts returned by parse_example_set_selection()
        min_passages: Minimum number of passages required
        max_passages: Maximum number of passages allowed

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(parsed, list):
        return False

    if not (min_passages <= len(parsed) <= max_passages):
        return False

    # Check each passage has required fields
    for passage in parsed:
        if 'passage_id' not in passage or not passage['passage_id']:
            return False

    return True

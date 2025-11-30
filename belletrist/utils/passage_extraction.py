"""
Utilities for extracting passages from stored samples for evaluation.

This module provides functions to extract text segments from ResultStore samples
for use in Stage 4 (passage evaluation) and Stage 5 (example set construction).
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import difflib
import re
import json


def extract_paragraph_windows(
    store,
    sample_ids: Optional[List[str]] = None,
    window_size: int = 3,
    overlap: int = 1
) -> List[Dict[str, Any]]:
    """
    Extract overlapping paragraph windows from samples.

    Args:
        store: ResultStore instance
        sample_ids: List of sample IDs to extract from. If None, uses all samples.
        window_size: Number of paragraphs per window
        overlap: Number of paragraphs to overlap between windows

    Returns:
        List of passage dicts with keys:
            - source_sample_id: Original sample ID
            - paragraph_range: String like "3-5" indicating paragraph indices
            - text: The extracted text
            - file_path: Source file path
            - file_index: Source file index

    Example:
        >>> passages = extract_paragraph_windows(store, window_size=5, overlap=2)
        >>> print(passages[0]['paragraph_range'])
        '0-4'
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must be >= 0 and < window_size")

    # Get samples to process
    if sample_ids is None:
        cursor = store.conn.execute("SELECT sample_id FROM samples ORDER BY created_at")
        sample_ids = [row[0] for row in cursor.fetchall()]

    passages = []

    for sample_id in sample_ids:
        sample = store.get_sample(sample_id)
        if not sample:
            continue

        # Split text into paragraphs (preserve original splitting)
        text = sample['text']
        paragraphs = text.split('\n\n')

        # Skip if not enough paragraphs
        if len(paragraphs) < window_size:
            # Include entire sample as single passage
            passages.append({
                'source_sample_id': sample_id,
                'paragraph_range': f"0-{len(paragraphs) - 1}",
                'text': text,
                'file_path': sample.get('file_path'),
                'file_index': sample.get('file_index')
            })
            continue

        # Extract sliding windows
        step = window_size - overlap
        for start_idx in range(0, len(paragraphs) - window_size + 1, step):
            end_idx = start_idx + window_size - 1
            window_text = '\n\n'.join(paragraphs[start_idx:start_idx + window_size])

            passages.append({
                'source_sample_id': sample_id,
                'paragraph_range': f"{start_idx}-{end_idx}",
                'text': window_text,
                'file_path': sample.get('file_path'),
                'file_index': sample.get('file_index')
            })

    return passages


def extract_logical_sections(
    store,
    sample_ids: Optional[List[str]] = None,
    min_length: int = 200,
    max_length: int = 800
) -> List[Dict[str, Any]]:
    """
    Extract logical sections from samples based on paragraph breaks and length.

    Attempts to identify natural sections by:
    1. Splitting on double paragraph breaks
    2. Grouping consecutive paragraphs to meet min_length
    3. Breaking when exceeding max_length

    Args:
        store: ResultStore instance
        sample_ids: List of sample IDs to extract from. If None, uses all samples.
        min_length: Minimum word count for a section
        max_length: Maximum word count for a section

    Returns:
        List of passage dicts with keys:
            - source_sample_id: Original sample ID
            - paragraph_range: String like "0-2" indicating paragraph indices
            - text: The extracted text
            - word_count: Number of words in passage
            - file_path: Source file path
            - file_index: Source file index

    Example:
        >>> sections = extract_logical_sections(store, min_length=300, max_length=600)
        >>> print(sections[0]['word_count'])
        451
    """
    if min_length < 1:
        raise ValueError("min_length must be at least 1")
    if max_length < min_length:
        raise ValueError("max_length must be >= min_length")

    # Get samples to process
    if sample_ids is None:
        cursor = store.conn.execute("SELECT sample_id FROM samples ORDER BY created_at")
        sample_ids = [row[0] for row in cursor.fetchall()]

    passages = []

    for sample_id in sample_ids:
        sample = store.get_sample(sample_id)
        if not sample:
            continue

        text = sample['text']
        paragraphs = text.split('\n\n')

        # Build sections by accumulating paragraphs
        current_section = []
        current_start_idx = 0
        current_word_count = 0

        for idx, para in enumerate(paragraphs):
            para_word_count = len(para.split())

            # Check if adding this paragraph would exceed max_length
            if current_section and (current_word_count + para_word_count > max_length):
                # Save current section if it meets min_length
                if current_word_count >= min_length:
                    section_text = '\n\n'.join(current_section)
                    passages.append({
                        'source_sample_id': sample_id,
                        'paragraph_range': f"{current_start_idx}-{idx - 1}",
                        'text': section_text,
                        'word_count': current_word_count,
                        'file_path': sample.get('file_path'),
                        'file_index': sample.get('file_index')
                    })

                # Start new section
                current_section = [para]
                current_start_idx = idx
                current_word_count = para_word_count
            else:
                # Add to current section
                current_section.append(para)
                current_word_count += para_word_count

        # Save final section if it meets min_length
        if current_section and current_word_count >= min_length:
            section_text = '\n\n'.join(current_section)
            passages.append({
                'source_sample_id': sample_id,
                'paragraph_range': f"{current_start_idx}-{len(paragraphs) - 1}",
                'text': section_text,
                'word_count': current_word_count,
                'file_path': sample.get('file_path'),
                'file_index': sample.get('file_index')
            })

    return passages


def get_full_sample_as_passage(
    store,
    sample_id: str
) -> Optional[Dict[str, Any]]:
    """
    Get a full sample as a passage for evaluation.

    Useful when you want to evaluate entire samples rather than windows or sections.

    Args:
        store: ResultStore instance
        sample_id: Sample ID to retrieve

    Returns:
        Passage dict or None if sample doesn't exist

    Example:
        >>> passage = get_full_sample_as_passage(store, 'sample_001')
        >>> print(passage['paragraph_range'])
        'full'
    """
    sample = store.get_sample(sample_id)
    if not sample:
        return None

    # Count paragraphs
    paragraphs = sample['text'].split('\n\n')

    return {
        'source_sample_id': sample_id,
        'paragraph_range': 'full',
        'text': sample['text'],
        'word_count': len(sample['text'].split()),
        'paragraph_count': len(paragraphs),
        'file_path': sample.get('file_path'),
        'file_index': sample.get('file_index')
    }


def extract_passages_by_indices(
    store,
    sample_id: str,
    paragraph_indices: List[Tuple[int, int]]
) -> List[Dict[str, Any]]:
    """
    Extract specific paragraph ranges from a sample.

    Useful when you have specific passages identified (e.g., from Stage 1-3 analyses
    that flagged signature moments or high-density sections).

    Args:
        store: ResultStore instance
        sample_id: Sample ID to extract from
        paragraph_indices: List of (start, end) tuples (inclusive)

    Returns:
        List of passage dicts

    Example:
        >>> # Extract paragraphs 0-2 and 5-7 from sample_001
        >>> passages = extract_passages_by_indices(
        ...     store, 'sample_001', [(0, 2), (5, 7)]
        ... )
        >>> print(passages[0]['paragraph_range'])
        '0-2'
    """
    sample = store.get_sample(sample_id)
    if not sample:
        return []

    paragraphs = sample['text'].split('\n\n')
    passages = []

    for start_idx, end_idx in paragraph_indices:
        if start_idx < 0 or end_idx >= len(paragraphs) or start_idx > end_idx:
            # Skip invalid ranges
            continue

        window_text = '\n\n'.join(paragraphs[start_idx:end_idx + 1])

        passages.append({
            'source_sample_id': sample_id,
            'paragraph_range': f"{start_idx}-{end_idx}",
            'text': window_text,
            'word_count': len(window_text.split()),
            'file_path': sample.get('file_path'),
            'file_index': sample.get('file_index')
        })

    return passages


# =============================================================================
# Quote-Based Passage Extraction
# =============================================================================

class QuoteMatchResult:
    """Result of fuzzy quote matching with metadata."""

    def __init__(
        self,
        found: bool,
        core_paragraph_index: Optional[int] = None,
        match_confidence: float = 0.0,
        matched_text: Optional[str] = None,
        core_range: Optional[Tuple[int, int]] = None,
        full_range: Optional[Tuple[int, int]] = None,
        padded_passage: Optional[str] = None
    ):
        self.found = found
        self.core_paragraph_index = core_paragraph_index
        self.match_confidence = match_confidence
        self.matched_text = matched_text
        self.core_range = core_range
        self.full_range = full_range
        self.padded_passage = padded_passage

    def to_dict(self) -> dict:
        """Convert to dict for storage/serialization."""
        return {
            'found': self.found,
            'core_paragraph_index': self.core_paragraph_index,
            'match_confidence': self.match_confidence,
            'matched_text': self.matched_text,
            'core_range': self.core_range,
            'full_range': self.full_range,
            'padded_passage': self.padded_passage
        }


def find_passage_by_quote(
    store,
    sample_id: str,
    quote_text: str,
    padding_before: int = 2,
    padding_after: int = 2,
    min_confidence: float = 0.6,
    allow_partial: bool = True
) -> QuoteMatchResult:
    """
    Find a quoted passage in a sample and return it with padding.

    Uses fuzzy string matching (difflib.SequenceMatcher) to locate the quote
    within the sample's paragraphs, then adds contextual padding to create
    a passage suitable for few-shot learning (typically 3-5 paragraphs).

    Strategy:
    1. Normalize quote and paragraph text (strip, lowercase, collapse whitespace)
    2. Try exact substring match first (fast path)
    3. Fall back to fuzzy matching if exact match fails
    4. For multi-paragraph quotes, find best matching span
    5. Apply symmetric padding (±N paragraphs, bounded by sample edges)
    6. Handle edge cases (quote at start/end of sample)

    Args:
        store: ResultStore instance
        sample_id: ID of sample containing the quote
        quote_text: The quoted text to find (1-3 sentences typical)
        padding_before: Number of paragraphs to include before match (default: 2)
        padding_after: Number of paragraphs to include after match (default: 2)
        min_confidence: Minimum similarity threshold 0.0-1.0 (default: 0.6)
        allow_partial: If True, allow matching partial sentences at boundaries

    Returns:
        QuoteMatchResult with:
            - found: Boolean indicating success
            - core_paragraph_index: Index of paragraph containing quote
            - match_confidence: Similarity score (0.0-1.0)
            - matched_text: Actual text that matched
            - core_range: (start, end) indices for core match
            - full_range: (start, end) indices after padding
            - padded_passage: Full text with padding applied

    Example:
        >>> result = find_passage_by_quote(
        ...     store,
        ...     'sample_001',
        ...     'Over the whole development of Russia...',
        ...     padding_before=2,
        ...     padding_after=2
        ... )
        >>> if result.found:
        ...     print(f"Match: {result.match_confidence:.2f}")
        ...     print(f"Range: {result.full_range}")
    """
    # Validate inputs
    sample = store.get_sample(sample_id)
    if not sample:
        raise ValueError(f"Sample '{sample_id}' not found")

    if not quote_text or not quote_text.strip():
        raise ValueError("quote_text cannot be empty")

    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError("min_confidence must be between 0.0 and 1.0")

    # Split sample into paragraphs
    paragraphs = sample['text'].split('\n\n')

    # Normalize quote for matching
    quote_normalized = _normalize_text(quote_text)

    # Fast path: Try exact substring match first
    best_match = _try_exact_match(paragraphs, quote_normalized)

    if not best_match:
        # Fuzzy path: Use difflib for approximate matching
        best_match = _try_fuzzy_match(
            paragraphs,
            quote_normalized,
            min_confidence,
            allow_partial
        )

    # No match found
    if not best_match:
        return QuoteMatchResult(found=False)

    # Extract match details
    core_start, core_end, confidence, matched_text = best_match

    # Apply padding (symmetric, bounded by sample edges)
    full_start = max(0, core_start - padding_before)
    full_end = min(len(paragraphs) - 1, core_end + padding_after)

    # Build padded passage
    padded_paragraphs = paragraphs[full_start:full_end + 1]
    padded_passage = '\n\n'.join(padded_paragraphs)

    return QuoteMatchResult(
        found=True,
        core_paragraph_index=core_start,
        match_confidence=confidence,
        matched_text=matched_text,
        core_range=(core_start, core_end),
        full_range=(full_start, full_end),
        padded_passage=padded_passage
    )


def _normalize_text(text: str) -> str:
    """Normalize text for matching: strip, collapse whitespace, lowercase."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text


def _try_exact_match(
    paragraphs: List[str],
    quote_normalized: str
) -> Optional[Tuple[int, int, float, str]]:
    """
    Try exact substring matching (fast path).

    Returns:
        Tuple of (start_idx, end_idx, confidence=1.0, matched_text) or None
    """
    # Try single paragraph match
    for i, para in enumerate(paragraphs):
        para_normalized = _normalize_text(para)
        if quote_normalized in para_normalized:
            return (i, i, 1.0, para)

    # Try multi-paragraph spans (up to 3 paragraphs)
    for span_size in [2, 3]:
        for i in range(len(paragraphs) - span_size + 1):
            span_text = '\n\n'.join(paragraphs[i:i + span_size])
            span_normalized = _normalize_text(span_text)
            if quote_normalized in span_normalized:
                return (i, i + span_size - 1, 1.0, span_text)

    return None


def _try_fuzzy_match(
    paragraphs: List[str],
    quote_normalized: str,
    min_confidence: float,
    allow_partial: bool
) -> Optional[Tuple[int, int, float, str]]:
    """
    Try fuzzy matching using difflib.SequenceMatcher.

    Returns:
        Tuple of (start_idx, end_idx, confidence, matched_text) or None
    """
    best_score = 0.0
    best_match = None

    # Try single paragraph fuzzy match
    for i, para in enumerate(paragraphs):
        para_normalized = _normalize_text(para)
        ratio = difflib.SequenceMatcher(None, quote_normalized, para_normalized).ratio()

        if ratio > best_score:
            best_score = ratio
            best_match = (i, i, ratio, para)

    # Try multi-paragraph spans (up to 3)
    for span_size in [2, 3]:
        for i in range(len(paragraphs) - span_size + 1):
            span_text = '\n\n'.join(paragraphs[i:i + span_size])
            span_normalized = _normalize_text(span_text)
            ratio = difflib.SequenceMatcher(None, quote_normalized, span_normalized).ratio()

            if ratio > best_score:
                best_score = ratio
                best_match = (i, i + span_size - 1, ratio, span_text)

    # Check if best match meets threshold
    if best_score >= min_confidence:
        return best_match

    return None


def extract_nominated_passages_from_analysis(
    store,
    llm,
    sample_id: str,
    analysis_text: str,
    analysis_type: str,
    padding_before: int = 2,
    padding_after: int = 2,
    min_confidence: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Extract passages nominated in Stage 1-2 analyses via LLM quote extraction + fuzzy matching.

    This orchestrates the full pipeline:
    1. LLM JSON extraction of quotes from analysis text
    2. Fuzzy matching to locate quotes in sample paragraphs
    3. Padding to create few-shot suitable passages

    Args:
        store: ResultStore instance
        llm: LLM instance (must support complete_json)
        sample_id: ID of the sample that was analyzed
        analysis_text: The Stage 1-2 analysis containing quotes
        analysis_type: Type of analysis (implied_author, decision_pattern, functional_texture)
        padding_before: Paragraphs to add before match (default: 2)
        padding_after: Paragraphs to add after match (default: 2)
        min_confidence: Minimum fuzzy match threshold (default: 0.6)

    Returns:
        List of passage dicts with keys:
            - source_sample_id: Original sample ID
            - paragraph_range: String like "3-7" (full range after padding)
            - core_paragraph_range: String like "5-5" (quote location before padding)
            - text: The padded passage text
            - quote_text: The original quote that was matched
            - quote_context_type: Type of quote (signature_moment, etc.)
            - quote_rationale: Why the quote was highlighted
            - match_confidence: Fuzzy match score (0.0-1.0)
            - extraction_method: Always "nominated_from_analysis"
            - file_path: Source file path (if available)
            - file_index: Source file index (if available)

    Raises:
        ValueError: If sample_id not found or analysis_text is empty
        RuntimeError: If LLM quote extraction fails

    Example:
        >>> passages = extract_nominated_passages_from_analysis(
        ...     store, llm, "sample_001", analysis, "implied_author"
        ... )
        >>> print(f"Found {len(passages)} nominated passages")
        >>> for p in passages:
        ...     print(f"  {p['paragraph_range']}: {p['match_confidence']:.2f}")
    """
    # Validate inputs
    if not store.get_sample(sample_id):
        raise ValueError(f"Sample '{sample_id}' not found")

    if not analysis_text or not analysis_text.strip():
        raise ValueError("analysis_text cannot be empty")

    # Import here to avoid circular dependency
    from belletrist.prompt_maker import PromptMaker
    from belletrist.models.author_modeling_models import (
        QuoteExtractionConfig,
        QuoteExtractionResponse
    )

    # Step 1: Extract quotes using LLM with JSON mode
    prompt_maker = PromptMaker()
    config = QuoteExtractionConfig(
        analysis_text=analysis_text,
        analysis_type=analysis_type
    )
    prompt = prompt_maker.render(config)

    try:
        response = llm.complete_json(prompt)
        quote_data = json.loads(response.content)
        extraction = QuoteExtractionResponse(**quote_data)
    except Exception as e:
        raise RuntimeError(f"LLM quote extraction failed for {sample_id}: {e}")

    # Step 2: For each quote, find and pad the passage
    passages = []

    for quote_obj in extraction.quotes:
        result = find_passage_by_quote(
            store=store,
            sample_id=sample_id,
            quote_text=quote_obj.quote_text,
            padding_before=padding_before,
            padding_after=padding_after,
            min_confidence=min_confidence
        )

        if result.found:
            sample = store.get_sample(sample_id)
            passages.append({
                'source_sample_id': sample_id,
                'paragraph_range': f"{result.full_range[0]}-{result.full_range[1]}",
                'core_paragraph_range': f"{result.core_range[0]}-{result.core_range[1]}",
                'text': result.padded_passage,
                'quote_text': quote_obj.quote_text,
                'quote_context_type': quote_obj.context_type,
                'quote_rationale': quote_obj.rationale,
                'match_confidence': result.match_confidence,
                'extraction_method': 'nominated_from_analysis',
                'file_path': sample.get('file_path'),
                'file_index': sample.get('file_index')
            })
        else:
            # Log failed match
            quote_preview = quote_obj.quote_text[:50] + "..." if len(quote_obj.quote_text) > 50 else quote_obj.quote_text
            print(f"Warning: Could not locate quote in {sample_id}: {quote_preview}")

    return passages

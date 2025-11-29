"""
Utilities for extracting passages from stored samples for evaluation.

This module provides functions to extract text segments from ResultStore samples
for use in Stage 4 (passage evaluation) and Stage 5 (example set construction).
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


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

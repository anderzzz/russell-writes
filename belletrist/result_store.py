"""
Result storage for text samples and multi-analyst outputs.

Uses SQLite with two tables:
- samples: Original text with provenance (file/paragraph indices)
- analyses: Analyst outputs keyed by (sample_id, analyst_type)
"""
from typing import Optional
import sqlite3
from pathlib import Path
import json


class ResultStore:
    """SQLite-backed storage for samples and multi-analyst results.

    Manages two tables:
    - samples: Text samples with provenance (where they came from)
    - analyses: Analyst outputs for each sample

    Example:
        from belletrist import DataSampler, ResultStore

        sampler = DataSampler()
        store = ResultStore(Path("results.db"))

        # Save a text segment with automatic provenance
        segment = sampler.sample_segment(10)
        store.save_segment("sample_001", segment)

        # Save analyses
        store.save_analysis("sample_001", "rhetorician", output="...", model="gpt-4")

        # Retrieve everything (returns dicts)
        sample, analyses = store.get_sample_with_analyses("sample_001")
        print(sample['text'])  # Access via dict keys
    """

    def __init__(self, filepath: Path):
        """Initialize store and create schema if needed.

        Args:
            filepath: Path to SQLite database file
        """
        self.filepath = filepath
        self.conn = sqlite3.connect(filepath)
        self.conn.row_factory = sqlite3.Row  # Dict-like row access
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                sample_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                file_index INTEGER,
                paragraph_start INTEGER,
                paragraph_end INTEGER
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                sample_id TEXT,
                analyst TEXT,
                output TEXT,
                model TEXT,
                PRIMARY KEY (sample_id, analyst),
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
            )
        """)
        self.conn.commit()

    def save_sample(
        self,
        sample_id: str,
        text: str,
        file_index: Optional[int] = None,
        paragraph_start: Optional[int] = None,
        paragraph_end: Optional[int] = None,
    ):
        """Store a text sample with provenance information.

        Args:
            sample_id: Unique identifier for this sample
            text: The actual text content
            file_index: Index of source file in DataSampler
            paragraph_start: Starting paragraph index (for slice)
            paragraph_end: Ending paragraph index (for slice)
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO samples
            VALUES (?, ?, ?, ?, ?)
        """, (
            sample_id,
            text,
            file_index,
            paragraph_start,
            paragraph_end,
        ))
        self.conn.commit()

    def save_analysis(
        self,
        sample_id: str,
        analyst: str,
        output: str,
        model: str
    ):
        """Save an analysis result for a sample.

        Args:
            sample_id: ID of the sample being analyzed
            analyst: Type of analyst (e.g., "rhetorician", "syntactician")
            output: Analysis output text
            model: Model used for analysis (e.g., "gpt-4")

        Raises:
            ValueError: If sample_id doesn't exist in samples table
        """
        # Verify sample exists
        if not self.get_sample(sample_id):
            raise ValueError(
                f"Sample '{sample_id}' not found. Save sample first with save_sample()."
            )

        self.conn.execute("""
            INSERT OR REPLACE INTO analyses
            VALUES (?, ?, ?, ?)
        """, (
            sample_id,
            analyst,
            output,
            model
        ))
        self.conn.commit()

    def save_segment(
        self,
        sample_id: str,
        segment: 'TextSegment',
    ):
        """Save a TextSegment directly to the store.

        Convenience method that extracts all fields from a TextSegment
        from the DataSampler.

        Args:
            sample_id: Unique identifier for this sample
            segment: TextSegment object from DataSampler

        """
        self.save_sample(
            sample_id=sample_id,
            text=segment.text,
            file_index=segment.file_index,
            paragraph_start=segment.paragraph_start,
            paragraph_end=segment.paragraph_end,
        )

    def get_sample(self, sample_id: str) -> Optional[dict]:
        """Retrieve a text sample by ID.

        Args:
            sample_id: Sample identifier

        Returns:
            Dictionary with sample data or None if not found.
            Keys: sample_id, text, file_index, paragraph_start, paragraph_end
        """
        row = self.conn.execute(
            "SELECT sample_id, text, file_index, paragraph_start, paragraph_end FROM samples WHERE sample_id=?",
            (sample_id,)
        ).fetchone()

        if not row:
            return None

        return {
            'sample_id': row['sample_id'],
            'text': row['text'],
            'file_index': row['file_index'],
            'paragraph_start': row['paragraph_start'],
            'paragraph_end': row['paragraph_end'],
        }

    def get_analysis(self, sample_id: str, analyst: str) -> Optional[str]:
        """Get one specific analysis output.

        Args:
            sample_id: Sample identifier
            analyst: Analyst type

        Returns:
            Analysis output text or None if not found
        """
        row = self.conn.execute(
            "SELECT output FROM analyses WHERE sample_id=? AND analyst=?",
            (sample_id, analyst)
        ).fetchone()
        return row['output'] if row else None

    def get_all_analyses(self, sample_id: str) -> dict[str, str]:
        """Get all analyst outputs for a sample.

        Args:
            sample_id: Sample identifier

        Returns:
            Dictionary mapping analyst type to output text
        """
        rows = self.conn.execute(
            "SELECT analyst, output FROM analyses WHERE sample_id=?",
            (sample_id,)
        ).fetchall()
        return {row['analyst']: row['output'] for row in rows}

    def get_sample_with_analyses(
        self,
        sample_id: str
    ) -> tuple[dict, dict[str, str]]:
        """Get both sample and all its analyses in one call.

        Args:
            sample_id: Sample identifier

        Returns:
            Tuple of (sample dict, dict of analyst outputs)

        Raises:
            ValueError: If sample not found
        """
        sample = self.get_sample(sample_id)
        if not sample:
            raise ValueError(f"Sample '{sample_id}' not found")
        analyses = self.get_all_analyses(sample_id)
        return sample, analyses

    def list_samples(self) -> list[str]:
        """Get all sample IDs in insertion order.

        Returns:
            List of sample IDs, oldest first
        """
        rows = self.conn.execute(
            "SELECT sample_id FROM samples ORDER BY rowid"
        ).fetchall()
        return [row['sample_id'] for row in rows]

    def is_complete(
        self,
        sample_id: str,
        required_analysts: list[str]
    ) -> bool:
        """Check if all required analyses exist for a sample.

        Args:
            sample_id: Sample identifier
            required_analysts: List of analyst types to check for

        Returns:
            True if all required analyses present, False otherwise
        """
        analyses = self.get_all_analyses(sample_id)
        return all(analyst in analyses for analyst in required_analysts)

    def close(self):
        """Close database connection."""
        self.conn.close()

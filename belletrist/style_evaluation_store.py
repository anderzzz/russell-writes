"""
SQLite-backed storage for style evaluation experiments.

This module provides crash-resilient storage for comparative style evaluation
workflows. Unlike ResultStore (designed for hierarchical synthesis pipelines),
StyleEvaluationStore is optimized for flat comparative evaluation:

Workflow: Samples → Flatten → Reconstruct (4 methods × M runs) → Judge comparatively

Key features:
- Atomic writes: Every LLM call saved immediately (crash resilient)
- Resume support: Check what's done, skip completed work
- Blind evaluation: Stores method mappings for anonymous judging
- Easy export: DataFrame with resolved rankings for analysis

Schema (3 tables):
1. samples: Original texts + flattened content + provenance
2. reconstructions: 4 methods × M runs per sample
3. comparative_judgments: Anonymous rankings + method mappings + confidence

Usage:
    store = StyleEvaluationStore(Path("results.db"))

    # Save sample
    store.save_sample("sample_001", original, flattened, "File 0, para 50-55")

    # Save reconstructions
    store.save_reconstruction("sample_001", run=0, "fewshot", text, "gpt-4")

    # Check if judgment needed
    if not store.has_judgment("sample_001", run=0):
        # Generate mapping and judgment
        mapping = store.create_random_mapping(seed=42)
        judgment = ... # Get from LLM
        store.save_judgment("sample_001", run=0, judgment, mapping, "claude-3.5")

    # Export for analysis
    df = store.to_dataframe()  # Rankings resolved to methods
"""
from typing import Optional, Literal
import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd
import random


class StyleEvaluationStore:
    """SQLite-backed crash-resilient storage for style evaluation experiments."""

    def __init__(self, filepath: Path):
        """Initialize store and create schema if needed.

        Args:
            filepath: Path to SQLite database file
        """
        self.filepath = filepath
        self.conn = sqlite3.connect(filepath)
        self.conn.row_factory = sqlite3.Row  # Dict-like row access

        # Enable foreign key constraints
        self.conn.execute("PRAGMA foreign_keys = ON")

        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        # Table 1: Test samples and flattened content
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                sample_id TEXT PRIMARY KEY,
                original_text TEXT NOT NULL,
                flattened_content TEXT NOT NULL,
                source_info TEXT
            )
        """)

        # Table 2: Reconstructions (4 per sample per run)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS reconstructions (
                sample_id TEXT NOT NULL,
                run INTEGER NOT NULL,
                method TEXT NOT NULL CHECK(method IN ('generic', 'fewshot', 'author', 'instructions')),
                reconstructed_text TEXT NOT NULL,
                reconstruction_model TEXT NOT NULL,
                PRIMARY KEY (sample_id, run, method),
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
            )
        """)

        # Table 3: Comparative judgments (1 per sample per run)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS comparative_judgments (
                sample_id TEXT NOT NULL,
                run INTEGER NOT NULL,

                -- Anonymous rankings (as judge returned them)
                ranking_text_a INTEGER NOT NULL CHECK(ranking_text_a BETWEEN 1 AND 4),
                ranking_text_b INTEGER NOT NULL CHECK(ranking_text_b BETWEEN 1 AND 4),
                ranking_text_c INTEGER NOT NULL CHECK(ranking_text_c BETWEEN 1 AND 4),
                ranking_text_d INTEGER NOT NULL CHECK(ranking_text_d BETWEEN 1 AND 4),

                -- Method mapping (which label = which method)
                method_text_a TEXT NOT NULL CHECK(method_text_a IN ('generic', 'fewshot', 'author', 'instructions')),
                method_text_b TEXT NOT NULL CHECK(method_text_b IN ('generic', 'fewshot', 'author', 'instructions')),
                method_text_c TEXT NOT NULL CHECK(method_text_c IN ('generic', 'fewshot', 'author', 'instructions')),
                method_text_d TEXT NOT NULL CHECK(method_text_d IN ('generic', 'fewshot', 'author', 'instructions')),

                -- Judgment metadata
                confidence TEXT NOT NULL CHECK(confidence IN ('high', 'medium', 'low')),
                reasoning TEXT NOT NULL,
                judge_model TEXT NOT NULL,
                timestamp TEXT NOT NULL,

                PRIMARY KEY (sample_id, run),
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
            )
        """)

        self.conn.commit()

    # ==========================================================================
    # Sample Management
    # ==========================================================================

    def save_sample(
        self,
        sample_id: str,
        original_text: str,
        flattened_content: str,
        source_info: Optional[str] = None
    ):
        """Store a text sample with its flattened content.

        Args:
            sample_id: Unique identifier for this sample
            original_text: The original gold standard text
            flattened_content: Content-only summary (output of style flattening)
            source_info: Optional provenance (e.g., "File 0, para 50-55")
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO samples
            VALUES (?, ?, ?, ?)
        """, (sample_id, original_text, flattened_content, source_info))
        self.conn.commit()

    def get_sample(self, sample_id: str) -> Optional[dict]:
        """Retrieve a sample by ID.

        Args:
            sample_id: Sample identifier

        Returns:
            Dictionary with keys: sample_id, original_text, flattened_content, source_info
            Returns None if not found
        """
        row = self.conn.execute(
            "SELECT * FROM samples WHERE sample_id=?",
            (sample_id,)
        ).fetchone()

        if not row:
            return None

        return {
            'sample_id': row['sample_id'],
            'original_text': row['original_text'],
            'flattened_content': row['flattened_content'],
            'source_info': row['source_info']
        }

    def list_samples(self) -> list[str]:
        """Get all sample IDs in insertion order.

        Returns:
            List of sample IDs
        """
        rows = self.conn.execute(
            "SELECT sample_id FROM samples ORDER BY rowid"
        ).fetchall()
        return [row['sample_id'] for row in rows]

    # ==========================================================================
    # Reconstruction Management
    # ==========================================================================

    def save_reconstruction(
        self,
        sample_id: str,
        run: int,
        method: str,
        reconstructed_text: str,
        model: str
    ):
        """Save a reconstruction result.

        Args:
            sample_id: ID of the sample being reconstructed
            run: Run number (0 to M-1 for M stochastic runs)
            method: Reconstruction method ('generic', 'fewshot', 'author', 'instructions')
            reconstructed_text: The reconstructed text
            model: Model used for reconstruction (e.g., 'mistral-large-2411')

        Raises:
            ValueError: If sample_id doesn't exist
        """
        # Verify sample exists
        if not self.get_sample(sample_id):
            raise ValueError(
                f"Sample '{sample_id}' not found. Save sample first with save_sample()."
            )

        self.conn.execute("""
            INSERT OR REPLACE INTO reconstructions
            VALUES (?, ?, ?, ?, ?)
        """, (sample_id, run, method, reconstructed_text, model))
        self.conn.commit()

    def has_reconstruction(self, sample_id: str, run: int, method: str) -> bool:
        """Check if a reconstruction exists.

        Args:
            sample_id: Sample identifier
            run: Run number
            method: Reconstruction method

        Returns:
            True if reconstruction exists, False otherwise
        """
        row = self.conn.execute("""
            SELECT 1 FROM reconstructions
            WHERE sample_id=? AND run=? AND method=?
        """, (sample_id, run, method)).fetchone()
        return row is not None

    def get_reconstructions(self, sample_id: str, run: int) -> dict[str, str]:
        """Get all 4 reconstructions for a sample/run.

        Args:
            sample_id: Sample identifier
            run: Run number

        Returns:
            Dictionary mapping method to reconstructed text
            Example: {'generic': '...', 'fewshot': '...', 'author': '...', 'instructions': '...'}
        """
        rows = self.conn.execute("""
            SELECT method, reconstructed_text
            FROM reconstructions
            WHERE sample_id=? AND run=?
        """, (sample_id, run)).fetchall()

        return {row['method']: row['reconstructed_text'] for row in rows}

    # ==========================================================================
    # Judgment Management
    # ==========================================================================

    def save_judgment(
        self,
        sample_id: str,
        run: int,
        judgment: 'StyleJudgmentComparative',
        mapping: 'MethodMapping',
        judge_model: str
    ):
        """Save a comparative judgment with method mapping.

        Args:
            sample_id: Sample identifier
            run: Run number
            judgment: StyleJudgmentComparative instance from LLM
            mapping: MethodMapping showing which label corresponds to which method
            judge_model: Model used for judging (e.g., 'claude-sonnet-4-5')

        Raises:
            ValueError: If sample doesn't exist or not all 4 reconstructions exist
        """
        # Verify sample exists
        if not self.get_sample(sample_id):
            raise ValueError(
                f"Sample '{sample_id}' not found. Save sample first."
            )

        # Verify all 4 reconstructions exist
        reconstructions = self.get_reconstructions(sample_id, run)
        if len(reconstructions) != 4:
            missing = set(['generic', 'fewshot', 'author', 'instructions']) - set(reconstructions.keys())
            raise ValueError(
                f"Missing reconstructions for sample '{sample_id}', run {run}: {missing}"
            )

        # Save judgment
        timestamp = datetime.now().isoformat()
        self.conn.execute("""
            INSERT OR REPLACE INTO comparative_judgments
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sample_id,
            run,
            judgment.ranking_text_a,
            judgment.ranking_text_b,
            judgment.ranking_text_c,
            judgment.ranking_text_d,
            mapping.text_a,
            mapping.text_b,
            mapping.text_c,
            mapping.text_d,
            judgment.confidence,
            judgment.reasoning,
            judge_model,
            timestamp
        ))
        self.conn.commit()

    def has_judgment(self, sample_id: str, run: int) -> bool:
        """Check if a judgment exists for a sample/run.

        Args:
            sample_id: Sample identifier
            run: Run number

        Returns:
            True if judgment exists, False otherwise
        """
        row = self.conn.execute("""
            SELECT 1 FROM comparative_judgments
            WHERE sample_id=? AND run=?
        """, (sample_id, run)).fetchone()
        return row is not None

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def create_random_mapping(self, seed: Optional[int] = None) -> 'MethodMapping':
        """Generate random label-to-method mapping for blind evaluation.

        Randomly assigns the 4 methods to labels A, B, C, D. Useful for
        eliminating position bias by varying order across samples/runs.

        Args:
            seed: Optional random seed for deterministic mapping

        Returns:
            MethodMapping instance with randomized assignments

        Example:
            >>> mapping = store.create_random_mapping(seed=42)
            >>> mapping.text_a  # Might be 'fewshot'
            >>> mapping.text_b  # Might be 'generic'
        """
        from belletrist.models import MethodMapping

        methods = ['generic', 'fewshot', 'author', 'instructions']
        if seed is not None:
            random.seed(seed)
        random.shuffle(methods)

        return MethodMapping(
            text_a=methods[0],
            text_b=methods[1],
            text_c=methods[2],
            text_d=methods[3]
        )

    def get_incomplete_work(self, n_runs: int) -> list[tuple[str, int]]:
        """Find (sample_id, run) pairs that need judgments.

        Useful for resume support: identify which work is incomplete.

        Args:
            n_runs: Expected number of runs per sample

        Returns:
            List of (sample_id, run) tuples that don't have judgments yet
        """
        incomplete = []
        for sample_id in self.list_samples():
            for run in range(n_runs):
                if not self.has_judgment(sample_id, run):
                    incomplete.append((sample_id, run))
        return incomplete

    # ==========================================================================
    # Export Methods
    # ==========================================================================

    def to_dataframe(self) -> pd.DataFrame:
        """Export all judgments to DataFrame with method-resolved rankings.

        Resolves anonymous rankings (text_a, text_b, etc.) to actual method
        rankings (ranking_generic, ranking_fewshot, etc.) for analysis.

        Returns:
            DataFrame with columns:
                - sample_id: Sample identifier
                - run: Run number
                - ranking_generic: Rank of generic method (1-4)
                - ranking_fewshot: Rank of fewshot method (1-4)
                - ranking_author: Rank of author method (1-4)
                - ranking_instructions: Rank of instructions method (1-4)
                - confidence: Judge confidence (high/medium/low)
                - reasoning: Judge's explanation
                - judge_model: Model used for judging
                - timestamp: When judgment was made
        """
        rows = self.conn.execute("""
            SELECT * FROM comparative_judgments ORDER BY sample_id, run
        """).fetchall()

        records = []
        for row in rows:
            # Build mapping: method -> ranking
            method_rankings = {}
            for label in ['a', 'b', 'c', 'd']:
                method = row[f'method_text_{label}']
                ranking = row[f'ranking_text_{label}']
                method_rankings[method] = ranking

            records.append({
                'sample_id': row['sample_id'],
                'run': row['run'],
                'ranking_generic': method_rankings['generic'],
                'ranking_fewshot': method_rankings['fewshot'],
                'ranking_author': method_rankings['author'],
                'ranking_instructions': method_rankings['instructions'],
                'confidence': row['confidence'],
                'reasoning': row['reasoning'],
                'judge_model': row['judge_model'],
                'timestamp': row['timestamp']
            })

        return pd.DataFrame(records)

    def to_csv(self, output_path: Path):
        """Export judgments to CSV file.

        Args:
            output_path: Path to write CSV file
        """
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)

    # ==========================================================================
    # Stats Methods
    # ==========================================================================

    def get_stats(self) -> dict:
        """Get statistics about the evaluation progress.

        Returns:
            Dictionary with:
                - n_samples: Total samples
                - n_reconstructions: Total reconstructions saved
                - n_judgments: Total judgments saved
                - completion_rate: Fraction of expected work completed
        """
        n_samples = len(self.list_samples())

        n_reconstructions = self.conn.execute(
            "SELECT COUNT(*) as count FROM reconstructions"
        ).fetchone()['count']

        n_judgments = self.conn.execute(
            "SELECT COUNT(*) as count FROM comparative_judgments"
        ).fetchone()['count']

        # Expected: n_samples × n_runs × 4 methods for reconstructions
        # Expected: n_samples × n_runs for judgments
        # But we don't know n_runs, so we can't compute completion_rate
        # without additional info

        return {
            'n_samples': n_samples,
            'n_reconstructions': n_reconstructions,
            'n_judgments': n_judgments
        }

    # ==========================================================================
    # Reset Methods
    # ==========================================================================

    def reset(self, scope: Literal['all', 'reconstructions_and_judgments', 'judgments_only'] = 'all'):
        """Clear data from the store with hierarchical scope control.

        The data has a clear dependency hierarchy:
            samples (base)
              ↓
            reconstructions (depends on samples via FK)
              ↓
            comparative_judgments (depends on samples via FK)

        Valid reset operations must respect this hierarchy - you cannot delete
        upstream data (e.g., samples) while preserving downstream data (e.g., judgments)
        without violating foreign key constraints.

        Args:
            scope: Reset scope controlling which tables to clear:
                - 'all': Delete everything (samples, reconstructions, judgments)
                - 'reconstructions_and_judgments': Keep samples, delete reconstructions + judgments
                - 'judgments_only': Keep samples + reconstructions, delete only judgments

        Warning:
            This operation is irreversible. All data in the selected scope
            will be permanently deleted.

        Design Notes:
            - Deletion order matters: must delete child tables before parents
            - Cannot delete samples while keeping reconstructions (would violate FKs)
            - Cannot delete reconstructions while keeping judgments (would violate FKs)

        Example:
            >>> store.reset('judgments_only')  # Re-run judging, keep reconstructions
            >>> store.reset('reconstructions_and_judgments')  # Re-run reconstruction + judging
            >>> store.reset('all')  # Fresh start
        """
        if scope == 'all':
            # Delete everything in reverse dependency order
            self.conn.execute("DELETE FROM comparative_judgments")
            self.conn.execute("DELETE FROM reconstructions")
            self.conn.execute("DELETE FROM samples")
        elif scope == 'reconstructions_and_judgments':
            # Keep samples, delete everything downstream
            self.conn.execute("DELETE FROM comparative_judgments")
            self.conn.execute("DELETE FROM reconstructions")
        elif scope == 'judgments_only':
            # Keep samples and reconstructions, delete only judgments
            self.conn.execute("DELETE FROM comparative_judgments")
        else:
            raise ValueError(
                f"Invalid scope: {scope}. "
                f"Must be 'all', 'reconstructions_and_judgments', or 'judgments_only'"
            )

        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

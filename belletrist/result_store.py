"""
Result storage for text samples and multi-analyst outputs.

Uses SQLite with five tables:
- samples: Original text with provenance (file/paragraph indices)
- analyses: Analyst outputs keyed by (sample_id, analyst_type)
- syntheses: Final synthesis outputs (cross-text synthesis, principles guide)
- synthesis_samples: Junction table linking syntheses to their input samples
- synthesis_metadata: Pre-computed metadata for efficient queries
"""
from typing import Optional, Literal, Union
import sqlite3
from pathlib import Path
import json
from datetime import datetime


class ResultStore:
    """SQLite-backed storage for samples and multi-analyst results.

    Manages five tables:
    - samples: Text samples with provenance (where they came from)
    - analyses: Analyst outputs for each sample
    - syntheses: Final synthesis outputs with auto-generated IDs
    - synthesis_samples: Links syntheses to their input samples/analyses
    - synthesis_metadata: Pre-computed metadata for queries

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

        # Save final syntheses with auto-generated IDs
        cross_text_id = store.save_synthesis(
            synthesis_type='cross_text_synthesis',
            output='...',
            model='mistral/mistral-large-2411',
            sample_contributions=[('sample_001', 'cross_perspective_integrator')],
            config=cross_text_config
        )

        # Export to filesystem
        store.export_synthesis(cross_text_id, Path('output.txt'))
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
        # Synthesis tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS syntheses (
                synthesis_id TEXT PRIMARY KEY,
                synthesis_type TEXT NOT NULL CHECK(synthesis_type IN ('cross_text_synthesis', 'principles_guide')),
                output TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL,
                parent_synthesis_id TEXT,
                config_json TEXT NOT NULL,
                FOREIGN KEY (parent_synthesis_id) REFERENCES syntheses(synthesis_id)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS synthesis_samples (
                synthesis_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                analyst TEXT NOT NULL,
                PRIMARY KEY (synthesis_id, sample_id, analyst),
                FOREIGN KEY (synthesis_id) REFERENCES syntheses(synthesis_id) ON DELETE CASCADE,
                FOREIGN KEY (sample_id, analyst) REFERENCES analyses(sample_id, analyst)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS synthesis_metadata (
                synthesis_id TEXT PRIMARY KEY,
                num_samples INTEGER NOT NULL,
                sample_ids_json TEXT NOT NULL,
                is_homogeneous_model INTEGER NOT NULL,
                models_json TEXT NOT NULL,
                FOREIGN KEY (synthesis_id) REFERENCES syntheses(synthesis_id) ON DELETE CASCADE
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

    def reset(self):
        """Clear all data from the store, removing all samples and analyses.

        This performs a complete cleanse of the database, deleting all records
        from both the samples and analyses tables. The schema/table structure
        is preserved, so you can immediately start storing new data.

        Warning:
            This operation is irreversible. All stored samples and analyses
            will be permanently deleted.
        """
        self.conn.execute("DELETE FROM analyses")
        self.conn.execute("DELETE FROM samples")
        self.conn.commit()

    def _get_next_synthesis_id(self, synthesis_type: str) -> str:
        """Generate next synthesis ID for the given type.

        Args:
            synthesis_type: Type of synthesis ('cross_text_synthesis' or 'principles_guide')

        Returns:
            Auto-generated ID like 'cross_text_synthesis_001'
        """
        # Get max counter for this type
        row = self.conn.execute("""
            SELECT synthesis_id FROM syntheses
            WHERE synthesis_type = ?
            ORDER BY synthesis_id DESC
            LIMIT 1
        """, (synthesis_type,)).fetchone()

        if row:
            # Extract counter from existing ID (e.g., 'cross_text_synthesis_001' -> 1)
            last_id = row['synthesis_id']
            counter = int(last_id.split('_')[-1])
            next_counter = counter + 1
        else:
            next_counter = 1

        return f"{synthesis_type}_{next_counter:03d}"

    def save_synthesis(
        self,
        synthesis_type: Literal['cross_text_synthesis', 'principles_guide'],
        output: str,
        model: str,
        sample_contributions: list[tuple[str, str]],
        config: 'BasePromptConfig',
        parent_synthesis_id: Optional[str] = None
    ) -> str:
        """Save a synthesis with auto-generated ID and full provenance.

        Args:
            synthesis_type: Type of synthesis
            output: The synthesis output text
            model: Model used to generate this synthesis
            sample_contributions: List of (sample_id, analyst) tuples that contributed
            config: Pydantic config object used to generate the prompt
            parent_synthesis_id: For principles_guide, the parent cross_text_synthesis ID

        Returns:
            Generated synthesis_id (e.g., 'cross_text_synthesis_001')

        Raises:
            ValueError: If validation fails or parent doesn't exist
        """
        # Validate parent exists if specified
        if parent_synthesis_id:
            parent = self.get_synthesis(parent_synthesis_id)
            if not parent:
                raise ValueError(f"Parent synthesis '{parent_synthesis_id}' not found")

        # Validate all sample contributions exist
        for sample_id, analyst in sample_contributions:
            analysis = self.get_analysis(sample_id, analyst)
            if not analysis:
                raise ValueError(
                    f"Analysis not found: sample_id='{sample_id}', analyst='{analyst}'. "
                    f"Save the analysis first with save_analysis()."
                )

        # Generate ID
        synthesis_id = self._get_next_synthesis_id(synthesis_type)

        # Serialize config
        config_dict = config.model_dump() if hasattr(config, 'model_dump') else config.dict()
        config_json = json.dumps(config_dict)

        # Save synthesis
        created_at = datetime.now().isoformat()
        self.conn.execute("""
            INSERT INTO syntheses
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            synthesis_id,
            synthesis_type,
            output,
            model,
            created_at,
            parent_synthesis_id,
            config_json
        ))

        # Save sample contributions
        for sample_id, analyst in sample_contributions:
            self.conn.execute("""
                INSERT INTO synthesis_samples
                VALUES (?, ?, ?)
            """, (synthesis_id, sample_id, analyst))

        # Compute and save metadata
        if sample_contributions:
            # Get unique sample IDs
            sample_ids = list(set(s[0] for s in sample_contributions))
            num_samples = len(sample_ids)

            # Check model homogeneity
            models = set()
            models.add(model)  # Include synthesis model
            for sample_id, analyst in sample_contributions:
                row = self.conn.execute(
                    "SELECT model FROM analyses WHERE sample_id=? AND analyst=?",
                    (sample_id, analyst)
                ).fetchone()
                if row:
                    models.add(row['model'])

            is_homogeneous = len(models) == 1
        else:
            # Principles guide with no direct contributions (inherited from parent)
            sample_ids = []
            num_samples = 0
            is_homogeneous = True
            models = {model}

        self.conn.execute("""
            INSERT INTO synthesis_metadata
            VALUES (?, ?, ?, ?, ?)
        """, (
            synthesis_id,
            num_samples,
            json.dumps(sample_ids),
            1 if is_homogeneous else 0,
            json.dumps(list(models))
        ))

        self.conn.commit()
        return synthesis_id

    def get_synthesis(self, synthesis_id: str) -> Optional[dict]:
        """Retrieve a synthesis by ID.

        Args:
            synthesis_id: Synthesis identifier

        Returns:
            Dictionary with synthesis data or None if not found.
            Keys: synthesis_id, type, output, model, created_at, parent_id, config
        """
        row = self.conn.execute("""
            SELECT synthesis_id, synthesis_type, output, model, created_at,
                   parent_synthesis_id, config_json
            FROM syntheses
            WHERE synthesis_id=?
        """, (synthesis_id,)).fetchone()

        if not row:
            return None

        return {
            'synthesis_id': row['synthesis_id'],
            'type': row['synthesis_type'],
            'output': row['output'],
            'model': row['model'],
            'created_at': row['created_at'],
            'parent_id': row['parent_synthesis_id'],
            'config': json.loads(row['config_json'])
        }

    def get_synthesis_with_metadata(self, synthesis_id: str) -> Optional[dict]:
        """Retrieve synthesis with computed metadata.

        Args:
            synthesis_id: Synthesis identifier

        Returns:
            Dictionary with synthesis data plus metadata, or None if not found.
            Additional metadata keys: num_samples, sample_ids, is_homogeneous_model, models_used
        """
        synthesis = self.get_synthesis(synthesis_id)
        if not synthesis:
            return None

        # Get metadata
        meta_row = self.conn.execute("""
            SELECT num_samples, sample_ids_json, is_homogeneous_model, models_json
            FROM synthesis_metadata
            WHERE synthesis_id=?
        """, (synthesis_id,)).fetchone()

        if meta_row:
            synthesis['metadata'] = {
                'num_samples': meta_row['num_samples'],
                'sample_ids': json.loads(meta_row['sample_ids_json']),
                'is_homogeneous_model': bool(meta_row['is_homogeneous_model']),
                'models_used': json.loads(meta_row['models_json'])
            }
        else:
            synthesis['metadata'] = {}

        return synthesis

    def list_syntheses(
        self,
        synthesis_type: Optional[str] = None
    ) -> list[str]:
        """List synthesis IDs, optionally filtered by type.

        Args:
            synthesis_type: Optional filter ('cross_text_synthesis' or 'principles_guide')

        Returns:
            List of synthesis IDs in creation order
        """
        if synthesis_type:
            rows = self.conn.execute("""
                SELECT synthesis_id FROM syntheses
                WHERE synthesis_type = ?
                ORDER BY created_at
            """, (synthesis_type,)).fetchall()
        else:
            rows = self.conn.execute("""
                SELECT synthesis_id FROM syntheses
                ORDER BY created_at
            """).fetchall()

        return [row['synthesis_id'] for row in rows]

    def get_synthesis_provenance(self, synthesis_id: str) -> Optional[dict]:
        """Get full provenance tree for a synthesis.

        Args:
            synthesis_id: Synthesis identifier

        Returns:
            Dictionary with synthesis info and recursive parent provenance, or None if not found.
            Structure:
            {
                'synthesis': {...},
                'sample_contributions': [(sample_id, analyst), ...],
                'parent': {...} or None  # Recursive structure for parent
            }
        """
        synthesis = self.get_synthesis(synthesis_id)
        if not synthesis:
            return None

        # Get sample contributions
        contrib_rows = self.conn.execute("""
            SELECT sample_id, analyst
            FROM synthesis_samples
            WHERE synthesis_id = ?
        """, (synthesis_id,)).fetchall()

        sample_contributions = [(row['sample_id'], row['analyst']) for row in contrib_rows]

        result = {
            'synthesis': synthesis,
            'sample_contributions': sample_contributions,
            'parent': None
        }

        # Recursively get parent provenance
        if synthesis['parent_id']:
            result['parent'] = self.get_synthesis_provenance(synthesis['parent_id'])

        return result

    def export_synthesis(
        self,
        synthesis_id: str,
        output_path: Path,
        metadata_format: Literal['yaml', 'json'] = 'yaml'
    ) -> None:
        """Export synthesis to filesystem with metadata header.

        Args:
            synthesis_id: Synthesis identifier
            output_path: Path to write output file
            metadata_format: Format for metadata header ('yaml' or 'json')

        Raises:
            ValueError: If synthesis not found
        """
        synthesis = self.get_synthesis_with_metadata(synthesis_id)
        if not synthesis:
            raise ValueError(f"Synthesis '{synthesis_id}' not found")

        # Build metadata header
        metadata = {
            'synthesis_id': synthesis['synthesis_id'],
            'synthesis_type': synthesis['type'],
            'model': synthesis['model'],
            'created_at': synthesis['created_at'],
        }

        if synthesis['parent_id']:
            metadata['parent_synthesis_id'] = synthesis['parent_id']

        if synthesis.get('metadata'):
            metadata.update({
                'num_samples': synthesis['metadata']['num_samples'],
                'sample_ids': synthesis['metadata']['sample_ids'],
                'is_homogeneous_model': synthesis['metadata']['is_homogeneous_model'],
                'models_used': synthesis['metadata']['models_used']
            })

        # Format metadata header
        if metadata_format == 'yaml':
            header = "---\n"
            for key, value in metadata.items():
                if isinstance(value, list):
                    header += f"{key}:\n"
                    for item in value:
                        header += f"  - {item}\n"
                elif isinstance(value, bool):
                    header += f"{key}: {str(value).lower()}\n"
                else:
                    header += f"{key}: {value}\n"
            header += "---\n\n"
        else:  # json
            header = json.dumps(metadata, indent=2) + "\n\n---\n\n"

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(synthesis['output'])

    def close(self):
        """Close database connection."""
        self.conn.close()

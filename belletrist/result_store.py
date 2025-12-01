"""
Persistent storage for multi-stage LLM text analysis workflows.

This module implements a tailored database schema for the specific workflow of
analyzing text samples through multiple specialist LLM agents, integrating their
findings, and synthesizing cross-text patterns. It is NOT a general-purpose LLM
workflow memory system.

## Design Philosophy

The core problem: Running LLMs is expensive and time-consuming. When a multi-stage
analysis pipeline involves:
1. Sampling text → 2. Running 5+ specialist analyses per sample →
3. Cross-perspective integration → 4. Cross-text synthesis → 5. Principles extraction

...we need to cache intermediate results to enable:
- Iterative development without re-running expensive LLM calls
- Reproducibility through full provenance tracking
- Partial re-runs (e.g., re-do synthesis without re-analyzing samples)
- Validation that all required analyses completed before proceeding

## Four-Tier Data Model

The schema mirrors the workflow's natural hierarchy:

**Tier 1: Samples** (base layer)
- Original text segments with source provenance (file index, paragraph range)
- Provenance enables exact reproducibility and source attribution
- Example: "sample_001" = paragraphs 372-377 from file 1

**Tier 2: Analyses** (per-sample, per-analyst)
- Multiple specialist analyses per sample (rhetorician, syntactician, etc.)
- Keyed by (sample_id, analyst_name) composite key
- Each analysis stores: output text + model used
- Foreign key constraint: analyses must reference existing samples

**Tier 3: Syntheses** (multi-sample aggregations)
- Cross-text synthesis: aggregates multiple per-sample integrations
- Principles guide: converts synthesis into prescriptive style instructions
- Field guide: unified recognition criteria for passage evaluation
- Auto-generated IDs prevent naming collisions (e.g., 'cross_text_synthesis_001')
- Parent linkage: principles_guide references its parent cross_text_synthesis
- Full provenance: tracks which samples/analyses contributed to each synthesis

**Tier 4: Passage Evaluations & Example Sets** (corpus mining)
- Passage evaluations: density ratings (1-5) of passages using field guide rubric
- Passage metadata: tasks demonstrated, techniques present, learning value
- Example sets: curated collections of high-density passages for few-shot learning
- Enables transition from analytical extraction to tacit transmission via examples

## Key Design Decisions

**1. Auto-Generated Synthesis IDs**
Unlike samples (user provides 'sample_001'), syntheses get auto-IDs to avoid
naming conflicts when iterating. The pattern {type}_{counter:03d} ensures
sequential versioning (cross_text_synthesis_001, _002, etc.).

**2. Metadata Inheritance**
Principles guides don't directly consume samples - they consume cross-text
syntheses. Yet we want to know which samples ultimately contributed. Solution:
- Store empty sample_contributions for principles guides in DB
- Inherit parent's sample metadata when exporting to filesystem
- Maintains clean relational model while providing useful exported metadata

**3. Hierarchical Reset**
Three reset scopes respect foreign key constraints:
- 'all': Delete everything (clean slate)
- 'analyses_and_syntheses': Keep samples, re-run analyses
- 'syntheses_only': Keep samples + analyses, re-run syntheses
Cannot delete upstream data while preserving downstream (would violate FKs).

**4. Provenance as First-Class Concern**
Every sample tracks exact source location. Every analysis tracks model used.
Every synthesis tracks contributing samples/analyses + parent linkage.
This enables full audit trails: "Which text produced this style principle?"

## SQLite Schema (9 Tables)

samples:             sample_id (PK) | text | file_index | paragraph_start | paragraph_end
analyses:            sample_id (FK) | analyst (PK) | output | model
syntheses:           synthesis_id (PK) | synthesis_type | output | model | created_at | parent_id (FK) | config_json
synthesis_samples:   synthesis_id (FK) | sample_id (FK) | analyst (FK)
synthesis_metadata:  synthesis_id (FK) | num_samples | sample_ids_json | is_homogeneous_model | models_json
passage_evaluations: evaluation_id (PK) | sample_id (FK) | paragraph_range | density_rating | task_coverage | teaching_value | recommendation | evaluator_model | field_guide_id (FK) | created_at
passage_metadata:    evaluation_id (PK/FK) | tasks_json | techniques_json | difficulty_level | learning_value
example_sets:        set_id (PK) | set_name | description | purpose | field_guide_id (FK) | created_at | curator_model
example_set_members: set_id (FK) | evaluation_id (FK) | position

## Usage Pattern

```python
from belletrist import DataSampler, ResultStore

sampler = DataSampler()
store = ResultStore(Path("results.db"))

# Stage 1: Sample text
segment = sampler.sample_segment(5)
store.save_segment("sample_001", segment)

# Stage 2: Run specialist analyses
for analyst in ['rhetorician', 'syntactician']:
    response = llm.complete(analyst_prompt)
    store.save_analysis("sample_001", analyst, response.content, response.model)

# Stage 3: Check completeness before proceeding
if store.is_complete("sample_001", required_analysts):
    sample, analyses = store.get_sample_with_analyses("sample_001")
    # Proceed to integration...

# Stage 4: Save cross-text synthesis
cross_text_id = store.save_synthesis(
    synthesis_type='cross_text_synthesis',
    output=synthesis_output,
    model='mistral-large-2411',
    sample_contributions=[('sample_001', 'cross_perspective_integrator')],
    config=cross_text_config
)

# Stage 5: Save principles guide (inherits provenance)
principles_id = store.save_synthesis(
    synthesis_type='principles_guide',
    output=principles_output,
    model='mistral-large-2411',
    sample_contributions=[],  # Empty - inherits from parent
    config=principles_config,
    parent_synthesis_id=cross_text_id
)

# Export with full metadata
store.export_synthesis(principles_id, Path('style_guide.txt'))
```

## Limitations & Scope

This is NOT:
- A general-purpose vector database for semantic search
- A conversation memory system for chatbots
- A tool for managing arbitrary LLM workflows

This IS:
- A cache for expensive multi-stage analysis pipelines
- A provenance tracker for reproducible research
- A schema tailored to the specific workflow: sample → analyze → integrate → synthesize
"""
from typing import Optional, Literal
import sqlite3
from pathlib import Path
import json
from datetime import datetime


class ResultStore:
    """SQLite-backed storage for multi-stage text analysis workflows.

    See module docstring for full design philosophy and data model details.

    This class provides CRUD operations for three data tiers:
    1. Samples: save_sample(), get_sample(), list_samples()
    2. Analyses: save_analysis(), get_analysis(), is_complete()
    3. Syntheses: save_synthesis(), get_synthesis(), export_synthesis()

    All methods return dicts (not custom objects) for lightweight integration.
    Foreign key constraints ensure referential integrity at the database level.
    """

    def __init__(self, filepath: Path):
        """Initialize store and create schema if needed.

        Args:
            filepath: Path to SQLite database file
        """
        self.filepath = filepath
        self.conn = sqlite3.connect(filepath)
        self.conn.row_factory = sqlite3.Row  # Dict-like row access

        # Enable foreign key constraints (disabled by default in SQLite)
        self.conn.execute("PRAGMA foreign_keys = ON")

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
                synthesis_type TEXT NOT NULL CHECK(synthesis_type IN (
                    'cross_text_synthesis',
                    'principles_guide',
                    'implied_author_synthesis',
                    'decision_pattern_synthesis',
                    'textural_synthesis',
                    'field_guide',
                    'author_model_definition',
                    'example_set'
                )),
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
        # Passage evaluation tables (Tier 4)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS passage_evaluations (
                evaluation_id TEXT PRIMARY KEY,
                sample_id TEXT NOT NULL,
                paragraph_range TEXT,
                density_rating INTEGER CHECK(density_rating BETWEEN 1 AND 5),
                task_coverage TEXT,
                teaching_value TEXT,
                recommendation INTEGER CHECK(recommendation IN (0, 1)),
                evaluator_model TEXT NOT NULL,
                field_guide_id TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id),
                FOREIGN KEY (field_guide_id) REFERENCES syntheses(synthesis_id)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS passage_metadata (
                evaluation_id TEXT PRIMARY KEY,
                tasks_json TEXT,
                techniques_json TEXT,
                difficulty_level TEXT CHECK(difficulty_level IN ('beginner', 'intermediate', 'advanced')),
                learning_value TEXT,
                FOREIGN KEY (evaluation_id) REFERENCES passage_evaluations(evaluation_id) ON DELETE CASCADE
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS example_sets (
                set_id TEXT PRIMARY KEY,
                set_name TEXT NOT NULL UNIQUE,
                description TEXT,
                purpose TEXT,
                field_guide_id TEXT,
                created_at TEXT NOT NULL,
                curator_model TEXT NOT NULL,
                FOREIGN KEY (field_guide_id) REFERENCES syntheses(synthesis_id)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS example_set_members (
                set_id TEXT NOT NULL,
                evaluation_id TEXT NOT NULL,
                position INTEGER,
                PRIMARY KEY (set_id, evaluation_id),
                FOREIGN KEY (set_id) REFERENCES example_sets(set_id) ON DELETE CASCADE,
                FOREIGN KEY (evaluation_id) REFERENCES passage_evaluations(evaluation_id) ON DELETE CASCADE
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

    def reset(self, scope: Literal['all', 'analyses_and_syntheses', 'syntheses_only', 'passage_evaluations_only'] = 'all'):
        """Clear data from the store with hierarchical scope control.

        The data has a clear dependency hierarchy:
            samples (base)
              ↓
            analyses (depends on samples via FK)
              ↓
            synthesis_samples (junction: depends on analyses via composite FK)
              ↓
            syntheses (can have parent_synthesis_id FK to itself)
              ↓
            synthesis_metadata (depends on syntheses via FK)
              ↓
            passage_evaluations (depends on samples and field_guide syntheses)
              ↓
            passage_metadata (depends on passage_evaluations via FK)
            example_set_members (junction: depends on example_sets and passage_evaluations)
              ↓
            example_sets (can depend on field_guide syntheses)

        Valid reset operations must respect this hierarchy - you cannot delete
        upstream data (e.g., samples) while preserving downstream data (e.g., syntheses)
        without violating foreign key constraints.

        Args:
            scope: Reset scope controlling which tables to clear:
                - 'all': Delete everything (samples, analyses, syntheses, evaluations, sets)
                - 'analyses_and_syntheses': Keep samples, delete analyses + syntheses + evaluations + sets
                - 'syntheses_only': Keep samples + analyses, delete syntheses + evaluations + sets
                - 'passage_evaluations_only': Keep samples + analyses + syntheses, delete evaluations + sets

        Warning:
            This operation is irreversible. All data in the selected scope
            will be permanently deleted.

        Design Notes:
            - Deletion order matters: must delete child tables before parents
            - passage_metadata and example_set_members have ON DELETE CASCADE
            - synthesis_samples and synthesis_metadata have ON DELETE CASCADE,
              so they're automatically cleaned when syntheses are deleted
            - Cannot delete samples while keeping analyses (would violate FKs)
            - Cannot delete analyses while keeping syntheses (would violate FKs)
        """
        if scope == 'all':
            # Delete everything in reverse dependency order
            # Cascading deletes handle: passage_metadata, example_set_members,
            # synthesis_metadata, synthesis_samples
            self.conn.execute("DELETE FROM example_sets")
            self.conn.execute("DELETE FROM passage_evaluations")
            self.conn.execute("DELETE FROM syntheses")
            self.conn.execute("DELETE FROM analyses")
            self.conn.execute("DELETE FROM samples")
        elif scope == 'analyses_and_syntheses':
            # Keep samples, delete everything downstream
            self.conn.execute("DELETE FROM example_sets")
            self.conn.execute("DELETE FROM passage_evaluations")
            self.conn.execute("DELETE FROM syntheses")
            self.conn.execute("DELETE FROM analyses")
        elif scope == 'syntheses_only':
            # Keep samples and analyses, delete syntheses and downstream
            self.conn.execute("DELETE FROM example_sets")
            self.conn.execute("DELETE FROM passage_evaluations")
            self.conn.execute("DELETE FROM syntheses")
        elif scope == 'passage_evaluations_only':
            # Keep samples, analyses, and syntheses; delete only passage evaluations and example sets
            self.conn.execute("DELETE FROM example_sets")
            self.conn.execute("DELETE FROM passage_evaluations")
        else:
            raise ValueError(
                f"Invalid scope: {scope}. Must be 'all', 'analyses_and_syntheses', "
                f"'syntheses_only', or 'passage_evaluations_only'"
            )

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
        synthesis_type: Literal[
            'cross_text_synthesis',
            'principles_guide',
            'implied_author_synthesis',
            'decision_pattern_synthesis',
            'textural_synthesis',
            'field_guide',
            'example_set'
        ],
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

        Metadata inheritance: If a synthesis has empty sample_contributions (e.g.,
        principles_guide) and a parent_id, it inherits num_samples and sample_ids
        from its parent during export. This provides useful provenance in exported
        files while maintaining clean relational structure in the database.

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

        # Handle metadata - inherit from parent if empty
        synth_metadata = synthesis.get('metadata', {})
        if synth_metadata.get('num_samples', 0) == 0 and synthesis['parent_id']:
            # Inherit sample metadata from parent
            parent = self.get_synthesis_with_metadata(synthesis['parent_id'])
            if parent and parent.get('metadata'):
                parent_metadata = parent['metadata']
                metadata.update({
                    'num_samples': parent_metadata['num_samples'],
                    'sample_ids': parent_metadata['sample_ids'],
                    'is_homogeneous_model': synth_metadata.get('is_homogeneous_model', True),
                    'models_used': synth_metadata.get('models_used', [synthesis['model']])
                })
            else:
                # Fallback to current metadata if parent has none
                metadata.update({
                    'num_samples': synth_metadata.get('num_samples', 0),
                    'sample_ids': synth_metadata.get('sample_ids', []),
                    'is_homogeneous_model': synth_metadata.get('is_homogeneous_model', True),
                    'models_used': synth_metadata.get('models_used', [synthesis['model']])
                })
        elif synth_metadata:
            metadata.update({
                'num_samples': synth_metadata['num_samples'],
                'sample_ids': synth_metadata['sample_ids'],
                'is_homogeneous_model': synth_metadata['is_homogeneous_model'],
                'models_used': synth_metadata['models_used']
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

    # =========================================================================
    # Passage Evaluation Methods (Tier 4)
    # =========================================================================

    def _get_next_passage_eval_id(self) -> str:
        """Generate next passage evaluation ID.

        Returns:
            Auto-generated ID like 'passage_eval_001'
        """
        row = self.conn.execute("""
            SELECT evaluation_id FROM passage_evaluations
            ORDER BY evaluation_id DESC
            LIMIT 1
        """).fetchone()

        if row:
            last_id = row['evaluation_id']
            counter = int(last_id.split('_')[-1])
            next_counter = counter + 1
        else:
            next_counter = 1

        return f"passage_eval_{next_counter:03d}"

    def _get_next_example_set_id(self) -> str:
        """Generate next example set ID.

        Returns:
            Auto-generated ID like 'example_set_001'
        """
        row = self.conn.execute("""
            SELECT set_id FROM example_sets
            ORDER BY set_id DESC
            LIMIT 1
        """).fetchone()

        if row:
            last_id = row['set_id']
            counter = int(last_id.split('_')[-1])
            next_counter = counter + 1
        else:
            next_counter = 1

        return f"example_set_{next_counter:03d}"

    def save_passage_evaluation(
        self,
        sample_id: str,
        density_rating: int,
        task_coverage: str,
        teaching_value: str,
        recommendation: bool,
        model: str,
        field_guide_id: Optional[str] = None,
        paragraph_range: Optional[str] = None
    ) -> str:
        """Save a passage evaluation with auto-generated ID.

        Args:
            sample_id: ID of the sample being evaluated
            density_rating: Density rating from 1-5
            task_coverage: Tasks demonstrated (text or JSON)
            teaching_value: Qualitative assessment
            recommendation: Boolean indicating suitability as example
            model: Model used for evaluation
            field_guide_id: Optional FK to field guide synthesis
            paragraph_range: Optional paragraph range (e.g., "5-7")

        Returns:
            Generated evaluation_id

        Raises:
            ValueError: If sample_id doesn't exist or rating invalid
        """
        # Validate sample exists
        if not self.get_sample(sample_id):
            raise ValueError(
                f"Sample '{sample_id}' not found. Save sample first with save_sample()."
            )

        # Validate field guide exists if specified
        if field_guide_id:
            guide = self.get_synthesis(field_guide_id)
            if not guide:
                raise ValueError(f"Field guide '{field_guide_id}' not found")

        # Validate rating
        if not (1 <= density_rating <= 5):
            raise ValueError(f"Density rating must be 1-5, got {density_rating}")

        # Generate ID
        evaluation_id = self._get_next_passage_eval_id()

        # Save evaluation
        created_at = datetime.now().isoformat()
        self.conn.execute("""
            INSERT INTO passage_evaluations
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            evaluation_id,
            sample_id,
            paragraph_range,
            density_rating,
            task_coverage,
            teaching_value,
            1 if recommendation else 0,
            model,
            field_guide_id,
            created_at
        ))

        self.conn.commit()
        return evaluation_id

    def get_passage_evaluation(self, evaluation_id: str) -> Optional[dict]:
        """Retrieve a passage evaluation by ID.

        Args:
            evaluation_id: Evaluation identifier

        Returns:
            Dictionary with evaluation data or None if not found
        """
        row = self.conn.execute("""
            SELECT evaluation_id, sample_id, paragraph_range, density_rating,
                   task_coverage, teaching_value, recommendation, evaluator_model,
                   field_guide_id, created_at
            FROM passage_evaluations
            WHERE evaluation_id=?
        """, (evaluation_id,)).fetchone()

        if not row:
            return None

        return {
            'evaluation_id': row['evaluation_id'],
            'sample_id': row['sample_id'],
            'paragraph_range': row['paragraph_range'],
            'density_rating': row['density_rating'],
            'task_coverage': row['task_coverage'],
            'teaching_value': row['teaching_value'],
            'recommendation': bool(row['recommendation']),
            'evaluator_model': row['evaluator_model'],
            'field_guide_id': row['field_guide_id'],
            'created_at': row['created_at']
        }

    def list_passage_evaluations(self) -> list[str]:
        """Get all passage evaluation IDs in creation order.

        Returns:
            List of evaluation IDs
        """
        rows = self.conn.execute("""
            SELECT evaluation_id FROM passage_evaluations
            ORDER BY created_at
        """).fetchall()

        return [row['evaluation_id'] for row in rows]

    def is_passage_evaluated(self, sample_id: str) -> bool:
        """Check if a passage evaluation exists for a sample.

        Args:
            sample_id: Sample identifier

        Returns:
            True if evaluation exists, False otherwise
        """
        row = self.conn.execute("""
            SELECT 1 FROM passage_evaluations
            WHERE sample_id=?
            LIMIT 1
        """, (sample_id,)).fetchone()

        return row is not None

    def get_passages_by_density(self, min_rating: int) -> list[dict]:
        """Get passage evaluations with rating >= min_rating.

        Args:
            min_rating: Minimum density rating (1-5)

        Returns:
            List of evaluation dicts sorted by rating (highest first)
        """
        rows = self.conn.execute("""
            SELECT evaluation_id, sample_id, paragraph_range, density_rating,
                   task_coverage, teaching_value, recommendation, evaluator_model,
                   field_guide_id, created_at
            FROM passage_evaluations
            WHERE density_rating >= ?
            ORDER BY density_rating DESC, created_at
        """, (min_rating,)).fetchall()

        return [{
            'evaluation_id': row['evaluation_id'],
            'sample_id': row['sample_id'],
            'paragraph_range': row['paragraph_range'],
            'density_rating': row['density_rating'],
            'task_coverage': row['task_coverage'],
            'teaching_value': row['teaching_value'],
            'recommendation': bool(row['recommendation']),
            'evaluator_model': row['evaluator_model'],
            'field_guide_id': row['field_guide_id'],
            'created_at': row['created_at']
        } for row in rows]

    def save_passage_metadata(
        self,
        evaluation_id: str,
        tasks: Optional[list[str]] = None,
        techniques: Optional[list[str]] = None,
        difficulty_level: Optional[Literal['beginner', 'intermediate', 'advanced']] = None,
        learning_value: Optional[str] = None
    ):
        """Save metadata for a passage evaluation.

        Args:
            evaluation_id: Evaluation identifier
            tasks: List of tasks demonstrated
            techniques: List of techniques present
            difficulty_level: Difficulty classification
            learning_value: Description of learning value

        Raises:
            ValueError: If evaluation_id doesn't exist
        """
        # Validate evaluation exists
        if not self.get_passage_evaluation(evaluation_id):
            raise ValueError(f"Evaluation '{evaluation_id}' not found")

        # Serialize lists to JSON
        tasks_json = json.dumps(tasks) if tasks else None
        techniques_json = json.dumps(techniques) if techniques else None

        self.conn.execute("""
            INSERT OR REPLACE INTO passage_metadata
            VALUES (?, ?, ?, ?, ?)
        """, (
            evaluation_id,
            tasks_json,
            techniques_json,
            difficulty_level,
            learning_value
        ))

        self.conn.commit()

    def get_passage_metadata(self, evaluation_id: str) -> Optional[dict]:
        """Retrieve metadata for a passage evaluation.

        Args:
            evaluation_id: Evaluation identifier

        Returns:
            Dictionary with metadata or None if not found
        """
        row = self.conn.execute("""
            SELECT evaluation_id, tasks_json, techniques_json, difficulty_level, learning_value
            FROM passage_metadata
            WHERE evaluation_id=?
        """, (evaluation_id,)).fetchone()

        if not row:
            return None

        return {
            'evaluation_id': row['evaluation_id'],
            'tasks': json.loads(row['tasks_json']) if row['tasks_json'] else None,
            'techniques': json.loads(row['techniques_json']) if row['techniques_json'] else None,
            'difficulty_level': row['difficulty_level'],
            'learning_value': row['learning_value']
        }

    def save_example_set(
        self,
        name: str,
        description: str,
        purpose: str,
        passage_ids: list[str],
        model: str,
        field_guide_id: Optional[str] = None
    ) -> str:
        """Save an example set with auto-generated ID.

        Args:
            name: Unique name for the set
            description: Description of the set
            purpose: Purpose (e.g., "argumentative", "expository")
            passage_ids: List of evaluation_ids to include
            model: Model used to curate the set
            field_guide_id: Optional FK to field guide

        Returns:
            Generated set_id

        Raises:
            ValueError: If name already exists or passage_ids invalid
        """
        # Validate all passage IDs exist
        for eval_id in passage_ids:
            if not self.get_passage_evaluation(eval_id):
                raise ValueError(f"Passage evaluation '{eval_id}' not found")

        # Validate field guide exists if specified
        if field_guide_id:
            guide = self.get_synthesis(field_guide_id)
            if not guide:
                raise ValueError(f"Field guide '{field_guide_id}' not found")

        # Generate ID
        set_id = self._get_next_example_set_id()

        # Save example set
        created_at = datetime.now().isoformat()
        self.conn.execute("""
            INSERT INTO example_sets
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            set_id,
            name,
            description,
            purpose,
            field_guide_id,
            created_at,
            model
        ))

        # Save set members with positions
        for position, eval_id in enumerate(passage_ids, 1):
            self.conn.execute("""
                INSERT INTO example_set_members
                VALUES (?, ?, ?)
            """, (set_id, eval_id, position))

        self.conn.commit()
        return set_id

    def get_example_set(self, set_id: str) -> Optional[dict]:
        """Retrieve an example set by ID (without passages).

        Args:
            set_id: Set identifier

        Returns:
            Dictionary with set data or None if not found
        """
        row = self.conn.execute("""
            SELECT set_id, set_name, description, purpose, field_guide_id, created_at, curator_model
            FROM example_sets
            WHERE set_id=?
        """, (set_id,)).fetchone()

        if not row:
            return None

        # Get member evaluation IDs
        member_rows = self.conn.execute("""
            SELECT evaluation_id FROM example_set_members
            WHERE set_id=?
            ORDER BY position
        """, (set_id,)).fetchall()

        return {
            'set_id': row['set_id'],
            'name': row['set_name'],
            'description': row['description'],
            'purpose': row['purpose'],
            'field_guide_id': row['field_guide_id'],
            'created_at': row['created_at'],
            'curator_model': row['curator_model'],
            'evaluation_ids': [r['evaluation_id'] for r in member_rows]
        }

    def get_example_set_with_passages(self, set_id: str) -> Optional[dict]:
        """Retrieve an example set with full passage data.

        Args:
            set_id: Set identifier

        Returns:
            Dictionary with set data and passages, or None if not found
        """
        example_set = self.get_example_set(set_id)
        if not example_set:
            return None

        # Get full passage data for each member
        passages = []
        for eval_id in example_set['evaluation_ids']:
            evaluation = self.get_passage_evaluation(eval_id)
            if evaluation:
                # Also get the sample text
                sample = self.get_sample(evaluation['sample_id'])
                evaluation['text'] = sample['text'] if sample else None
                # Get metadata if exists
                metadata = self.get_passage_metadata(eval_id)
                if metadata:
                    evaluation['metadata'] = metadata
                passages.append(evaluation)

        example_set['passages'] = passages
        return example_set

    def list_example_sets(self) -> list[str]:
        """Get all example set IDs in creation order.

        Returns:
            List of set IDs
        """
        rows = self.conn.execute("""
            SELECT set_id FROM example_sets
            ORDER BY created_at
        """).fetchall()

        return [row['set_id'] for row in rows]

    def close(self):
        """Close database connection."""
        self.conn.close()

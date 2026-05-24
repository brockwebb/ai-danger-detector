# ADD Evidence Data I/O Design

Date: 2026-05-24

## Purpose

The evidence corpus currently works as an in-memory backend object. The next useful slice is a small data I/O layer that can load source registry entries and evidence units from versioned local files, validate links between them, and return an `EvidenceCorpus` ready for summaries, splits, snapshots, and later calibration runs.

This slice does not collect external evidence, train models, run calibration, or define final data sources. It creates the machinery needed to make future evidence sets reproducible and auditable.

## Scope

Build `core_model/evidence_io.py` with:

- `load_source_registry(path)`: load a JSON array of source registry entries.
- `load_evidence_units(path)`: load JSON Lines evidence records.
- `load_corpus(source_path, evidence_path)`: load a registry and corpus together, validating that each evidence record points to a known source.
- `EvidenceLoadError`: a structured exception that includes file path and optional line number context.

Update exports, tests, README, and the evidence architecture document. Add a tiny local example dataset only if it is needed to document expected format; otherwise keep fixture data in tests for this slice.

## File Formats

Source registry files are JSON arrays. Each object maps directly to `SourceRegistry.add_source` fields and may include an optional `status` and `status_reason`.

```json
[
  {
    "source_id": "src-active",
    "source_name": "Active adjudicated set",
    "source_type": "case_set",
    "owner_or_publisher": "ADD",
    "license_or_access": "private",
    "update_cadence": "one-time",
    "coverage": ["health"],
    "known_biases": [],
    "quality_tier": "tier_1",
    "status": "active",
    "status_reason": "approved"
  }
]
```

Evidence files are JSON Lines. Each nonblank line is one object that maps directly to `EvidenceUnit`.

```jsonl
{"evidence_id":"case-001","source_id":"src-active","evidence_type":"case_review","collection_date":"2026-05-24","event_date":"2026-05-20","domain":"health","task_type":"medical symptom advice","model_family":"unknown","model_version":"unknown","user_expertise":"non_expert","governance_context":"informal","outcome_label":"harm","oversight_label":"expert_review_required","harm_severity":0.8,"detectability":0.2,"reversibility":0.3,"verification_burden":0.9,"workflow_path":["S0","S1","S2","S6","S8"],"confidence":0.7,"source_quality":"tier_1","bias_notes":[],"relevance_limits":[]}
```

## Behavior

The loader should be strict enough to catch bad data early but small enough to remain easy to inspect.

- Missing files raise `EvidenceLoadError`.
- Malformed JSON raises `EvidenceLoadError` with path and line number when the source is JSONL.
- Unknown enum values raise `EvidenceLoadError`.
- Duplicate source IDs and duplicate evidence IDs reuse the existing registry and corpus validation, wrapped with file context.
- Evidence records referencing missing sources raise `EvidenceLoadError`.
- Blank JSONL lines are ignored.
- List fields are accepted and converted by existing schema constructors where applicable.

## Components

`core_model/evidence_io.py` is the only new implementation file.

Internal helpers:

- `_read_json_array(path)`: load and validate a top-level JSON array.
- `_enum(enum_type, value, field_name, path, line_number=None)`: convert string values into enum members with useful errors.
- `_load_source(entry, registry, path)`: add one source entry and apply optional status.
- `_load_evidence(line_data, path, line_number)`: create one `EvidenceUnit`.
- `_wrap_load_error(path, line_number, message, cause=None)`: keep error text consistent.

Public API:

- `EvidenceLoadError`
- `load_source_registry`
- `load_evidence_units`
- `load_corpus`

## Data Flow

`load_corpus(source_path, evidence_path)` loads sources first, then evidence units, then adds each unit to an `EvidenceCorpus`. Source validation happens at corpus insertion time. Callers can immediately use existing corpus methods such as `coverage_summary()`, `create_split()`, and `create_snapshot()`.

No external network or database access is involved.

## Testing

Tests live in `tests/test_evidence_io.py`.

Required coverage:

- Valid source registry JSON loads into active and experimental source entries.
- Valid JSONL evidence loads into an `EvidenceCorpus`.
- Malformed source JSON raises `EvidenceLoadError` with path context.
- Malformed evidence JSONL raises `EvidenceLoadError` with path and line context.
- Evidence that references a missing source raises `EvidenceLoadError`.
- `load_corpus` output can create a deterministic `DataSnapshot`.

## Non-Goals

- No CSV, Parquet, database, or remote URL support.
- No schema migration layer.
- No real external evidence collection.
- No model fitting or ensemble weighting.
- No GUI or end-user workflow.

## Success Criteria

- The data I/O layer can load a small local corpus into the existing backend objects.
- Bad data fails with actionable file and line context.
- The implementation stays standard-library only.
- Full test suite passes.

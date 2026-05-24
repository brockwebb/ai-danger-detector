# ADD Evidence Corpus Design

Date: 2026-05-24

## Purpose

Build the next backend layer for ADD: an evidence corpus that manages collections of `EvidenceUnit` records against a `SourceRegistry`.

The corpus should make ADD's evidence base usable for future calibration and model comparison without deciding which model family wins. Bayesian, statistical, rubric, Markov, and ensemble models should all be able to consume the same filtered records, feature rows, summaries, splits, and data snapshot metadata.

## Scope

This slice implements corpus mechanics only. It does not ingest external data, tune model weights, perform Bayesian updating, train predictive models, or build a user interface.

The corpus should support:

- adding evidence only when the referenced source exists,
- rejecting duplicate evidence identifiers,
- excluding quarantined, deprecated, or removed sources from calibration by default,
- excluding quarantined evidence records from calibration by default,
- producing model-ready feature rows while preserving evidence and source identifiers,
- summarizing coverage by domain, task type, source, evidence type, and quality tier,
- creating deterministic train/calibration/holdout splits,
- creating a `DataSnapshot` object with version metadata for reproducible runs.

## Architecture

Add `core_model/evidence_corpus.py`.

The module should depend on:

- `EvidenceUnit` and `EvidenceQualityTier` from `core_model/evidence_schema.py`,
- `SourceRegistry` and `SourceStatus` from `core_model/source_registry.py`.

It should expose:

- `EvidenceCorpus`
- `EvidenceSplit`
- `DataSnapshot`

`EvidenceCorpus` owns evidence records and validates them against a source registry. `EvidenceSplit` stores deterministic partitioned evidence IDs. `DataSnapshot` records metadata about the corpus, schema version, source registry version, feature transformation version, and included evidence/source counts.

## Corpus Behavior

`EvidenceCorpus` should be initialized with a `SourceRegistry`.

Adding evidence should:

- reject a duplicate `evidence_id`,
- reject an unknown `source_id`,
- store records by stable ID,
- leave source lifecycle decisions to the registry.

Calibration eligibility should require both:

- the evidence record is calibration eligible,
- the source status is not quarantined, deprecated, or removed.

Experimental sources may remain available for exploratory runs, but the default calibration filter should include only active sources unless the caller explicitly includes experimental sources.

## Feature Rows

Feature rows should come from `EvidenceUnit.to_feature_row()` and add source metadata needed for later modeling:

- `source_status`
- `source_quality`
- `source_type`

This keeps model-ready data traceable back to both evidence records and source registry entries.

## Coverage Summary

Coverage summaries should be simple dictionaries. The first implementation should count:

- records by domain,
- records by task type,
- records by source ID,
- records by evidence type,
- records by evidence quality tier,
- records by source status.

This is not a statistical report. It is a fast backend sanity check for coverage and skew.

## Deterministic Splits

The corpus should create deterministic splits from evidence IDs using a seed. The same evidence IDs, ratios, and seed should produce the same split every time.

The first implementation should support:

- train ratio,
- calibration ratio,
- holdout ratio,
- seed,
- optional calibration-only filtering before splitting.

Ratios must be positive and sum to one. Empty corpora should be rejected.

## Data Snapshot

`DataSnapshot` should record:

- snapshot ID,
- schema version,
- source registry version,
- feature transformation version,
- created date,
- evidence count,
- source count,
- included evidence IDs,
- included source IDs.

The snapshot should be serializable to a dictionary for later logging or file persistence.

## Tests

Use TDD. Tests should cover:

- unknown source IDs are rejected,
- duplicate evidence IDs are rejected,
- quarantined sources are excluded by default,
- experimental sources are excluded from calibration by default but can be included explicitly,
- quarantined evidence records are excluded from calibration,
- feature rows include evidence and source traceability,
- coverage summaries count expected dimensions,
- deterministic splits are stable for the same seed,
- invalid split ratios are rejected,
- data snapshots record version and inclusion metadata.

## Documentation

Update `docs/evidence-data-architecture.md` and `README.md` after implementation to name the corpus layer as the current bridge between source/evidence schema and future calibration runs.

## Success Criteria

The slice is successful when:

- the corpus can safely collect evidence records against known sources,
- calibration-ready records can be filtered without mutating source or evidence data,
- feature rows and summaries are traceable and deterministic,
- data splits are reproducible,
- snapshots provide enough metadata to identify what data a future model run used,
- all tests pass on `main` after merge.

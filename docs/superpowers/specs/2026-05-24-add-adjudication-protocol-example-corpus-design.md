# ADD Adjudication Protocol and Example Corpus Design

Date: 2026-05-24

## Purpose

ADD now has an evidence schema, source registry, corpus container, and local data loader. The next slice defines how a case should become labeled evidence and adds a tiny illustrative corpus that exercises that machinery.

The protocol is a governance and data-quality artifact, not a validation claim. The example corpus is synthetic and exists to demonstrate file format, label variety, source traceability, and loader behavior. It must not be treated as calibration evidence.

## Scope

This slice adds:

- `docs/adjudication-protocol.md`: case intake, reviewer roles, label definitions, disagreement handling, quality gates, calibration eligibility, and reporting expectations.
- `data/examples/sources.json`: one synthetic/illustrative source registry entry.
- `data/examples/evidence.jsonl`: a small set of synthetic example evidence units.
- `data/examples/README.md`: file-format notes and explicit limitations.
- `tests/test_example_corpus.py`: tests proving the example corpus loads, remains excluded from calibration by default, has label variety, and can produce a snapshot.

The slice also updates README and evidence/validation docs to point to the protocol and example corpus.

## Example Corpus Rules

The example corpus should contain six fictional ADD cases across multiple domains and outcomes:

- creative benign use,
- health harm,
- legal near miss,
- finance loss,
- software corrected error,
- education benign or bounded use.

Every record uses the existing `EvidenceUnit` fields and loads through `core_model.evidence_io.load_corpus`. The source registry entry is marked:

- `source_type`: `synthetic`,
- `quality_tier`: `tier_4`,
- `status`: `experimental`,
- known biases include that the cases are fictional, non-representative, and not calibrated.

Because the source is experimental, `EvidenceCorpus.calibration_evidence()` should return no records by default. This makes the demonstration useful while preserving the rule that calibration requires reviewed, active sources.

## Adjudication Protocol Requirements

The protocol should define:

- case intake and inclusion/exclusion decisions,
- required source and evidence metadata,
- reviewer roles,
- label workflow,
- oversight-label interpretation,
- harm severity, detectability, reversibility, and verification burden anchors,
- workflow path coding using S0 through S8,
- disagreement handling,
- quality tiering and quarantine rules,
- calibration eligibility gates,
- privacy and governance limits,
- reporting expectations.

The protocol should also state that weak or disputed evidence may still be useful for hypothesis generation, but should widen uncertainty or remain excluded from calibration unless it clears review gates.

## Non-Goals

- No real incident collection.
- No external data scraping.
- No user-facing tool.
- No model fitting, Bayesian update, or ensemble weighting.
- No claim that the example corpus validates ADD.

## Success Criteria

- The protocol is specific enough that a future reviewer can label a case consistently.
- The example corpus loads through the existing data loader.
- The example source is excluded from calibration by default.
- Tests validate the example corpus shape and snapshot behavior.
- Public docs point readers to the protocol without overstating evidence strength.

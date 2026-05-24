from pathlib import Path

from core_model.evidence_io import load_corpus, load_source_registry
from core_model.evidence_schema import OutcomeLabel, OversightLabel
from core_model.source_registry import SourceStatus, SourceType


EXAMPLE_DIR = Path("data/examples")
SOURCE_PATH = EXAMPLE_DIR / "sources.json"
EVIDENCE_PATH = EXAMPLE_DIR / "evidence.jsonl"


def test_example_corpus_loads_and_has_expected_shape():
    corpus = load_corpus(SOURCE_PATH, EVIDENCE_PATH)

    assert len(corpus.evidence) == 6
    assert corpus.coverage_summary()["by_domain"] == {
        "creative": 1,
        "education": 1,
        "finance": 1,
        "health": 1,
        "law": 1,
        "software": 1,
    }


def test_example_source_is_illustrative_and_not_calibration_ready():
    registry = load_source_registry(SOURCE_PATH)
    source = registry.get_source("src-illustrative-add-cases")
    corpus = load_corpus(SOURCE_PATH, EVIDENCE_PATH)

    assert source.source_type is SourceType.SYNTHETIC
    assert source.status is SourceStatus.EXPERIMENTAL
    assert "fictional examples" in source.known_biases
    assert corpus.calibration_evidence() == ()


def test_example_corpus_contains_outcome_and_oversight_variety():
    corpus = load_corpus(SOURCE_PATH, EVIDENCE_PATH)
    outcomes = {unit.outcome_label for unit in corpus.evidence}
    oversight = {unit.oversight_label for unit in corpus.evidence}

    assert {
        OutcomeLabel.BENIGN_USE,
        OutcomeLabel.CORRECTED_ERROR,
        OutcomeLabel.HARM,
        OutcomeLabel.LOSS,
        OutcomeLabel.NEAR_MISS,
    }.issubset(outcomes)
    assert {
        OversightLabel.CASUAL_EXPLORATORY,
        OversightLabel.ASSISTED_BOUNDED,
        OversightLabel.TRAINED_REVIEW_REQUIRED,
        OversightLabel.EXPERT_REVIEW_REQUIRED,
        OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE,
    }.issubset(oversight)


def test_example_corpus_can_create_snapshot():
    corpus = load_corpus(SOURCE_PATH, EVIDENCE_PATH)

    snapshot = corpus.create_snapshot(
        snapshot_id="example-snapshot-001",
        schema_version="evidence-schema-v1",
        source_registry_version="example-sources-v1",
        feature_transformation_version="feature-row-v1",
        created_date="2026-05-24",
    )

    assert snapshot.evidence_count == 6
    assert snapshot.source_count == 1
    assert snapshot.included_source_ids == ("src-illustrative-add-cases",)

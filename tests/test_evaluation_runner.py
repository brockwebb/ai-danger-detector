from core_model.evaluation_runner import EvaluationReport, EvaluationRow, evaluate_corpus
from core_model.evidence_corpus import EvidenceCorpus
from core_model.evidence_io import load_corpus
from core_model.evidence_schema import (
    EvidenceQualityTier,
    EvidenceType,
    EvidenceUnit,
    OutcomeLabel,
    OversightLabel,
    UserExpertise,
)
from core_model.source_registry import SourceRegistry, SourceStatus, SourceType


def _registry():
    registry = SourceRegistry()
    registry.add_source(
        source_id="src-active",
        source_name="Active adjudicated cases",
        source_type=SourceType.CASE_SET,
        owner_or_publisher="ADD",
        license_or_access="private",
        update_cadence="one-time",
        coverage=("creative", "health", "law"),
        known_biases=(),
        quality_tier=EvidenceQualityTier.TIER_1,
    )
    registry.update_status("src-active", SourceStatus.ACTIVE, reason="approved")
    return registry


def _evidence(evidence_id, *, oversight_label, **overrides):
    values = {
        "evidence_id": evidence_id,
        "source_id": "src-active",
        "evidence_type": EvidenceType.CASE_REVIEW,
        "collection_date": "2026-05-25",
        "event_date": "2026-05-20",
        "domain": "creative",
        "task_type": "draft low-stakes text",
        "model_family": "unknown",
        "model_version": "unknown",
        "user_expertise": UserExpertise.TRAINED,
        "governance_context": "ordinary user review",
        "outcome_label": OutcomeLabel.BENIGN_USE,
        "oversight_label": oversight_label,
        "harm_severity": 0.05,
        "detectability": 0.9,
        "reversibility": 0.95,
        "verification_burden": 0.1,
        "workflow_path": ("S0", "S1", "S3", "S7"),
        "confidence": 0.7,
        "source_quality": EvidenceQualityTier.TIER_1,
        "bias_notes": (),
        "relevance_limits": (),
    }
    values.update(overrides)
    return EvidenceUnit(**values)


def _corpus(*units):
    corpus = EvidenceCorpus(_registry())
    for unit in units:
        corpus.add(unit)
    return corpus


def test_evaluate_corpus_preserves_traceable_rows():
    corpus = _corpus(
        _evidence("case-001", oversight_label=OversightLabel.CASUAL_EXPLORATORY)
    )

    report = evaluate_corpus(corpus)

    assert report.record_count == 1
    assert report.rows[0].evidence_id == "case-001"
    assert report.rows[0].source_id == "src-active"
    assert report.rows[0].domain == "creative"
    assert report.rows[0].adjudicated_band is OversightLabel.CASUAL_EXPLORATORY
    assert report.rows[0].predicted_band is OversightLabel.CASUAL_EXPLORATORY
    assert report.rows[0].error_direction == "match"
    assert isinstance(report.rows[0].drivers, tuple)


def test_evaluate_corpus_reports_agreement_and_band_error():
    corpus = _corpus(
        _evidence("case-001", oversight_label=OversightLabel.CASUAL_EXPLORATORY),
        _evidence(
            "case-002",
            oversight_label=OversightLabel.EXPERT_REVIEW_REQUIRED,
            domain="health",
            harm_severity=0.9,
            detectability=0.2,
            reversibility=0.2,
            verification_burden=0.9,
            user_expertise=UserExpertise.NON_EXPERT,
            governance_context="no expert review",
        ),
    )

    report = evaluate_corpus(corpus)

    assert report.record_count == 2
    assert report.evaluable_count == 2
    assert report.metrics["exact_band_agreement"] == 0.5
    assert report.metrics["mean_absolute_band_error"] == 0.5
    assert report.predicted_band_counts["casual_exploratory"] == 1
    assert report.adjudicated_band_counts["expert_review_required"] == 1


def test_evaluate_corpus_counts_under_and_over_escalation():
    corpus = _corpus(
        _evidence(
            "case-under",
            oversight_label=OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE,
            harm_severity=0.2,
            detectability=0.9,
            reversibility=0.9,
            verification_burden=0.2,
        ),
        _evidence(
            "case-over",
            oversight_label=OversightLabel.ASSISTED_BOUNDED,
            harm_severity=0.95,
            detectability=0.2,
            reversibility=0.2,
            verification_burden=0.95,
            user_expertise=UserExpertise.NON_EXPERT,
            governance_context="no review",
        ),
    )

    report = evaluate_corpus(corpus)

    assert report.metrics["under_escalation_rate"] == 0.5
    assert report.metrics["over_escalation_rate"] == 0.5
    assert report.metrics["false_reassurance_rate"] == 1.0
    assert report.metrics["false_escalation_rate"] == 1.0


def test_evaluate_corpus_retains_unknown_label_rows_but_excludes_from_metrics():
    corpus = _corpus(
        _evidence("case-known", oversight_label=OversightLabel.CASUAL_EXPLORATORY),
        _evidence("case-unknown", oversight_label=OversightLabel.UNKNOWN),
    )

    report = evaluate_corpus(corpus)

    assert report.record_count == 2
    assert report.evaluable_count == 1
    assert report.rows[1].is_evaluable is False
    assert report.rows[1].error_direction == "not_evaluable"
    assert report.metrics["exact_band_agreement"] == 1.0


def test_example_corpus_can_be_evaluated_end_to_end():
    corpus = load_corpus("data/examples/sources.json", "data/examples/evidence.jsonl")

    report = evaluate_corpus(corpus)

    assert isinstance(report, EvaluationReport)
    assert all(isinstance(row, EvaluationRow) for row in report.rows)
    assert report.record_count == 6
    assert report.evaluable_count == 6
    assert report.coverage_summary["by_source_id"] == {"src-illustrative-add-cases": 6}
    assert report.predicted_band_counts["expert_led_or_no_autonomous_use"] >= 2


def test_evaluation_runner_exports_public_api():
    import core_model

    assert "EvaluationReport" in core_model.__all__
    assert "EvaluationRow" in core_model.__all__
    assert "evaluate_corpus" in core_model.__all__
    assert core_model.EvaluationReport is EvaluationReport
    assert core_model.EvaluationRow is EvaluationRow
    assert core_model.evaluate_corpus is evaluate_corpus

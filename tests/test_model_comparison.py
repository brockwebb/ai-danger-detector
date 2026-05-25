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
from core_model.evaluation_runner import EvaluationReport
from core_model.model_comparison import (
    ScorerDefinition,
    ScorerOutputType,
    baseline_rubric_scorer,
    compare_models,
)
from core_model.rubric_scorer import RubricScore
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


def _score_as(label):
    def scorer(unit):
        return RubricScore(
            score=0.0,
            band=label,
            factor_scores={},
            drivers=("test scorer",),
            assumptions=("test-only scorer",),
        )

    return scorer


def _score_as_adjudicated(unit):
    band = (
        OversightLabel.CASUAL_EXPLORATORY
        if unit.oversight_label is OversightLabel.UNKNOWN
        else unit.oversight_label
    )
    return RubricScore(
        score=1.0,
        band=band,
        factor_scores={},
        drivers=("test oracle",),
        assumptions=("test-only scorer",),
    )


def test_baseline_rubric_scorer_declares_metric_compatibility():
    scorer = baseline_rubric_scorer()

    assert scorer.name == "baseline_rubric"
    assert scorer.output_type is ScorerOutputType.ORDINAL_BAND
    assert scorer.scorer is not None
    assert "brier_score" in scorer.deferred_metric_names
    assert "log_loss" in scorer.deferred_metric_names
    assert "expected_calibration_error" in scorer.deferred_metric_names
    assert any(
        "not a calibrated probability" in note for note in scorer.native_metric_notes
    )
    assert any("directly" in note for note in scorer.equivalence_notes)


def test_compare_models_evaluates_example_corpus_with_baseline_scorer():
    corpus = load_corpus("data/examples/sources.json", "data/examples/evidence.jsonl")

    report = compare_models(corpus, (baseline_rubric_scorer(),))

    assert report.scorer_count == 1
    assert report.record_count == 6
    assert report.coverage_summary["by_source_id"] == {"src-illustrative-add-cases": 6}
    assert report.rows[0].scorer_name == "baseline_rubric"
    assert isinstance(report.rows[0].evaluation_report, EvaluationReport)
    assert report.rows[0].common_metrics == report.rows[0].evaluation_report.metrics
    assert report.rows[0].common_metrics["exact_band_agreement"] == 0.5


def test_compare_models_selects_best_by_explicit_metric_only():
    corpus = _corpus(
        _evidence("case-001", oversight_label=OversightLabel.CASUAL_EXPLORATORY),
        _evidence("case-002", oversight_label=OversightLabel.EXPERT_REVIEW_REQUIRED),
    )
    always_low = ScorerDefinition(
        name="always_low",
        description="Always predicts the lowest oversight band.",
        output_type=ScorerOutputType.ORDINAL_BAND,
        scorer=_score_as(OversightLabel.CASUAL_EXPLORATORY),
        native_metric_notes=("Ordinal band-error metrics apply.",),
        equivalence_notes=("Returns a band directly for comparison.",),
    )
    oracle = ScorerDefinition(
        name="oracle",
        description="Returns the adjudicated label for test comparison.",
        output_type=ScorerOutputType.ORDINAL_BAND,
        scorer=_score_as_adjudicated,
        native_metric_notes=("Ordinal band-error metrics apply.",),
        equivalence_notes=("Returns a band directly for comparison.",),
    )

    report = compare_models(corpus, (always_low, oracle))

    assert report.scorer_count == 2
    assert report.best_by_metric("exact_band_agreement").scorer_name == "oracle"
    assert (
        report.best_by_metric("mean_absolute_band_error", lower_is_better=True)
        .scorer_name
        == "oracle"
    )


def test_probability_metrics_are_deferred_for_baseline_rubric():
    corpus = load_corpus("data/examples/sources.json", "data/examples/evidence.jsonl")

    report = compare_models(corpus, (baseline_rubric_scorer(),))
    row = report.rows[0]

    assert row.native_metrics == {}
    assert "brier_score" not in row.common_metrics
    assert "log_loss" not in row.common_metrics
    assert "expected_calibration_error" not in row.common_metrics
    assert row.deferred_metric_names == (
        "brier_score",
        "log_loss",
        "expected_calibration_error",
    )


def test_model_comparison_exports_public_api():
    import core_model

    assert "ScorerDefinition" in core_model.__all__
    assert "ScorerOutputType" in core_model.__all__
    assert "ModelComparisonReport" in core_model.__all__
    assert "ModelComparisonRow" in core_model.__all__
    assert "baseline_rubric_scorer" in core_model.__all__
    assert "compare_models" in core_model.__all__
    assert core_model.baseline_rubric_scorer is baseline_rubric_scorer
    assert core_model.compare_models is compare_models

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
from core_model.model_comparison import (
    ScorerOutputType,
    baseline_rubric_scorer,
    compare_models,
)
from core_model.reference_scorers import (
    always_assisted_scorer,
    always_casual_scorer,
    always_expert_led_scorer,
    always_expert_review_scorer,
    always_trained_review_scorer,
    reference_policy_scorers,
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


def test_fixed_policy_scorers_return_expected_bands():
    unit = _evidence("case-001", oversight_label=OversightLabel.CASUAL_EXPLORATORY)

    assert always_casual_scorer().scorer(unit).band is OversightLabel.CASUAL_EXPLORATORY
    assert always_assisted_scorer().scorer(unit).band is OversightLabel.ASSISTED_BOUNDED
    assert (
        always_trained_review_scorer().scorer(unit).band
        is OversightLabel.TRAINED_REVIEW_REQUIRED
    )
    assert (
        always_expert_review_scorer().scorer(unit).band
        is OversightLabel.EXPERT_REVIEW_REQUIRED
    )
    assert (
        always_expert_led_scorer().scorer(unit).band
        is OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE
    )


def test_reference_policy_scorers_have_stable_order_and_metadata():
    scorers = reference_policy_scorers()

    assert [scorer.name for scorer in scorers] == [
        "always_casual",
        "always_assisted",
        "always_trained_review",
        "always_expert_review",
        "always_expert_led",
    ]
    assert all(scorer.output_type is ScorerOutputType.ORDINAL_BAND for scorer in scorers)
    assert all("policy baseline" in scorer.description for scorer in scorers)
    assert all("brier_score" in scorer.deferred_metric_names for scorer in scorers)
    assert all(any("TEVV" in note for note in scorer.native_metric_notes) for scorer in scorers)
    assert all(any("not learned" in note for note in scorer.equivalence_notes) for scorer in scorers)


def test_reference_policies_run_through_model_comparison():
    corpus = load_corpus("data/examples/sources.json", "data/examples/evidence.jsonl")

    report = compare_models(
        corpus,
        (baseline_rubric_scorer(), *reference_policy_scorers()),
    )

    assert report.scorer_count == 6
    assert [row.scorer_name for row in report.rows][0] == "baseline_rubric"
    assert [row.scorer_name for row in report.rows][1:] == [
        "always_casual",
        "always_assisted",
        "always_trained_review",
        "always_expert_review",
        "always_expert_led",
    ]
    assert all(row.common_metrics for row in report.rows)


def test_reference_policies_expose_expected_tradeoffs():
    corpus = _corpus(
        _evidence("case-low", oversight_label=OversightLabel.CASUAL_EXPLORATORY),
        _evidence(
            "case-high",
            oversight_label=OversightLabel.EXPERT_LED_OR_NO_AUTONOMOUS_USE,
        ),
    )

    report = compare_models(
        corpus,
        (always_casual_scorer(), always_expert_led_scorer()),
    )
    casual_row = report.rows[0]
    expert_led_row = report.rows[1]

    assert casual_row.common_metrics["under_escalation_rate"] == 0.5
    assert casual_row.common_metrics["false_reassurance_rate"] == 1.0
    assert casual_row.common_metrics["over_escalation_rate"] == 0.0
    assert expert_led_row.common_metrics["over_escalation_rate"] == 0.5
    assert expert_led_row.common_metrics["false_escalation_rate"] == 1.0
    assert expert_led_row.common_metrics["under_escalation_rate"] == 0.0


def test_reference_scorers_export_public_api():
    import core_model

    assert "always_casual_scorer" in core_model.__all__
    assert "always_assisted_scorer" in core_model.__all__
    assert "always_trained_review_scorer" in core_model.__all__
    assert "always_expert_review_scorer" in core_model.__all__
    assert "always_expert_led_scorer" in core_model.__all__
    assert "reference_policy_scorers" in core_model.__all__
    assert core_model.always_casual_scorer is always_casual_scorer
    assert core_model.reference_policy_scorers is reference_policy_scorers

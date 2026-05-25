import pytest

from core_model.calibration_observations import (
    CalibrationObservationConfig,
    CalibrationObservationSet,
    CalibrationParameter,
    ExcludedEvidence,
    build_calibration_observations,
)
from core_model.evidence_corpus import EvidenceCorpus
from core_model.evidence_schema import (
    EvidenceQualityTier,
    EvidenceType,
    EvidenceUnit,
    OutcomeLabel,
    OversightLabel,
    UserExpertise,
)
from core_model.source_registry import SourceRegistry, SourceStatus, SourceType


def _registry() -> SourceRegistry:
    registry = SourceRegistry()
    registry.add_source(
        source_id="src-active",
        source_name="Active adjudicated set",
        source_type=SourceType.CASE_SET,
        owner_or_publisher="ADD",
        license_or_access="private",
        update_cadence="one-time",
        coverage=("health",),
        known_biases=(),
        quality_tier=EvidenceQualityTier.TIER_1,
    )
    registry.update_status("src-active", SourceStatus.ACTIVE, reason="approved")
    registry.add_source(
        source_id="src-experimental",
        source_name="Experimental incident feed",
        source_type=SourceType.INCIDENT_REPOSITORY,
        owner_or_publisher="ADD",
        license_or_access="public",
        update_cadence="periodic",
        coverage=("software",),
        known_biases=("severity bias",),
        quality_tier=EvidenceQualityTier.TIER_3,
    )
    return registry


def _evidence(
    evidence_id: str = "case-001",
    source_id: str = "src-active",
    **overrides,
) -> EvidenceUnit:
    values = {
        "evidence_id": evidence_id,
        "source_id": source_id,
        "evidence_type": EvidenceType.CASE_REVIEW,
        "collection_date": "2026-05-24",
        "event_date": "2026-05-20",
        "domain": "health",
        "task_type": "medical symptom advice",
        "model_family": "unknown",
        "model_version": "unknown",
        "user_expertise": UserExpertise.NON_EXPERT,
        "governance_context": "informal",
        "outcome_label": OutcomeLabel.HARM,
        "oversight_label": OversightLabel.EXPERT_REVIEW_REQUIRED,
        "harm_severity": 0.8,
        "detectability": 0.2,
        "reversibility": 0.3,
        "verification_burden": 0.9,
        "workflow_path": ("S0", "S1", "S2", "S6", "S8"),
        "confidence": 0.8,
        "source_quality": EvidenceQualityTier.TIER_1,
        "bias_notes": (),
        "relevance_limits": (),
    }
    values.update(overrides)
    return EvidenceUnit(**values)


def _corpus(*units: EvidenceUnit) -> EvidenceCorpus:
    corpus = EvidenceCorpus(_registry())
    for unit in units:
        corpus.add(unit)
    return corpus


def _observations_by_id(
    observation_set: CalibrationObservationSet,
):
    return {
        observation.evidence_ids[0]: observation
        for observation in observation_set.observations
    }


def test_error_parameter_maps_outcomes_and_preserves_traceability():
    corpus = _corpus(
        _evidence("case-harm", outcome_label=OutcomeLabel.HARM, confidence=0.8),
        _evidence(
            "case-benign",
            outcome_label=OutcomeLabel.BENIGN_USE,
            confidence=0.5,
            source_quality=EvidenceQualityTier.TIER_2,
        ),
    )

    observation_set = build_calibration_observations(
        corpus, CalibrationParameter.P_ERROR_PER_TASK
    )

    assert observation_set.parameter == CalibrationParameter.P_ERROR_PER_TASK
    assert observation_set.config_version == "calibration-observation-v1"
    assert observation_set.exclusions == ()
    by_id = _observations_by_id(observation_set)
    assert by_id["case-harm"].successes == 1.0
    assert by_id["case-harm"].failures == 0.0
    assert by_id["case-harm"].weight == pytest.approx(0.8)
    assert by_id["case-harm"].source_id == "src-active"
    assert by_id["case-harm"].evidence_ids == ("case-harm",)
    assert "parameter=p_error_per_task" in by_id["case-harm"].notes
    assert "outcome_label=harm" in by_id["case-harm"].notes
    assert by_id["case-benign"].successes == 0.0
    assert by_id["case-benign"].failures == 1.0
    assert by_id["case-benign"].weight == pytest.approx(0.375)


def test_error_parameter_excludes_unknown_and_unresolved_outcomes():
    corpus = _corpus(
        _evidence("case-unknown", outcome_label=OutcomeLabel.UNKNOWN),
        _evidence("case-unresolved", outcome_label=OutcomeLabel.UNRESOLVED),
    )

    observation_set = build_calibration_observations(
        corpus, CalibrationParameter.P_ERROR_PER_TASK
    )

    assert observation_set.observations == ()
    assert {
        exclusion.evidence_id: exclusion.reason_code
        for exclusion in observation_set.exclusions
    } == {
        "case-unknown": "unsupported_outcome_label",
        "case-unresolved": "unsupported_outcome_label",
    }
    assert all(
        exclusion.parameter == CalibrationParameter.P_ERROR_PER_TASK
        for exclusion in observation_set.exclusions
    )


def test_detectability_and_reversibility_use_labeled_pseudo_observations():
    corpus = _corpus(
        _evidence(
            "case-scalar",
            detectability=0.25,
            reversibility=0.8,
            confidence=0.6,
            source_quality=EvidenceQualityTier.TIER_3,
        )
    )
    config = CalibrationObservationConfig(pseudo_observation_strength=2.0)

    detectability = build_calibration_observations(
        corpus, CalibrationParameter.DETECTABILITY, config=config
    ).observations[0]
    reversibility = build_calibration_observations(
        corpus, CalibrationParameter.REVERSIBILITY, config=config
    ).observations[0]

    assert detectability.successes == pytest.approx(0.5)
    assert detectability.failures == pytest.approx(1.5)
    assert detectability.weight == pytest.approx(0.3)
    assert "pseudo-observation" in detectability.notes
    assert "scalar_value=0.250" in detectability.notes
    assert reversibility.successes == pytest.approx(1.6)
    assert reversibility.failures == pytest.approx(0.4)
    assert reversibility.weight == pytest.approx(0.3)
    assert "parameter=reversibility" in reversibility.notes


def test_confidence_and_quality_control_observation_weighting_and_exclusion():
    corpus = _corpus(
        _evidence("case-low-confidence", confidence=0.2),
        _evidence(
            "case-tier4",
            confidence=0.8,
            source_quality=EvidenceQualityTier.TIER_4,
        ),
    )
    config = CalibrationObservationConfig(minimum_confidence=0.25)

    observation_set = build_calibration_observations(
        corpus, CalibrationParameter.P_ERROR_PER_TASK, config=config
    )

    assert len(observation_set.observations) == 1
    assert observation_set.observations[0].evidence_ids == ("case-tier4",)
    assert observation_set.observations[0].weight == pytest.approx(0.2)
    assert observation_set.exclusions == (
        ExcludedEvidence(
            evidence_id="case-low-confidence",
            source_id="src-active",
            parameter=CalibrationParameter.P_ERROR_PER_TASK,
            reason_code="below_minimum_confidence",
            notes=("confidence=0.200", "minimum_confidence=0.250"),
        ),
    )


def test_experimental_source_inclusion_is_explicit():
    corpus = _corpus(
        _evidence(
            "case-experimental",
            source_id="src-experimental",
            domain="software",
            source_quality=EvidenceQualityTier.TIER_3,
        )
    )

    default_set = build_calibration_observations(
        corpus, CalibrationParameter.P_ERROR_PER_TASK
    )
    included_set = build_calibration_observations(
        corpus,
        CalibrationParameter.P_ERROR_PER_TASK,
        config=CalibrationObservationConfig(include_experimental_sources=True),
    )

    assert default_set.observations == ()
    assert default_set.exclusions[0].reason_code == "source_status_experimental"
    assert len(included_set.observations) == 1
    assert included_set.observations[0].evidence_ids == ("case-experimental",)


def test_empty_observation_set_summary_captures_exclusions():
    corpus = _corpus(_evidence("case-unknown", outcome_label=OutcomeLabel.UNKNOWN))

    observation_set = build_calibration_observations(
        corpus, CalibrationParameter.P_ERROR_PER_TASK
    )

    assert observation_set.summary() == {
        "parameter": "p_error_per_task",
        "config_version": "calibration-observation-v1",
        "observation_count": 0,
        "exclusion_count": 1,
        "source_ids": (),
        "evidence_ids": (),
        "excluded_evidence_ids": ("case-unknown",),
        "exclusion_reasons": {"unsupported_outcome_label": 1},
    }


def test_config_validation_and_unsupported_parameters():
    with pytest.raises(ValueError, match="pseudo_observation_strength must be positive"):
        CalibrationObservationConfig(pseudo_observation_strength=0)

    with pytest.raises(ValueError, match="minimum_confidence must be between 0 and 1"):
        CalibrationObservationConfig(minimum_confidence=1.1)

    with pytest.raises(ValueError, match="quality weight must be non-negative"):
        CalibrationObservationConfig(
            quality_weights={EvidenceQualityTier.TIER_1: -0.1}
        )

    with pytest.raises(ValueError, match="config_version must be non-empty"):
        CalibrationObservationConfig(config_version=" ")

    with pytest.raises(ValueError, match="unsupported calibration parameter"):
        build_calibration_observations(_corpus(), "severity")


def test_calibration_observations_export_public_api():
    import core_model

    assert "CalibrationObservationConfig" in core_model.__all__
    assert "CalibrationObservationSet" in core_model.__all__
    assert "CalibrationParameter" in core_model.__all__
    assert "ExcludedEvidence" in core_model.__all__
    assert "build_calibration_observations" in core_model.__all__
    assert core_model.CalibrationParameter is CalibrationParameter
    assert core_model.build_calibration_observations is build_calibration_observations

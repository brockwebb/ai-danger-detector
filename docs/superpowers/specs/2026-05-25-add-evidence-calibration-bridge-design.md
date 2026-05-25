# Evidence-to-Bayesian Calibration Bridge Design

Date: 2026-05-25

## Purpose

ADD now has an evidence corpus and beta-binomial calibration primitives, but there is no bridge between them. This slice adds a small backend layer that converts eligible evidence records into traceable `BetaObservation` objects for bounded probability parameters.

The bridge is calibration machinery, not a calibrated ADD model. It should make evidence-to-parameter assumptions explicit so future reviewers can tune, challenge, or replace them.

## Scope

This slice should add:

- A module for building beta-binomial observations from an `EvidenceCorpus`.
- Parameter-specific mappings for bounded probabilities.
- Inclusion and exclusion reporting.
- Transparent weighting from source quality and record confidence.
- Tests that prove traceability, weighting, and exclusion behavior.
- Documentation explaining how the bridge should and should not be used.

This slice should not add:

- A Bayesian scorer.
- MCMC or posterior sampling.
- Gamma-Poisson, negative-binomial, or other rate models.
- Automatic prior selection.
- Claims that ADD is calibrated.

## Supported Parameters

The first bridge should support three bounded probability parameters:

| Parameter | Evidence mapping |
| --- | --- |
| `p_error_per_task` | Outcome labels map to event/non-event observations. Harm, loss, near miss, and corrected error count as meaningful error or misuse events. Benign use counts as a non-event. Unresolved or unknown outcomes are excluded. |
| `detectability` | The evidence record's `detectability` field becomes a pseudo-observation expressing how detectable the issue was under the record context. |
| `reversibility` | The evidence record's `reversibility` field becomes a pseudo-observation expressing how reversible or recoverable the issue was under the record context. |

Pseudo-observations are allowed only if they are clearly labeled in notes. They are not empirical event counts. They convert bounded expert/adjudicated scalar fields into a weak beta-binomial-compatible form for early calibration experiments.

## Data Flow

1. A caller provides an `EvidenceCorpus`.
2. The bridge selects `corpus.calibration_evidence()` by default.
3. The caller may opt into experimental sources with an explicit configuration flag.
4. Each evidence unit is mapped to a `BetaObservation` for the requested parameter.
5. Excluded records are captured with stable reason codes.
6. The result returns both observations and exclusion metadata.
7. A caller may pass observations into `update_beta_binomial()` with an explicit prior.

The bridge should not update priors by itself. Keeping observation extraction separate from posterior updates makes it easier to inspect assumptions before running calibration.

## API Shape

Create `core_model/calibration_observations.py`.

Public objects:

- `CalibrationParameter`: enum with `P_ERROR_PER_TASK`, `DETECTABILITY`, and `REVERSIBILITY`.
- `CalibrationObservationConfig`: frozen dataclass controlling inclusion and weighting.
- `ExcludedEvidence`: frozen dataclass for records not converted into observations.
- `CalibrationObservationSet`: frozen dataclass containing the requested parameter, observation tuple, exclusions tuple, config version, and summary helpers.
- `build_calibration_observations(corpus, parameter, config=None)`: returns a `CalibrationObservationSet`.

Recommended configuration fields:

- `include_experimental_sources: bool = False`
- `config_version: str = "calibration-observation-v1"`
- `pseudo_observation_strength: float = 1.0`
- `quality_weights: dict[EvidenceQualityTier, float]`
- `minimum_confidence: float = 0.0`

Default quality weights should be conservative and explicit:

- Tier 1: `1.0`
- Tier 2: `0.75`
- Tier 3: `0.5`
- Tier 4: `0.25`
- Quarantined: `0.0`

Quarantined records should remain excluded by `EvidenceCorpus.calibration_evidence()` unless the existing corpus API changes in a later slice.

## Observation Mapping

For `p_error_per_task`:

- `OutcomeLabel.HARM`, `LOSS`, `NEAR_MISS`, and `CORRECTED_ERROR` map to `successes=1`, `failures=0`.
- `OutcomeLabel.BENIGN_USE` maps to `successes=0`, `failures=1`.
- `OutcomeLabel.UNRESOLVED` and `UNKNOWN` are excluded with reason code `unsupported_outcome_label`.

Here, "success" means a meaningful error, misuse, near miss, or corrected error event occurred. The name comes from the beta-binomial primitive, not from a value judgment.

For `detectability`:

- `successes = unit.detectability * pseudo_observation_strength`
- `failures = (1 - unit.detectability) * pseudo_observation_strength`

For `reversibility`:

- `successes = unit.reversibility * pseudo_observation_strength`
- `failures = (1 - unit.reversibility) * pseudo_observation_strength`

All observations should carry:

- `source_id`
- `evidence_ids=(unit.evidence_id,)`
- notes describing the mapping, quality tier, confidence, and config version

## Weighting

Observation weight should combine source quality and evidence confidence:

```text
weight = quality_weight[source_quality] * confidence
```

Records below `minimum_confidence` should be excluded with reason code `below_minimum_confidence`.

Tier 4 evidence should produce very weak observations by default, not full-strength calibration input. This keeps synthetic or anecdotal records visible for experiments while preserving the warning that they are not strong calibration evidence.

## Error Handling

The bridge should raise `ValueError` when:

- An unsupported parameter is requested.
- `pseudo_observation_strength` is not positive.
- `minimum_confidence` is outside `[0, 1]`.
- A configured quality weight is negative.
- `config_version` is empty.

The bridge should not raise just because all records are excluded. It should return an empty observation set with exclusions, because that is a useful diagnostic state for early data work.

## Documentation Updates

Update:

- `README.md` project tree and implementation summary.
- `docs/evidence-data-architecture.md` to describe the evidence-to-observation bridge.
- `docs/numerical-framework.md` to explain pseudo-observations and the separation between observation extraction and posterior updates.
- `docs/validation-agenda.md` to note that bridge mappings are assumptions requiring sensitivity checks.

Docs must keep the limitation clear: this bridge prepares observations for calibration experiments but does not validate ADD or estimate real-world risk by itself.

## Testing

Add `tests/test_calibration_observations.py` covering:

- Outcome-label mapping for `p_error_per_task`.
- Exclusion of unknown and unresolved outcomes.
- Detectability pseudo-observation mapping.
- Reversibility pseudo-observation mapping.
- Confidence and quality weighting.
- Experimental-source inclusion behavior.
- Empty observation set with exclusion reasons.
- Public API exports from `core_model`.

Existing tests should continue to pass.

## Acceptance Criteria

- A caller can build traceable beta observations from an `EvidenceCorpus`.
- Each observation preserves source and evidence IDs.
- Weighting is explicit and versioned.
- Pseudo-observations are clearly marked as pseudo-observations.
- Unknown or unresolved records are excluded with reason codes.
- The bridge remains independent of scorer and model-comparison code.
- The full test suite passes.

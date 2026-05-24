# ADD Evidence Data Architecture

Date: 2026-05-24

## Purpose

ADD needs an evidence layer before it can claim meaningful calibration. This document defines the backend data architecture for collecting, scoring, versioning, and testing evidence used to tune ADD risk models.

The goal is not to lock the project into one dataset, one scoring formula, or one model family. The goal is to make evidence streams swappable, auditable, and testable so weak sources can be downweighted or removed and useful sources can be added without rewriting the whole framework.

## Core Prediction Targets

ADD should separate model outputs into related but distinct targets:

| Target | Meaning | Example output |
| --- | --- | --- |
| Oversight requirement | Minimum human review posture needed before AI-assisted output influences action | Oversight band, threshold probability, escalation recommendation |
| Error or misuse likelihood | Probability or rate of meaningful AI error, misuse, hallucination, omission, or context failure | Probability, rate, or uncertainty interval |
| Harm or loss severity | Conditional impact if a meaningful error or misuse event matters | Ordinal severity, expected loss, conditional cost distribution |
| Detectability | Probability that the relevant user or reviewer detects the issue before action | Probability or ordinal score by expertise level |
| Reversibility | Probability or feasibility of correcting the issue after action | Probability, ordinal score, or recovery-cost distribution |
| Verification burden | Effort, expertise, time, or cost required to check the AI-assisted output | Ordinal score or cost estimate |

The current numerical framework primarily estimates oversight threshold probabilities and workflow outcomes. Future model families may estimate the other targets directly and feed an ensemble.

## Evidence Unit

The basic data object should be an evidence unit. An evidence unit may represent an incident, near miss, benign comparison case, expert-labeled scenario, benchmark result, deployment observation, user study result, or adjudicated review.

Each evidence unit should preserve source context and uncertainty. Weak or incomplete evidence is still useful if it is labeled honestly.

Required fields:

| Field | Meaning |
| --- | --- |
| evidence_id | Stable identifier for the record |
| source_id | Link to the source registry entry |
| evidence_type | Incident, near miss, benchmark, expert elicitation, deployment log, user study, case review, or comparison case |
| collection_date | Date the evidence was collected, published, observed, or reviewed |
| event_date | Date or period when the underlying event occurred, if known |
| domain | Health, law, finance, education, software, public benefits, employment, creative work, or other domain |
| task_type | Specific AI-assisted task being evaluated |
| model_family | AI model, model family, tool, or system class when known |
| model_version | Version, release, or observation window when known |
| user_expertise | Non-expert, trained, domain-familiar, expert, or unknown |
| governance_context | Controls present around the AI-assisted workflow |
| outcome_label | Harm, loss, near miss, corrected error, benign use, unresolved, or unknown |
| oversight_label | Adjudicated oversight level that would have been appropriate |
| harm_severity | Conditional harm or loss magnitude if applicable |
| detectability | How likely the relevant user could detect the issue before action |
| reversibility | How feasible it was to undo, correct, contain, or recover from the issue |
| verification_burden | Effort or expertise needed to verify the output |
| workflow_path | Observed or inferred path through generation, checking, escalation, action, correction, or harm |
| confidence | Confidence in the record and labels |
| source_quality | Quality score or tier for the source |
| bias_notes | Known reporting, selection, incentive, or interpretation concerns |
| relevance_limits | Domains, tasks, users, or model versions where the evidence should not be transferred without caution |

Optional fields:

- financial_loss_estimate
- time_loss_estimate
- affected_population
- regulatory_or_rights_impact
- prompt_or_input_context
- retrieval_or_tool_context
- reviewer_ids
- inter_rater_agreement
- adjudication_notes
- citation_or_url
- raw_artifact_reference

## Source Registry

Every evidence unit should point to a source registry entry. The source registry exists so ADD can add, remove, test, and compare evidence streams without hard-coding trust in any one source.

Each source should include:

| Field | Meaning |
| --- | --- |
| source_id | Stable source identifier |
| source_name | Human-readable source name |
| source_type | Incident repository, benchmark suite, expert panel, deployment log, user study, audit, literature review, or internal case set |
| owner_or_publisher | Organization, research group, reviewer, or maintainer |
| license_or_access | Public, restricted, private, synthetic, or unknown |
| update_cadence | One-time, periodic, continuous, or unknown |
| coverage | Domains, model families, languages, user groups, dates, and task types represented |
| known_biases | Reporting bias, selection bias, severity bias, survivorship bias, vendor bias, advocacy bias, or other limitations |
| quality_tier | Initial source-quality tier |
| active_status | Active, experimental, quarantined, deprecated, or removed |
| removal_reason | Required when a source is deprecated or removed |

Source status should be revisable. A source can begin as experimental, become active after review, or be quarantined if it produces unstable or misleading model behavior.

## Data Quality Tiers

Evidence quality should affect uncertainty and model weight. It should not be reduced to a binary trusted/untrusted flag.

| Tier | Description | Modeling treatment |
| --- | --- | --- |
| Tier 1 | Adjudicated cases with clear inclusion criteria, independent review, and enough context to score factors | Strong calibration input |
| Tier 2 | Structured benchmark, user study, audit, or deployment evidence with known limits | Useful calibration input with documented limits |
| Tier 3 | Credible incident reports, near misses, or expert-labeled cases with incomplete context | Informative but uncertainty-widening |
| Tier 4 | Anecdotes, media summaries, synthetic examples, or weakly documented claims | Hypothesis generation only unless corroborated |
| Quarantined | Evidence stream found to be misleading, unstable, duplicated, or too biased for current use | Excluded from calibration runs by default |

Quality tiers should be source-aware and record-aware. A high-quality source can still contain a weak individual record, and a low-quality source can occasionally contain useful leads.

## Evidence Streams

ADD should support multiple evidence streams:

- Incident and near-miss records.
- Benign comparison cases where AI use did not produce meaningful harm.
- Benchmark and evaluation results for task-specific error behavior.
- Structured expert elicitation.
- Domain expert case review.
- User studies on error detection and automation bias.
- Deployment observations where privacy, consent, and governance allow use.
- Synthetic stress cases used for sensitivity testing, clearly labeled as synthetic.

Benign and comparison cases matter. A dataset made only of dramatic incidents will inflate severity, distort base rates, and make calibration too pessimistic. A useful evidence architecture needs both failures and non-failures.

## Feature Schema

Evidence should be transformed into model-ready features without losing traceability back to source records.

Core feature groups:

| Feature group | Examples |
| --- | --- |
| Domain and task | Domain, task class, regulated status, public-facing status |
| AI system context | Model family, model version, retrieval, tools, autonomy level, interface constraints |
| User context | Expertise level, training, incentives, ability to verify, authority to act |
| Harm context | Harm type, severity, affected parties, reversibility, time sensitivity |
| Verification context | Verification burden, availability of ground truth, independent checks, expert access |
| Governance context | Human review, approval workflow, audit trail, monitoring, prohibition, escalation path |
| Workflow context | Generated output, acceptance, checking, expert review, correction, action, outcome |
| Evidence quality | Source tier, confidence, missingness, bias flags, relevance limits |

Features should be versioned. If a factor definition changes, older feature rows should remain interpretable under their original schema version.

## Model Families

The evidence architecture should support multiple candidate model families from the start:

- Bayesian models for priors, posterior updates, and uncertainty-aware calibration.
- Frequentist or statistical models for discrimination, calibration, and validation baselines.
- Rule-based or rubric models for interpretability and expert review.
- Markov workflow models for reliance pathways and escalation points.
- Ensemble models that combine candidate models into a more robust score.

No model family should be treated as permanently correct. Models should earn weight through validation performance, stability, interpretability, and relevance to the target use.

## Ensemble and Source Flexibility

ADD should be able to combine multiple models and multiple sources while keeping the contribution of each visible.

An initial ensemble can use a weighted linear combination:

```text
ensemble_score =
    w1 * rubric_score
  + w2 * bayesian_score
  + w3 * markov_workflow_score
  + w4 * expected_loss_score
  + w5 * uncertainty_penalty
```

Weights should be explicit, versioned, and initially provisional. As validation data improves, weights can be tuned using documented performance. If a source or model fails validation, it should be downweighted, quarantined, or removed.

The ensemble should report:

- component scores,
- component weights,
- aggregate score,
- uncertainty interval,
- key drivers,
- source coverage,
- evidence quality summary,
- model version and data snapshot version.

## Uncertainty Treatment

ADD should use uncertainty language carefully:

- Use uncertainty interval or simulation interval for Monte Carlo propagation over assumed distributions.
- Use credible interval only when the interval comes from Bayesian posterior or posterior predictive samples.
- Use confidence interval only when the method supports frequentist coverage claims.
- Avoid margin of error unless the evidence comes from a sampling design that justifies that term.

Sparse, biased, or weak evidence should widen uncertainty. It should not produce a narrow score just because the system can compute one.

## Performance Metrics

Model performance should be measured separately for each target. A model that predicts escalation well may not predict realized harm well, and a model that ranks high-risk cases well may still be poorly calibrated.

Core metrics:

| Metric | What it tests |
| --- | --- |
| Calibration | Whether predicted probabilities match observed or adjudicated frequencies |
| Brier score | Accuracy of probabilistic predictions |
| Log loss | Penalty for confident wrong probabilistic predictions |
| ROC-AUC | Ranking performance when positive and negative labels are available |
| PR-AUC | Ranking performance when high-risk or harmful cases are rare |
| False reassurance rate | Moderate-or-higher concern cases classified too low |
| False escalation rate | Low-concern cases classified too high |
| Uncertainty coverage | Whether observed/adjudicated outcomes fall within predicted intervals at expected rates |
| Inter-rater agreement | Whether human reviewers can apply labels and factors consistently |
| Sensitivity stability | Whether outputs remain stable under plausible input noise |
| Domain transfer performance | Whether assumptions from one domain degrade in another |
| Temporal stability | Whether model behavior holds across model releases and workflow changes |

False reassurance should receive special attention because ADD is meant to constrain risky reliance. A model that looks accurate overall but repeatedly under-escalates consequential cases should be revised or rejected.

## Source Admission and Removal

New evidence sources should pass an admission review before becoming active:

1. Identify coverage: domains, tasks, model families, dates, and labels.
2. Identify missingness and bias.
3. Map source fields to the ADD evidence schema.
4. Run duplicate and leakage checks.
5. Assign an initial quality tier.
6. Run sensitivity tests with and without the source.
7. Mark the source active only if it improves coverage or model performance without introducing unacceptable distortion.

Sources should be removed or quarantined if they:

- duplicate other evidence without adding value,
- distort calibration through severity or reporting bias,
- produce unstable model weights,
- degrade holdout performance,
- cannot be mapped consistently to ADD factors,
- become stale because model behavior or workflow context changed,
- lack enough provenance to support audit or review.

Removal should preserve an audit trail. Do not silently delete sources from historical runs.

## Data Splits and Validation Design

ADD should avoid evaluating models on the same evidence used to tune them.

Recommended splits:

- Retrospective development set for early model building.
- Calibration set for setting weights, thresholds, priors, and transition assumptions.
- Holdout set for model comparison.
- Temporal holdout for later evidence or later model versions.
- Domain holdout for transfer testing across fields such as health, law, finance, education, and software.

When evidence is sparse, the project should prefer wider uncertainty and smaller claims over overfitted performance numbers.

## Versioning

Each run should record:

- data snapshot version,
- schema version,
- source registry version,
- feature transformation version,
- model version,
- prior version,
- ensemble weight version,
- threshold version,
- evaluation metric version.

This allows later readers to answer: what data was used, what assumptions were active, what changed, and whether performance improved.

## Privacy and Governance

Evidence collection should follow data minimization. Deployment logs, user studies, or internal case records may contain sensitive information. ADD should avoid collecting personal data unless it is necessary, lawful, consented where required, and governed by clear access controls.

Public examples should be redacted or summarized when needed. The evidence architecture should support raw artifacts, but public reporting should generally point to derived fields and review notes rather than exposing sensitive inputs.

## Initial Implementation Path

The first backend implementation should create:

- `core_model/evidence_schema.py` for dataclasses or typed records.
- `core_model/source_registry.py` for source metadata and active/quarantined status.
- `core_model/evidence_corpus.py` for calibration filtering, feature rows, coverage summaries, reproducible splits, and data snapshots.
- `core_model/evidence_io.py` for loading local source registry JSON and evidence JSONL into the corpus.
- `core_model/performance_metrics.py` for calibration, scoring, and false reassurance metrics.
- `tests/test_evidence_schema.py` for schema validation and quality-tier behavior.
- `tests/test_evidence_corpus.py` for corpus validation, traceability, splitting, and snapshot behavior.
- `tests/test_evidence_io.py` for local data loading and load-error context.
- `tests/test_source_registry.py` for source admission, deprecation, and audit trail behavior.

This should come before a GUI or end-user workflow. The immediate goal is to make ADD's evidence base inspectable, testable, and flexible enough to survive better data.

## Reference Implementation

The first backend evidence implementation lives in `core_model/evidence_schema.py`, `core_model/source_registry.py`, `core_model/evidence_corpus.py`, `core_model/evidence_io.py`, and `core_model/performance_metrics.py`.

It does not yet collect real evidence or tune model weights. Its purpose is to define the typed records, source lifecycle behavior, corpus bridge, local file-backed loading path, reproducible snapshot metadata, and metric calculations that future calibration and model-comparison runs will need.

## Open Research Questions

- Which evidence types best predict oversight requirement rather than merely documenting harm after the fact?
- How much weight should expert elicitation receive when empirical evidence is sparse?
- How should ADD represent disagreement among qualified reviewers?
- How should model-version drift change priors and source relevance?
- Which domains require separate models rather than shared weights?
- When should the framework refuse to score because evidence is too weak or missing?

## Success Criteria

The evidence architecture is successful when:

- evidence records can be traced back to source context,
- new sources can be added without changing model code,
- weak sources widen uncertainty rather than creating false precision,
- sources can be quarantined or removed with an audit trail,
- Bayesian, statistical, rubric, Markov, and ensemble models can all consume the same evidence layer,
- validation metrics can identify models or sources that are unstable, misleading, or not useful.

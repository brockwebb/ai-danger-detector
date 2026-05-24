# ADD Adjudication Protocol

Date: 2026-05-24

## Purpose

This protocol defines how a case becomes ADD evidence. It is provisional and does not validate ADD by itself.

The goal is to make evidence labels inspectable, repeatable, and honest about uncertainty. A case should not enter calibration merely because it is interesting. It should have enough provenance, review, and labeling consistency to support the claim being made from it.

Synthetic or illustrative cases are excluded from calibration by default.

## Case Intake

Each candidate case should begin with an intake record. The intake record should identify:

- the source and source owner,
- collection date and event date when known,
- domain and task type,
- model or system family and version when known,
- user expertise and workflow context,
- available artifacts or citations,
- known privacy, consent, or access constraints,
- reason the case is being considered,
- expected relevance limits.

Candidate cases may include incidents, near misses, benign comparisons, benchmark failures, user-study observations, deployment logs, expert-labeled hypotheticals, or synthetic stress cases. The case type must be explicit.

Cases should be excluded or quarantined when provenance is too weak, privacy cannot be handled responsibly, the record duplicates another case without adding information, or the case cannot be mapped consistently to ADD fields.

## Reviewer Roles

ADD separates review roles so that one person does not silently decide every label.

| Role | Responsibility |
| --- | --- |
| Intake reviewer | Confirms that the case has enough source context to enter review. |
| Primary labeler | Applies ADD evidence fields and records rationale. |
| Second labeler | Labels the same case independently when feasible. |
| Domain reviewer | Checks domain-specific assumptions, severity, detectability, and verification burden. |
| Adjudicator | Resolves or records disagreements and assigns the adjudicated label used for analysis. |
| Data steward | Confirms privacy, access, redaction, source status, and versioning rules. |

For low-risk illustrative examples, one reviewer may fill multiple roles, but the record must say that review was not independent. Calibration evidence should use independent review whenever feasible.

## Label Workflow

Each reviewed case should follow this sequence:

1. Intake reviewer decides whether the case enters review, is rejected, or is quarantined.
2. Labelers independently assign evidence fields when independent review is available.
3. Reviewers record rationale, uncertainty, and missing information.
4. Domain reviewer comments on domain-specific assumptions and transfer limits.
5. Adjudicator records the final adjudicated labels and unresolved disagreements.
6. Data steward assigns or confirms source status, evidence quality tier, and calibration eligibility.
7. The corpus snapshot records the evidence and source versions used in any run.

The final record should preserve individual reviewer disagreement as evidence, not erase it as noise. If reviewers disagree in a way that changes the oversight band or outcome interpretation, the case should be flagged for sensitivity analysis or excluded from calibration until resolved.

## Required Labels

Each evidence unit should include the fields defined in the evidence architecture:

- stable `evidence_id`,
- linked `source_id`,
- evidence type,
- collection and event dates,
- domain and task type,
- model family and version when known,
- user expertise,
- governance context,
- outcome label,
- oversight label,
- harm severity,
- detectability,
- reversibility,
- verification burden,
- workflow path,
- confidence,
- source quality,
- bias notes,
- relevance limits.

Optional fields can record financial loss estimates, time loss estimates, affected population, regulatory or rights impact, raw artifact references, reviewer identifiers, inter-rater agreement, adjudication notes, and citations.

## Outcome Labels

Outcome labels should describe what happened, not what reviewers feared might happen.

| Label | Use when |
| --- | --- |
| `harm` | A meaningful harm occurred. |
| `loss` | A material financial, operational, time, legal, or opportunity loss occurred. |
| `near_miss` | A meaningful issue was caught or avoided before material impact. |
| `corrected_error` | AI output was materially wrong or misleading but corrected before action. |
| `benign_use` | No meaningful error, harm, or near miss is evident from the record. |
| `unresolved` | The outcome cannot be adjudicated from available evidence. |
| `unknown` | The outcome is not available or not applicable. |

When a case contains multiple outcomes, reviewers should record the primary adjudicated outcome and note secondary outcomes in optional fields or adjudication notes.

## Oversight Labels

Oversight labels should answer: what minimum human review posture would have been appropriate before the AI-assisted output influenced action?

| Label | Meaning |
| --- | --- |
| `casual_exploratory` | AI use is limited to brainstorming, drafting, or low-consequence exploration. |
| `assisted_bounded` | AI may assist within clear limits, with ordinary user checking and easy correction. |
| `trained_review_required` | A trained reviewer should check the output before it affects action. |
| `expert_review_required` | A qualified expert should review before action. |
| `expert_led_or_no_autonomous_use` | The task should be expert-led, or AI should not be used autonomously for the consequential decision. |
| `unknown` | The record is too weak to assign an oversight posture. |

Labels should consider user expertise, harm severity, reversibility, detectability, verification burden, workflow controls, and the authority of the AI-assisted output.

## Scoring Anchors

Numerical fields use the interval from 0.0 to 1.0. These anchors are guideposts, not claims of precision.

### Harm Severity

- 0.0 to 0.2: negligible or easily corrected inconvenience.
- 0.2 to 0.4: minor rework, confusion, or low-impact loss.
- 0.4 to 0.6: meaningful operational, financial, educational, or personal impact.
- 0.6 to 0.8: serious harm, rights impact, material loss, or high-stakes consequence.
- 0.8 to 1.0: severe, irreversible, safety-critical, legal, medical, or life-altering impact.

### Detectability

- 0.0 to 0.2: typical user is unlikely to detect the issue before action.
- 0.2 to 0.4: issue may be detected by a trained or unusually careful user.
- 0.4 to 0.6: issue is detectable with structured checking or comparison.
- 0.6 to 0.8: issue is likely to be detected by the intended reviewer.
- 0.8 to 1.0: issue is obvious or mechanically verifiable before action.

### Reversibility

- 0.0 to 0.2: hard or impossible to undo after action.
- 0.2 to 0.4: correction is possible but costly, delayed, or incomplete.
- 0.4 to 0.6: correction is feasible with meaningful effort.
- 0.6 to 0.8: correction is usually practical before major impact.
- 0.8 to 1.0: correction is easy and low-cost.

### Verification Burden

- 0.0 to 0.2: ordinary user can verify quickly.
- 0.2 to 0.4: checking requires care but little specialized knowledge.
- 0.4 to 0.6: checking requires trained review, tools, or structured process.
- 0.6 to 0.8: checking requires domain expertise or substantial time.
- 0.8 to 1.0: checking requires scarce expertise, high effort, or access to external records.

If reviewers cannot justify a narrow value, they should record lower confidence and preserve uncertainty rather than force agreement.

## Workflow Path Coding

Workflow paths use the numerical framework states:

| State | Meaning |
| --- | --- |
| S0 | AI use proposed |
| S1 | AI output generated |
| S2 | Non-expert acceptance |
| S3 | Non-expert checking |
| S4 | Expert review |
| S5 | Error detected or corrected |
| S6 | Action taken |
| S7 | Harmless outcome |
| S8 | Realized harm or loss |

The path should represent the observed or best-supported sequence. If the sequence is inferred, reviewers should record that limitation. S7 and S8 are terminal outcomes. S5 may lead to corrected action, abandonment, renewed generation, or further review.

## Disagreement Handling

Reviewer disagreement is a signal. It can reveal ambiguous cases, weak evidence, unclear rubric wording, or domain-specific exceptions.

The adjudication record should preserve:

- labels assigned by each reviewer,
- rationale for material disagreements,
- final adjudicated label,
- whether disagreement changed the oversight band,
- whether the case remains calibration eligible,
- changes recommended to the protocol or rubric.

If disagreement remains unresolved, the case may still be retained for hypothesis generation, but it should not be used as strong calibration evidence.

## Quality Tiers

Quality tiering applies at both source and record level.

| Tier | Use when |
| --- | --- |
| Tier 1 | Clear provenance, independent review, adequate context, and consistent labels. |
| Tier 2 | Structured source with known limits and usable labels. |
| Tier 3 | Credible but incomplete report, near miss, or expert-labeled case. |
| Tier 4 | Anecdote, media summary, synthetic example, or weakly documented case. |
| Quarantined | Misleading, unstable, duplicated, privacy-blocked, or too biased for current use. |

Weak evidence may still be useful, but it should widen uncertainty and may be excluded from calibration.

## Calibration Eligibility

Calibration eligibility requires:

- active source status,
- adequate provenance,
- reviewed labels,
- documented reviewer role or adjudication process,
- no unresolved disagreement that changes the oversight band,
- no quarantine flags,
- source and record quality strong enough for the target analysis,
- documented relevance limits,
- inclusion in a versioned data snapshot.

Experimental, synthetic, illustrative, deprecated, removed, or quarantined sources are excluded from calibration by default. A future calibration run may include experimental sources only when the run explicitly says so and reports the sensitivity of results with and without those sources.

## Privacy and Governance

Cases may involve sensitive information. ADD should use data minimization by default.

Public records should avoid personal data, confidential prompts, protected information, private documents, and identifiers unless there is a lawful, consented, and necessary reason to include them. Raw artifacts should be referenced through controlled access when needed rather than copied into public evidence files.

If privacy constraints prevent meaningful review, the case should be summarized, redacted, quarantined, or excluded.

## Reporting Expectations

Any report based on adjudicated evidence should state:

- which sources were used,
- which cases were included or excluded,
- reviewer roles and independence,
- disagreement rates and themes,
- evidence quality distribution,
- known biases and missingness,
- calibration eligibility decisions,
- data snapshot version,
- model or rubric version,
- limitations on transfer to other domains or model versions.

Reports should distinguish what is observed, what is inferred, and what remains uncertain. A narrow numerical result is not credible unless the evidence and adjudication process can support that precision.

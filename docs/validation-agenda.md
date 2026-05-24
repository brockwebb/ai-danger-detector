# ADD Validation Agenda

Date: 2026-05-23

## Purpose

ADD is not validated by assertion. This document defines the evidence and evaluation work needed to tune, challenge, and improve the framework.

The agenda treats validation as an ongoing governance and technical process. ADD should become more useful when evidence is added, assumptions are challenged, and weak parts of the framework are revised. It should not be presented as an empirically validated safety instrument until that work has been done and documented.

## Evidence Sources

ADD validation should draw from multiple evidence sources because no single source can establish model reliability, workflow behavior, or real-world harm patterns.

- Documented incidents involving AI error, misuse, automation bias, or failed oversight.
- Domain expert panels that review factor definitions, oversight bands, example cases, and escalation thresholds.
- Structured expert elicitation that records uncertainty, disagreement, confidence, and rationale rather than forcing premature consensus.
- Benchmark and model evaluation results, especially when they are task-specific, versioned, reproducible, and relevant to the target workflow.
- Deployment logs where ethically and legally available, with attention to privacy, consent, data minimization, and reporting bias.
- User studies on error detection, including whether non-experts can notice, contest, and correct AI mistakes before action.
- Case reviews from high-stakes domains such as health, law, finance, education, employment, public benefits, infrastructure, and safety-critical operations.

Each evidence source should be tagged with date, domain, task type, model or system version, evidence quality, known biases, and relevance limits. Evidence that is anecdotal, sparse, or collected under unusual conditions should widen uncertainty rather than produce confident calibration.

## Calibration Risks

ADD calibration will be vulnerable to predictable distortions. These risks should be tracked explicitly when setting priors, weights, transition probabilities, and oversight thresholds.

- Human overprediction of vivid harms: dramatic incidents can receive too much weight compared with quieter base-rate evidence.
- Underreporting of mundane failures: routine errors, near misses, rework, delays, and small losses may be invisible if users do not report them.
- Survivorship bias: successful or harmless AI uses may be easier to observe in some settings, while abandoned, blocked, or failed uses disappear from the record.
- Domain skew: evidence from one domain, user group, organization, language, or model family may not transfer to another.
- Model-version drift: new model releases, retrieval systems, tool integrations, prompting practices, safeguards, and interfaces can change error rates and user behavior.
- Incentives to downplay risk: vendors, teams, or users may minimize failures to protect reputation, adoption, funding, or operational freedom.
- Incentives to exaggerate risk: advocates, competitors, institutions, or reviewers may overstate risk to advance policy, commercial, reputational, or ideological goals.

Calibration should preserve uncertainty when these risks cannot be resolved. A transparent wide range is preferable to a narrow estimate that hides weak evidence.

## Validation Milestones

1. Face-validity review by domain experts.
2. Retrospective scoring of documented cases.
3. Sensitivity analysis over weights, priors, and transition probabilities.
4. Inter-rater reliability testing across reviewers.
5. Prospective use-case triage in low-risk organizational settings.
6. Periodic recalibration as models and evidence change.

These milestones should be treated as cumulative checks. Passing an early milestone does not validate the full framework. For example, face-validity review can show that the rubric is understandable and plausible, but it cannot prove predictive accuracy or operational reliability.

Face-validity review should ask domain experts whether the factors, oversight bands, examples, and escalation logic match known failure modes. Reviewers should identify missing factors, confusing terms, domain-specific exceptions, and cases where the rubric would encourage either excessive confidence or unnecessary restriction.

Retrospective scoring should use a documented case set with inclusion criteria. The case set should define which incidents, near misses, benign uses, and comparison cases are included; why they are included; and what evidence is available for each case. Case labeling should follow `docs/adjudication-protocol.md`, including reviewer roles, label anchors, disagreement handling, source quality tiers, and calibration eligibility gates. Independent reviewers should apply ADD without seeing the final adjudicated label when feasible. A separate adjudication process should record harm outcomes, oversight outcomes, and whether escalation would have been appropriate. Validation should then compare ADD escalation against those adjudicated outcomes, including under-escalation, over-escalation, and cases where the available record is too weak for a confident judgment.

Sensitivity analysis should test whether conclusions depend on fragile choices of weights, priors, thresholds, or Markov transition probabilities. Reports should identify which assumptions most affect the oversight band, the probability of threshold crossing, and workflow outcomes such as unverified action or realized harm.

Inter-rater reliability testing should have independent reviewers apply the rubric to the same cases using the same written instructions. Reports should include agreement statistics appropriate to the rating structure, disagreement themes, and examples of factor definitions that caused inconsistent scoring. Ambiguous rubric language should be revised and retested rather than treated as reviewer error by default.

Prospective triage should begin in low-risk organizational settings where AI use can be observed without exposing people to severe, irreversible, or rights-impacting consequences. Each pilot should define pre-specified metrics such as escalation frequency, reviewer burden, user acceptance, correction rates, near misses, under-escalation concerns, and cases abandoned because review requirements were too high. Escalations, overrides, and near misses should be reviewed to determine whether ADD changed decisions in a useful, explainable, and proportionate way.

Recalibration should happen on a periodic cadence and when material triggers occur. Triggers include model or system version changes, workflow changes, prompt or interface changes, retrieval or tooling changes, newly observed failures or near misses, benchmark regressions, policy changes, and credible domain-transfer evidence showing that assumptions from one setting do or do not apply in another. Recalibration reports should preserve prior versions of priors, weights, thresholds, transition assumptions, and rubric text so readers can see what changed and why.

## Failure Criteria

ADD should be revised if validation work shows any of the following patterns:

- It repeatedly under-escalates moderate-or-higher harm cases.
- Different reviewers cannot apply the rubric consistently.
- Numerical outputs imply precision unsupported by evidence.
- Markov transition assumptions do not match observed workflow behavior.
- Model improvements make old priors obsolete.

Revision may include changing factor definitions, adjusting weights, widening uncertainty intervals, replacing priors, changing oversight thresholds, adding domain-specific guidance, or narrowing the scope where ADD should be used. If evidence shows that a domain cannot be evaluated reliably with the current framework, ADD should say so rather than force a score.

## Reporting Expectations

Validation reports should separate what is known, what is assumed, and what remains uncertain. At minimum, a report should identify the evidence sources used, the cases reviewed, reviewer roles, disagreements, sensitivity results, model or system versions, and any changes made to the framework.

Numerical results should be reported as provisional decision-support outputs, not as precise measurements of safety. Governance readers should be able to see why a scenario escalated, what evidence supported that judgment, and what would cause the conclusion to change.

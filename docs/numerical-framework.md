# ADD Numerical Framework

Date: 2026-05-23

## Purpose

This document describes the numerical foundation for ADD: Bayesian calibration for uncertain assumptions, Monte Carlo simulation for propagating uncertainty, and Markov reliance workflow modeling for how AI-assisted decisions move through checking, escalation, action, correction, or harm.

The framework is provisional. It defines a structure that future code can implement and test, but it does not claim empirical validity yet.

## Method Overview

ADD uses the three methods together:

- Bayesian calibration describes how beliefs about error, severity, detectability, and reversibility should update as evidence arrives.
- Monte Carlo simulation propagates uncertainty through the oversight model so outputs can be reported as ranges and threshold-crossing probabilities.
- Markov workflow modeling represents AI reliance as a sequence of states rather than a one-time score.

The point of combining these methods is to make uncertainty visible. ADD should not hide weak evidence behind a single confident score. It should show the range of plausible oversight requirements and the workflow paths that create risk.

## Bayesian Calibration

Bayesian calibration treats ADD parameters as uncertain quantities. Initial priors should be explicit, documented, and revisable. As evidence arrives from expert review, incident reports, controlled evaluations, near-miss analysis, field studies, or deployment monitoring, priors can be updated into posteriors.

Those updates require more than collecting examples. Each update should specify an explicit likelihood or evidence-weighting rule, source-quality notes, an assessment of bias and missingness, and uncertainty widening when evidence is anecdotal, sparse, or not comparable to the target workflow.

| Parameter | Meaning | Example prior form |
| --- | --- | --- |
| p_error_per_task | Bounded task-level probability of a meaningful AI error or misuse event | Beta distribution or bounded triangular distribution |
| severity | Magnitude of harm/loss if an error matters | Triangular, log-normal, or expert-elicited ordinal distribution |
| detectability | Probability a user detects a meaningful error before action | Beta distribution conditioned on expertise |
| reversibility | Probability harm can be corrected after action | Beta distribution conditioned on domain and action type |
| verification_burden | Effort or expertise required to check the output | Ordinal distribution mapped to oversight bands |

A bounded task-level probability such as `p_error_per_task` is suitable for Beta priors. If ADD needs to model event intensity over time, exposure, transactions, or usage volume, it should use a separate `lambda_error_rate` parameter with Gamma or log-normal priors and Poisson or negative-binomial count models. This preserves the lambda/rate concept without forcing a Beta prior onto an unbounded rate.

Priors should be versioned by domain, model family, task type, and date. A prior for consumer medical triage should not silently carry over to legal drafting, financial analysis, software deployment, or public-benefits administration. A prior for one model family or release date may also become stale as model capability, interface design, retrieval quality, prompting practices, and governance controls change.

Calibration should preserve disagreement when experts do not agree. A wide or multimodal prior can be more honest than forcing a single consensus value.

## Monte Carlo Simulation

Monte Carlo simulation propagates uncertainty through the ADD model. Instead of producing only one oversight score, the simulation repeatedly samples uncertain parameters and records the resulting oversight band and threshold crossings.

```text
for simulation in 1..N:
    sample p_error_per_task from posterior or prior
    sample severity from posterior or prior
    sample detectability from posterior or prior
    sample reversibility from posterior or prior
    sample verification_burden from posterior or prior
    sample transition probabilities from posterior or prior

    calculate oversight_score
    assign oversight_band
    run or analytically evaluate the Markov workflow

    record whether thresholds are crossed:
        trained_review_required
        expert_review_required
        expert_led_or_no_autonomous_use
    record workflow outcomes:
        unverified_action
        realized_harm
        expected_loss

report:
    median oversight_score
    uncertainty interval
    P(trained_review_required)
    P(expert_review_required)
    P(expert_led_or_no_autonomous_use)
    P(unverified_action)
    P(realized_harm)
    expected_loss
```

The simulation output should be read as a decision-support summary, not as a precise measurement. The most useful outputs are ranges, probabilities of crossing review thresholds, workflow outcome probabilities, expected loss estimates, and sensitivity to assumptions.

## Markov Workflow Model

Markov workflow modeling represents AI reliance as movement through states. This is useful because harm often depends less on the initial AI output alone than on whether the output is checked, escalated, corrected, or converted into action.

| State | Label | Description |
| --- | --- | --- |
| S0 | AI use proposed | A user or organization considers using AI for a task. |
| S1 | AI output generated | The system produces an output that may be correct, incomplete, wrong, or misused. |
| S2 | Non-expert acceptance | A non-expert accepts the output without meaningful checking or escalation. |
| S3 | Non-expert checking | A non-expert attempts to verify, compare, or reason through the output. |
| S4 | Expert review | A trained or expert reviewer evaluates the output before action. |
| S5 | Error detected/corrected | A meaningful error is found, corrected, escalated, or contained. |
| S6 | Action taken | The output influences a decision, communication, operation, or other real-world action. |
| S7 | Harmless outcome | The workflow exits without material harm or loss. |
| S8 | Realized harm/loss | The workflow exits with harm occurrence or material impact. |

S7 and S8 are terminal states. S5 is not necessarily terminal: it may loop back to S1 for a regenerated output, back to S3 for additional checking, back to S4 for expert review, or out of the workflow if the use is abandoned.

S8 represents that material harm or impact occurred; it is not itself the severity magnitude. Severity or loss should be attached as a separate conditional cost variable when S8 is reached or when explicitly modeled risky transitions occur.

The Markov layer can be used to estimate not only "how risky is this use?" but also "where should the workflow add checkpoints?" A scenario with modest error rates can still be dangerous if transition probabilities push users from generated output to unverified action.

## Transition Probability Logic

Transition probabilities should be calibrated by domain, workflow design, user expertise, governance controls, and evidence about model behavior. Until calibration exists, they should be treated as explicit assumptions.

- Higher user expertise increases P(S3 -> S5) and P(S3 -> S4).
- Higher domain complexity decreases P(S3 -> S5) for non-experts.
- Strong governance increases P(S1 -> S4) and decreases P(S1 -> S2).
- Higher detectability increases error correction before action.
- Lower reversibility increases conditional loss given erroneous action or realized harm. It should not be modeled as a Markov state probability unless the implementation explicitly encodes reversibility as a transition effect.
- Model improvement can reduce P(error at S1), but does not eliminate verification burden.

These assumptions are directional, not validated coefficients. Future implementation should expose them for review, sensitivity testing, and revision.

## Example Outputs

The following values are illustrative, not calibrated. Their purpose is reporting shape: they show the kind of output ADD should produce once priors, scoring rules, thresholds, and transition probabilities are defined.

- 82% probability that trained review is required.
- 46% probability that expert review is required.
- 18% simulated probability of unverified action.
- 6% simulated probability of realized material harm/loss.

These numbers should not be used to judge any real deployment. A real estimate would require documented assumptions, domain-specific priors, evidence quality notes, sensitivity analysis, and versioning.

## Limitations

- False precision: numerical outputs may look more certain than the evidence supports.
- Sparse evidence: many task classes lack reliable base rates for AI errors, misuse, near misses, or harms.
- Expert disagreement: qualified reviewers may differ on severity, detectability, reversibility, and acceptable oversight.
- Overfitting to dramatic incidents: vivid failures can distort priors if they are not balanced against base rates and comparison cases.
- Changing model capability: model releases, tool use, retrieval systems, prompting, safeguards, and user interfaces can change risk assumptions quickly.
- Transition probabilities requiring calibration: workflow movement from output to checking, escalation, action, correction, or harm is not yet empirically grounded.
- No empirical validity yet: ADD is a research and governance design framework, not a validated safety instrument.

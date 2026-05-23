# AI Danger Detector Whitepaper Foundation Design

Date: 2026-05-23

## Purpose

Evolve AI Danger Detector from an abandoned prototype into a credible public foundation: a governance-facing whitepaper with enough technical structure that researchers, builders, and responsible AI practitioners can reference, critique, and extend it.

The work should plant a stake in the ground without overstating maturity. The project should say: this is a provisional, inspectable framework for estimating when AI-assisted work requires human expertise, expert review, or restraint.

## Audience

Primary audience: AI governance, responsible AI, risk, compliance, and policy-adjacent practitioners.

Secondary audience: technical builders and researchers who may want to improve the model, calibrate weights, build tools, or challenge assumptions.

The work is not aimed primarily at general consumer advice. It should remain readable, but it should not dangerously simplify the core claim. The practical stance is that when potential harm or loss is moderate or greater, non-expert use should be bounded to ideation, drafting, exploration, or supervised assistance rather than final decision-making.

## Core Thesis

For AI-assisted work, the minimum safe human expertise is a function of potential harm, task complexity, model error uncertainty, reversibility, verification burden, and the user's ability to detect and correct mistakes.

Risk-bearing AI use requires domain competence because harm detection is itself domain work.

## Deliverables

Create five public-facing project artifacts:

1. `docs/whitepaper.md`
   - Main governance-facing paper.
   - Polished enough for GitHub, LinkedIn, and lightweight public reference.
   - Clear that the framework is provisional and not peer-reviewed.

2. `docs/model-rubric.md`
   - Technical foundation for the scoring/rubric approach.
   - Defines factors, example scales, decision bands, and how weights should be interpreted.
   - Explains that current parameters are assumptions requiring calibration.

3. `docs/numerical-framework.md`
   - Technical explanation of the numerical machinery behind the whitepaper.
   - Defines the three core layers: Bayesian calibration, Monte Carlo uncertainty propagation, and Markov reliance workflow modeling.
   - Describes state definitions, transition probabilities, threshold-crossing estimates, and how uncertainty should be reported.
   - Makes clear that priors, transition probabilities, and thresholds are provisional and require tuning.

4. `docs/validation-agenda.md`
   - Honest research and validation roadmap.
   - Describes how to tune weights using documented cases, incident evidence, expert elicitation, sensitivity analysis, and empirical testing.
   - Names the risks of anecdotal skew and human overprediction of vivid negative events.

5. `README.md` update
   - Reframe the repository away from a finished "danger detector" product.
   - Link to the whitepaper and supporting docs.
   - Keep the existing prototype visible as a reference implementation, not a validated tool.

Code cleanup, tests, CLI work, or a web demo are future phases. They are outside this first artifact-focused scope unless needed to avoid misleading claims.

## Whitepaper Structure

`docs/whitepaper.md` should contain:

1. Abstract
   - AI risk depends not only on model capability, but on whether the human user can recognize, contest, and mitigate wrong outputs.

2. Problem
   - AI advice is cheap, fluent, and increasingly available.
   - Non-expert reliance becomes risky in domains where mistakes are consequential.

3. Central claim
   - Required human expertise rises with harm, complexity, uncertainty/error rate, irreversibility, and verification difficulty.

4. The ADD framework
   - Define harm/loss potential, domain complexity, error uncertainty, reversibility, verification burden, user expertise, and oversight requirement.

5. Decision rule
   - Low harm: non-expert use may be acceptable with ordinary checks.
   - Moderate harm/loss: non-expert use should be bounded to ideation or drafting; trained or expert review is needed before action.
   - High/critical harm: expert-led use only; AI should not become autonomous authority.

6. Reference model
   - Present the current formula as an inspectable early scoring model.
   - Avoid presenting it as final, objective, validated, or authoritative.

7. Numerical framework
   - Introduce Bayesian calibration for uncertain assumptions.
   - Introduce Monte Carlo simulation for propagating uncertainty through oversight thresholds.
   - Introduce Markov-state reliance modeling for AI use workflows that move through generation, checking, escalation, action, correction, and harm states.
   - Point technical readers to `docs/numerical-framework.md`.

8. Calibration, noise, and the moving target problem
   - Treat this as a core methodological section.
   - Explain rate/lambda, severity, detectability, reversibility, model improvement, and expert judgment.

9. Governance alignment
   - Map the framework to NIST AI RMF, NIST Generative AI Profile, EU AI Act high-risk concepts, ISO/IEC 42001 management-system thinking, and incident taxonomy work.

10. Limitations
   - No empirical calibration yet.
   - Domain weights are provisional.
   - Error rates are difficult to estimate.
   - The project is not legal, medical, financial, safety, or compliance advice.

11. Research and builder agenda
   - Invite calibration, domain testing, alternative scenario classifiers, richer datasets, and validation against real or documented outcomes.

12. Technical appendix
   - Formula, parameters, examples, saturation issue, and proposed next-generation rubric.

## Methodological Stance

ADD is an inspectable oversight triage framework, not an oracle.

Its technical foundation should combine three methods:

1. Bayesian calibration
   - Use priors to represent initial assumptions about error rates, severity, detectability, reversibility, verification burden, and domain complexity.
   - Update those assumptions as documented cases, expert review, incident data, model evaluations, and deployment evidence become available.
   - Treat expert judgment as evidence with uncertainty, not as unquestionable truth.

2. Monte Carlo uncertainty propagation
   - Sample plausible parameter ranges rather than relying on single-point estimates.
   - Estimate how often a scenario crosses oversight thresholds under conservative, moderate, and permissive assumptions.
   - Report uncertainty bands and threshold-crossing probabilities rather than only point scores.

3. Markov reliance workflow modeling
   - Model AI-assisted work as a sequence of states, not a one-time score.
   - Core states should include at least: AI use proposed, AI output generated, non-expert acceptance, non-expert checking, expert review, error detected/corrected, action taken, harmless outcome, and realized harm/loss.
   - Transition probabilities should depend on model reliability, domain complexity, user expertise, detectability, reversibility, and governance controls.
   - The Markov layer is core because ADD is about reliance pathways: whether a system tends toward unchecked action or toward verification, escalation, correction, and safer outcomes.

The weights and parameters are provisional. They should be versioned, challenged, and tuned through a combination of:

- documented cases and incident data,
- domain expert review,
- sensitivity analysis,
- empirical validation where feasible,
- structured expert elicitation when empirical evidence is sparse,
- updates as model capabilities and error profiles change.

The framework should distinguish:

- Rate/lambda: how often meaningful AI errors or misuse events occur.
- Severity: how damaging the outcome can be when an error matters.
- Detectability: whether the user can notice or contest the error.
- Reversibility: whether the harm can be corrected after the fact.
- Model drift/improvement: whether assumptions remain current.

The paper should explicitly note that human judgment is necessary but noisy. People can overpredict dramatic negative events, especially when examples are recent, vivid, or socially amplified. Incident anecdotes should inform calibration but should not dominate it. The framework should use ranges, confidence labels, sensitivity analysis, and expert review rather than pretending fear is evidence.

## Numerical Framework Document

`docs/numerical-framework.md` should be written for the technical secondary audience. It should include:

- a plain-language overview of why Bayesian, Monte Carlo, and Markov methods are all used;
- a proposed parameter set and prior assumptions;
- example distributions for rate/lambda, severity, detectability, and reversibility;
- a simple threshold-crossing simulation structure;
- a Markov state diagram or table for reliance workflows;
- example transition probability logic;
- example outputs such as probability of expert-review threshold crossing, probability of unverified action, and probability of realized harm/loss;
- warnings about over-calibration, false precision, sparse evidence, and moving model capabilities.

The document should not require a complete implementation in this first pass. It should define the numerical foundation clearly enough that future code can implement it.

## Good-Enough Standard

The goal is triage accuracy, not oracle accuracy.

ADD is good enough when it consistently identifies cases where non-expert reliance should be constrained, even if it cannot precisely predict the probability or magnitude of every failure.

For moderate-or-higher harm contexts, false reassurance is worse than conservative escalation. In low-harm contexts, false positives are usually annoying but tolerable. This asymmetry should be explicit.

The project should avoid perfection as the target. A useful baseline is one that distinguishes:

- casual low-risk use,
- supervised or trained use,
- expert review before action,
- expert-led use or no autonomous AI authority.

## Tone

The tone should be candid, serious, and constructive.

Acceptable framing:

- "A preliminary framework."
- "An inspectable reference model."
- "A foundation for calibration and critique."
- "A stake in the ground."

Avoid:

- Claims of scientific validation.
- Claims that the model detects safety objectively.
- Claims that the numeric score is authoritative.
- Claims that natural-language classification is implemented if it is not.

## Non-Goals

This first evolution does not need to:

- rebuild the Python package,
- create a full CLI,
- train ML models,
- build a web app,
- prove empirical validity,
- solve consumer AI literacy.

Those are candidate follow-up phases if the whitepaper generates useful feedback.

## Success Criteria

The first pass is successful if:

- the repository contains a coherent whitepaper and supporting docs,
- the README points readers to the paper without overclaiming,
- the model is described as provisional and falsifiable,
- Bayesian, Monte Carlo, and Markov methods are included as core numerical foundations,
- calibration and tuning uncertainty are treated as central,
- governance readers can understand why the framework matters,
- technical readers can see how to improve or challenge it.

# AI Danger Detector: A Framework for Estimating Human Expertise Requirements in AI-Assisted Work

Date: 2026-05-23

## Abstract

AI risk is often discussed as a property of the model or the application domain. AI Danger Detector (ADD) starts from a different but complementary premise: for many AI-assisted tasks, risk also depends on whether the human user has enough expertise to recognize, contest, escalate, and correct wrong outputs before they become consequential actions.

ADD proposes an inspectable oversight triage framework for estimating when AI use can remain casual, when it requires trained or expert review, and when AI should not function as autonomous authority. The framework is provisional. Its weights, parameters, priors, and transition assumptions require calibration against documented cases, expert judgment, incident evidence, model evaluations, and future empirical work.

## Core Thesis

Risk-bearing AI use requires domain competence because harm detection is itself domain work.

## Problem

AI advice is cheap, fluent, and increasingly available. General-purpose AI systems can now generate plausible answers, summaries, plans, diagnoses, legal-style analysis, financial reasoning, code, and policy language for users who may have little ability to evaluate the output.

That fluency changes the risk profile. A confident answer can mask uncertainty, hallucination, omission, misplaced confidence, or context failure. The surface quality of the response can make weak evidence look settled, can hide missing assumptions, and can invite users to treat an assistive system as if it were a qualified decision maker.

Non-expert reliance is most dangerous when mistakes are consequential, hard to detect, or hard to reverse. In those settings, the relevant question is not only "is the model capable?" but "is the human-in-the-loop capable of noticing when the model is wrong?"

ADD focuses on that second question. It treats AI-assisted work as a relationship among task risk, model uncertainty, verification burden, human expertise, and oversight controls.

## ADD Framework

ADD estimates the level of human oversight required before an AI-assisted output is used for consequential action. It does not estimate truth, correctness, or safety. A low ADD oversight requirement does not mean an output is true; it means the scenario appears more suitable for ordinary review. A high requirement means the scenario should be bounded, escalated, or expert-led.

The framework evaluates several factors:

- **Harm or loss potential:** the expected consequence if the AI output is wrong, incomplete, misused, or over-relied upon.
- **Domain complexity:** the specialized knowledge required to understand the task, its constraints, and its failure modes.
- **Error uncertainty:** uncertainty about model reliability in the specific task, domain, context, and deployment setting.
- **Detectability:** the likelihood that the user can notice, contest, or investigate an incorrect output before action.
- **Reversibility:** the ability to undo, repair, compensate for, or otherwise recover from a wrong AI-assisted action.
- **Verification burden:** the effort, evidence, review process, or external validation required to check the output responsibly.
- **User expertise:** the user's practical ability to detect errors, understand tradeoffs, escalate uncertainty, and correct the output.
- **Governance or oversight controls:** process safeguards such as review, approval, audit trails, monitoring, expert signoff, use restrictions, or prohibition.

Together, these factors shift attention from model output alone to the whole reliance pathway. The same model answer can be relatively low risk in a sandboxed brainstorming context and high risk when used as the basis for a medical, legal, financial, safety, rights-impacting, or operational decision.

## Decision Rule

ADD treats AI use as an oversight triage problem:

- **Low harm / low loss:** non-expert use may be acceptable with ordinary checks.
- **Moderate harm / moderate loss:** non-expert use should be bounded to ideation, drafting, exploration, or supervised assistance; trained or expert review is needed before action.
- **High or critical harm / loss:** AI use should be expert-led; AI output should not become autonomous authority.

The boundary is intentionally conservative. In moderate-or-higher harm contexts, false reassurance is more dangerous than conservative escalation.

## Reference Model and Numerical Framework

The early ADD prototype uses an inspectable reference model for estimating required expertise:

```text
E_required = E_min
           + beta_h * harm^alpha_h
           + beta_c * complexity^alpha_c
           + beta_e * error_uncertainty^alpha_e
           + beta_i * (harm * complexity * error_uncertainty)^alpha_i
```

This formula is inspectable and intended to become testable once variables, thresholds, calibration data, and validation cases are specified. It is not a final scientific model, and it should not be treated as an authoritative safety judgment.

The planned numerical layer will combine Bayesian calibration, Monte Carlo uncertainty propagation, and Markov reliance workflow modeling.

A companion `docs/numerical-framework.md` document will define the technical foundation for this numerical layer.

## Calibration, Noise, and Moving Target

ADD depends on quantities that are noisy, context-sensitive, and likely to change. Calibration must therefore treat the framework as a moving target rather than a fixed scorecard.

- **Rate/lambda:** how often meaningful errors, misuse events, omissions, or over-reliance events occur in a task class.
- **Severity:** how bad outcomes are when errors matter, including financial loss, health impact, rights impact, safety impact, delay, or operational disruption.
- **Detectability:** whether ordinary users, trained reviewers, or experts can notice and contest wrong outputs before reliance becomes action.
- **Reversibility:** whether bad outcomes can be corrected quickly, corrected only with cost or delay, or not fully corrected at all.
- **Model drift and model improvement:** current assumptions decay as models, tools, retrieval systems, prompts, safeguards, user interfaces, and deployment environments change.
- **Human overprediction of vivid negative events:** salient failures and anecdotes are useful signals, but they should not dominate calibration without base rates, comparison cases, and uncertainty estimates.

The goal is not perfect prediction. ADD is good enough when it reliably separates casual low-risk use from consequential use requiring trained or expert oversight.

## Governance Alignment

ADD is designed to align with existing AI governance work without claiming compliance certification. It can support risk identification, oversight design, documentation, and escalation decisions, but it is not a substitute for a formal management system, legal analysis, regulatory classification, or independent audit.

Relevant alignment targets include:

- NIST AI Risk Management Framework.
- NIST Generative AI Profile.
- EU AI Act high-risk concepts.
- ISO/IEC 42001 AI management systems.
- AI incident taxonomy and incident repository work.

ADD's contribution is a specific emphasis on human expertise requirements inside AI reliance pathways. Governance programs often ask whether a system is risky, whether controls exist, and whether the application is appropriate. ADD adds a practical question: who is competent to notice when the AI-assisted output is wrong before the output shapes a consequential decision?

## Limitations and Builder Agenda

ADD is provisional. It has no empirical calibration yet. Weights and priors are provisional. Transition probabilities need expert evaluation and empirical grounding. Error rates are difficult to estimate and change with model capabilities, deployment context, prompting, retrieval quality, user interface design, and user behavior.

This is not legal, medical, financial, safety, or compliance advice. ADD should be treated as a research and governance design framework, not as a validated instrument for approving high-stakes AI use.

The builder agenda is to:

- calibrate priors and weights using documented incidents, model evaluations, expert judgment, and controlled studies;
- test the framework against documented cases of AI failure, near-miss, misuse, and successful oversight;
- compare ADD recommendations with domain expert judgments across medicine, law, finance, education, safety, software, public administration, and organizational operations;
- implement Monte Carlo and Markov simulation so uncertainty and reliance pathways are visible rather than hidden;
- build scenario classifiers only after the rubric is stable enough to avoid automating weak assumptions;
- version assumptions over time as models, tasks, laws, organizational practices, and incident evidence change.

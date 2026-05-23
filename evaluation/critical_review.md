# Critical Review: AI Danger Detector

Date: 2026-05-23

## Executive Decision

AI Danger Detector is worth refreshing only as a narrow, transparent prototype or public essay/demo about one useful question:

> How much human expertise should be in the loop before using AI for this task?

It is not currently worth advertising as a reliable product, objective detector, or scientifically validated tool. The current repo is a concept prototype with a hand-tuned formula, broken application entry points, no dependency manifest, no test suite, and claims in the README that exceed what the implementation supports.

The most valuable part is the framing. Most AI risk tools ask "is this system compliant?" or "how risky is this model?" This project asks a more user-centered question: "Am I the right person to rely on AI here, or do I need expert review?" That remains a good angle.

## Publish Or Advertise?

Recommendation:

- Free publishing: yes, but only after a cleanup pass and with careful framing.
- Free advertising: very lightly, after refresh. LinkedIn, GitHub, and a short blog post are appropriate. Do not push it as a finished tool.
- Paid advertising or product-style launch: no.

Good public framing:

- "A prototype rubric for estimating the expertise needed to safely use AI in a task."
- "A decision aid for human oversight, not a detector of truth or safety."
- "An educational model with visible assumptions."

Avoid these claims unless the project is rebuilt and externally validated:

- "Science-based."
- "Objective classification."
- "Validated AI safety detector."
- "Works from natural language descriptions."
- "Tells you when AI is safe."

## Why It Might Still Be Worth Saving

The central idea has legs because current AI governance work increasingly emphasizes risk classification, human oversight, traceability, monitoring, and post-deployment accountability. The project's "expertise required" framing fits this landscape if it is aligned with established frameworks rather than presented as a standalone science.

Useful direction:

- Map scenarios to harm, complexity, reversibility, user vulnerability, evidence quality, model reliability, and required human review.
- Output practical oversight guidance instead of a single "danger score."
- Cite and align with NIST AI RMF, the NIST Generative AI Profile, EU AI Act risk categories, and ISO/IEC 42001.

## Why It Should Not Be Promoted Yet

The repo does not currently support its public claims.

Evidence:

- README promises natural-language scenario classification, but the code exposes numeric inputs for harm, complexity, and error rate.
- README claims rigorous analytical models and scientific validation, but the project uses hand-set domain parameters and synthetic data.
- The app path is broken: `applications/interactive_assessment.py` has an `IndentationError`.
- `applications/__init__.py` imports `sensitivity_analyzer`, but the file is named `sensitivity_tool.py`.
- There is no dependency manifest, so a fresh environment cannot import most packages.
- `machine_learning/validation.py` is empty despite the README listing it as ML validation.
- Many outputs saturate at 10, making the score less discriminating for medical, legal, and financial domains.

## Product Viability

As a product, this is not compelling yet. The market has moved toward governance platforms, audit evidence, policy mapping, vendor/model inventories, and compliance workflows. A formula-only calculator would be easy to dismiss unless it becomes either:

- a lightweight public educational tool with excellent explanation, or
- a narrow component inside a larger AI governance/risk workflow.

The better route is the first one: a small, honest educational demo that helps people reason about when expertise matters.

## Best Revival Shape

Recommended scope: "AI Expertise Gate."

Inputs:

- Domain/use case.
- Potential harm.
- Domain complexity.
- Reversibility of mistakes.
- User vulnerability.
- Model reliability/evidence quality.
- Whether expert review is available.

Outputs:

- Required oversight level.
- Suggested reviewer role.
- What to verify before acting.
- Whether AI should be used for ideation only, supervised assistance, or decision support.
- A visible explanation of assumptions.

## Go/No-Go Criteria

Refresh only if you are willing to do a short, disciplined rebuild:

- Fix packaging and imports.
- Replace overclaims with transparent limitations.
- Add a minimal CLI or web demo.
- Add tests for monotonicity, boundaries, and app smoke tests.
- Add a source-backed rubric aligned with current AI risk frameworks.
- Publish as educational/prototype work, not validated advice.

Stop if the goal is a general-purpose risk detector or ML-trained safety classifier. That path would require data, validation, domain experts, and ongoing maintenance well beyond the likely payoff.

## Final Call

Worth refreshing: yes, as a small, thoughtful, governance-aware educational tool.

Worth evolving into a serious product: not without a major repositioning and validation budget.

Worth free publishing: yes, after a cleanup pass.

Worth advertising now: no.

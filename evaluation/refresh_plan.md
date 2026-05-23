# Refresh Plan

Date: 2026-05-23

## Recommended Path: Small Refresh

Goal: turn the project into a credible educational prototype in 1-2 focused sessions.

Deliverable:

> AI Danger Detector: a transparent prototype for estimating the human expertise and oversight needed for AI-assisted tasks.

## Phase 1: Stabilize

1. Add dependency file.
2. Fix broken imports and indentation.
3. Remove generated notebook/checkpoint clutter from consideration.
4. Add minimal smoke tests.
5. Make the calculator importable in a clean environment.

Acceptance checks:

```bash
python3 -m compileall -q $(git ls-files '*.py')
python3 data_generation/calculator.py
python3 - <<'PY'
from core_model.model_definition import expertise_required
print(round(expertise_required(5, 5, 0.2), 2))
PY
```

## Phase 2: Reframe

Rewrite README around limitations and actual capabilities.

Replace:

- "detect when AI is safe"
- "objective"
- "science-based"
- "rigorous validation"
- "natural language classification"

With:

- "transparent prototype"
- "human expertise and oversight estimator"
- "assumption-driven model"
- "educational decision aid"
- "not medical, legal, financial, or safety advice"

## Phase 3: Simplify The Model

Current formula:

```text
expertise = E_min + harm_component + complexity_component + error_component + interaction
```

Problems:

- Too many hand-tuned coefficients.
- Heavy saturation at 10 in high-stakes domains.
- Error rate is hard for normal users to estimate.
- The ML layer trains against synthetic labels, not real-world validation.

Suggested replacement inputs:

- Harm severity.
- Reversibility.
- Domain complexity.
- User vulnerability.
- Evidence/verifiability.
- Model reliability/confidence.
- Available expert review.

Suggested outputs:

- Expertise required: basic, familiar, trained, expert.
- Oversight mode: self-check, second review, expert review, do not use AI for final decision.
- Verification checklist.
- Explanation of which factors drove the result.

## Phase 4: Build One Polished Demo

Choose one:

- CLI questionnaire.
- Static web page.
- Jupyter notebook.

Best choice for free publishing: static web page or simple CLI plus README examples.

Avoid building:

- Full ML pipeline.
- Large synthetic dataset generator.
- Complex dashboard.
- Natural-language classifier before the rubric is stable.

## Phase 5: Publish Carefully

Suggested title:

> AI Danger Detector: estimating when AI use needs expert oversight

Suggested post outline:

1. The problem: AI advice is not equally safe across tasks.
2. The key question: who needs to review the output before acting?
3. The prototype rubric.
4. Examples: creative writing, flu symptoms, legal contract, financial trading, statistical analysis.
5. Limitations and request for critique.

Suggested channels:

- GitHub.
- LinkedIn.
- Personal site/blog.
- Relevant AI governance or responsible AI communities.

Do not publish to Product Hunt or run ads until the tool is usable, honest, and visually clear.

## Larger Product Path

Only pursue this if there is evidence of demand after the small publication.

Potential product shape:

- AI-use intake form for organizations.
- Maps use cases to oversight requirements.
- Exports a lightweight policy memo.
- Aligns each output to NIST AI RMF / EU AI Act / ISO 42001 language.
- Tracks reviewer roles and approvals.

This would require:

- Better source-backed rubric.
- Expert calibration.
- Clear legal disclaimers.
- Audit trail.
- Security/privacy review.
- Much more UX work.

## Kill Criteria

Stop the project if any of these are true:

- The desired claim is "this objectively detects AI danger."
- There is no appetite to maintain current framework alignment.
- No domain experts will review the rubric.
- The project expands into ML training without real calibration data.
- The main output remains just a numeric score with no action guidance.

## Best Next Step

Do a one-day cleanup and republish as a humble prototype. If that gets useful feedback, evolve it. If not, preserve the idea as a public note and move on.

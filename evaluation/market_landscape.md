# Current Landscape Notes

Date: 2026-05-23

## Why The Topic Is Still Relevant

AI risk management has become more structured since this project was started. Several current frameworks emphasize exactly the territory this project wants to reason about: risk classification, human oversight, traceability, monitoring, and accountability.

Key references:

- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework
- NIST Generative AI Profile, NIST AI 600-1: https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf
- EU AI Act overview: https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- EU AI Act FAQ/high-risk obligations: https://digital-strategy.ec.europa.eu/en/faqs/navigating-ai-act
- ISO/IEC 42001 AI management systems: https://www.iso.org/standard/42001
- MIT AI Risk Repository / Incident Tracker taxonomy: https://airisk.mit.edu/ai-incident-tracker/harm-taxonomy

## What Has Changed

The field has moved away from generic "AI is dangerous" messaging and toward operational governance:

- AI inventories.
- Use-case classification.
- Risk assessment and mitigation workflows.
- Human oversight assignment.
- Logging and traceability.
- Model/vendor documentation.
- Post-deployment monitoring.
- Incident response.
- Evidence for audit or compliance.

The EU AI Act explicitly classifies some AI use cases as high-risk and ties those systems to obligations such as risk assessment, data quality, documentation, traceability, transparency, human oversight, accuracy, cybersecurity, and robustness.

NIST AI RMF and the Generative AI Profile provide a strong voluntary framework in the US context. ISO/IEC 42001 adds an organizational management-system route for AI governance.

## Comparable Free Or Lightweight Tools

There are already free and commercial tools around AI risk and governance:

- VerifyWise offers free AI compliance/risk tools, including EU AI Act readiness and an AI risk calculator: https://verifywise.ai/tools
- Tavo positions itself as an AI risk copilot with a free GenAI risk assessment tool: https://www.gettavo.com/
- Rigour Labs maps deterministic AI-code quality gates to NIST AI RMF and presents a free/open-source local-first approach: https://rigour.run/nist-ai-rmf

This means AI Danger Detector should not compete as "another AI risk calculator" unless it has a sharper wedge.

## Best Differentiation

The strongest differentiated wedge is not compliance. It is user capability and oversight:

> Given this use case, what expertise must the human have before AI output can be trusted enough to act on?

That is more personal and practical than most governance tools. It could be useful for:

- general AI literacy training,
- organizational AI-use policies,
- educational demos,
- intake triage for AI use-case review,
- "what kind of reviewer do we need?" decisions.

## Positioning Risk

The name "AI Danger Detector" is catchy but slightly overpromises. It implies detection, objectivity, and perhaps model-level safety. The actual concept is closer to:

- AI Expertise Gate
- AI Oversight Estimator
- AI Use-Case Risk Triage
- Human-in-the-Loop Estimator
- Expertise Required Calculator

If keeping "AI Danger Detector," add a clear subtitle:

> A prototype for estimating human expertise and oversight needs in AI-assisted tasks.

## Publishing Recommendation

Free publishing is worthwhile if the artifact is honest and polished:

- GitHub repo refresh.
- Short README with limitations.
- One blog/LinkedIn post explaining the model.
- A small static demo or CLI.
- Invite critique from AI governance, healthcare, legal, education, and data science contacts.

Do not advertise as a working safety product. The current trust bar in AI governance is higher than this repo clears.

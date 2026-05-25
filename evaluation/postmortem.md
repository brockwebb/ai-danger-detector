# Project Postmortem: Why We Stopped Building AI Danger Detector as a Risk Score

Date: 2026-05-25

## Summary

AI Danger Detector began as an attempt to estimate when AI-assisted real-world tasks require human expertise, review, or restraint. The core concern remains valid: human-in-the-loop oversight only works when the human can detect, contest, and correct the relevant failure modes before harm occurs.

The project became less convincing as we tried to make that concern quantitative. Each additional layer made the evidence burden clearer. A general AI danger or confabulation risk score would require large, domain-specific experiments, reliable denominators, expert adjudication, benign comparison cases, model-version tracking, and holdout validation. Without that, the result would mostly be assumptions with precise-looking numbers.

The calibrated risk-score version should be considered not viable in its current form.

## Original Goal

The project set out to answer a practical governance question:

> When is AI use acceptable with ordinary review, when does it require trained or expert oversight, and when should the system refuse or avoid autonomous action?

The underlying motivation was autonomy gating. A more capable AI system should be able to decide not to follow a path when the risk is too high, to seek review, to reduce risk through controls, or to stop with a defensible assessment rather than proceed into a harmful action.

That goal is still important. The failure was not the concern. The failure was the attempt to turn it into a general, scalable, calibrated risk score without the empirical infrastructure such a score would require.

## What We Built

The refresh produced useful scaffolding:

- A whitepaper and rubric framing oversight as a function of harm, detectability, reversibility, verification burden, user expertise, and governance controls.
- A numerical framework using Monte Carlo simulation and Markov workflow modeling.
- An evidence schema, source registry, corpus loader, adjudication protocol, and synthetic example corpus.
- A provisional rubric scorer and ordinal evaluation runner.
- Model-comparison machinery with reference policy baselines.
- Beta-binomial calibration primitives and an evidence-to-observation bridge.

That machinery made the central problem clearer: a trustworthy score would not come from more code. It would come from costly evidence.

## Where The Approach Broke

### 1. The Evidence Does Not Scale Cleanly

Public stories, incident reports, lawsuits, and news articles can reveal failure modes, but they cannot estimate base rates. They overrepresent dramatic failures and underrepresent ordinary successful or corrected uses.

To estimate risk honestly, ADD would need denominators:

- how many similar AI-assisted tasks occurred,
- how often errors appeared,
- how often users detected them,
- how often review prevented harm,
- how often harms were reversible,
- how often comparable benign use occurred.

Most public evidence does not provide that.

### 2. Human Judgment Becomes The Bottleneck

The project depends on judgments about harm severity, detectability, reversibility, required expertise, and appropriate oversight. Those judgments are domain-sensitive and often contested.

Scaling them would require:

- trained reviewers,
- domain experts for high-stakes cases,
- inter-rater reliability checks,
- disagreement resolution,
- periodic re-review as models and workflows change.

If every useful case requires expert review, the process becomes too expensive and slow for a broad public tool. If cases are accepted without review, the dataset becomes too noisy to support calibration.

### 3. Numeric Scores Create False Authority

A risk score looks more objective than the evidence usually permits. In high-consequence contexts, a number may be operationally useless or dangerous.

For example, if a medical professional were told that an AI-assisted use has a 3% chance of contributing to death or severe harm, that number would not settle the decision. It would raise more questions:

- Where did the number come from?
- Is it valid for this patient, workflow, model version, and clinician?
- Does professional judgment mitigate the risk?
- Is any nonzero risk acceptable?
- Would the number create liability or false reassurance?

In low-stakes contexts the number may be unnecessary. In high-stakes contexts it may be indefensible. That is a bad shape for a general-purpose scoring tool.

### 4. Confabulation Risk Is Contextual

"Hallucination" or "confabulation" only matters in a practical sense when a false output is relied on and the reliance creates consequences. A benchmark can measure whether a model fabricates facts, but ADD wanted to measure whether a human workflow can safely absorb or catch that failure.

That requires evidence about:

- the task,
- the user,
- the stakes,
- the review process,
- detectability,
- reversibility,
- available controls.

Those are not model-only properties. They are properties of a situated human-machine workflow.

### 5. Benchmark Gaming Would Be A Real Risk

If ADD became a benchmark, model builders could optimize for its visible tests without improving real-world safety. This has happened with many benchmarks: systems learn the test rather than the underlying capability.

Avoiding this would require private holdouts, rotating cases, domain slices, adversarial examples, and outcome validation. That again turns the project into an ongoing institution, not a lightweight public framework.

## What Remains Useful

The project still surfaces a useful principle:

> The central question is not whether AI can produce an answer, but whether the human or workflow relying on it can detect and correct a harmful wrong answer before action.

That principle may be worth preserving in a shorter essay or position paper.

Useful concepts from ADD:

- Human-in-the-loop is weak if the human lacks domain competence.
- Detectability matters as much as model error rate.
- Reversibility matters because not all mistakes are recoverable.
- Governance controls should change whether AI can proceed autonomously.
- False reassurance is often more dangerous than conservative escalation.
- General numerical risk scoring is not credible without domain-specific evidence.

## What We Should Not Claim

ADD should not be presented as:

- a validated AI safety tool,
- a medical, legal, financial, or compliance decision aid,
- a calibrated risk model,
- a hallucination or confabulation benchmark,
- an autonomous approval system,
- evidence that a scenario is safe to use.

The repository contains exploratory machinery, but the machinery does not create empirical validity.

## Possible Salvage

The strongest salvage is a concise postmortem or position paper, not a product.

Possible title:

> Why We Did Not Build an AI Danger Detector

Possible thesis:

> General AI danger scoring is the wrong abstraction. The more defensible question is whether a specific human-machine workflow has enough expertise, evidence, and governance control to proceed without unacceptable risk.

This could become a useful short piece about the limits of risk scoring and the weakness of superficial HITL claims.

## Final Decision

The calibrated risk-score path should stop here.

The project was worth exploring because the failure is informative. It showed that a simple risk score would either rest on unvalidated assumptions or require an empirical program much larger than the tool itself.

Knowing when to stop is part of the work. In this case, stopping is the honest result.

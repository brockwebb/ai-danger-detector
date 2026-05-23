# ADD Whitepaper Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a credible public whitepaper foundation for AI Danger Detector, including the main paper, model rubric, numerical framework, validation agenda, and README reframing.

**Architecture:** This is a documentation-first evolution. The main whitepaper explains the governance argument; supporting docs separate the operational rubric, numerical machinery, and validation agenda so technical readers can inspect and extend the foundation without overloading the public narrative.

**Tech Stack:** Markdown documentation, existing Python prototype as background context only, Git for checkpoints.

---

## File Structure

Create or modify these files:

- Create: `docs/whitepaper.md`
  - Main public-facing paper for governance and responsible AI readers.
- Create: `docs/model-rubric.md`
  - Operational scoring rubric, decision bands, factor definitions, and interpretation guidance.
- Create: `docs/numerical-framework.md`
  - Technical foundation for Bayesian calibration, Monte Carlo uncertainty propagation, and Markov reliance workflow modeling.
- Create: `docs/validation-agenda.md`
  - Evidence, calibration, and validation roadmap.
- Modify: `README.md`
  - Reframe repository and link to the new docs.

Do not modify Python code in this plan. Existing prototype issues are acknowledged but out of scope for this documentation pass.

---

### Task 1: Create Main Whitepaper

**Files:**
- Create: `docs/whitepaper.md`

- [ ] **Step 1: Create the whitepaper file with title, abstract, and thesis**

Add this structure and opening substance:

```markdown
# AI Danger Detector: A Framework for Estimating Human Expertise Requirements in AI-Assisted Work

Date: 2026-05-23

## Abstract

AI risk is often discussed as a property of the model or the application domain. AI Danger Detector (ADD) starts from a different but complementary premise: for many AI-assisted tasks, risk also depends on whether the human user has enough expertise to recognize, contest, escalate, and correct wrong outputs before they become consequential actions.

ADD proposes an inspectable oversight triage framework for estimating when AI use can remain casual, when it requires trained or expert review, and when AI should not function as autonomous authority. The framework is provisional. Its weights, parameters, priors, and transition assumptions require calibration against documented cases, expert judgment, incident evidence, model evaluations, and future empirical work.

## Core Thesis

Risk-bearing AI use requires domain competence because harm detection is itself domain work.
```

- [ ] **Step 2: Add the problem section**

Include these claims in prose:

- AI advice is cheap, fluent, and increasingly available.
- Fluency can mask uncertainty, hallucination, omission, misplaced confidence, or context failure.
- Non-expert reliance is most dangerous when mistakes are consequential, hard to detect, or hard to reverse.
- The question is not only "is the model capable?" but "is the human-in-the-loop capable of noticing when the model is wrong?"

- [ ] **Step 3: Add the ADD framework section**

Define these factors in the whitepaper:

- Harm or loss potential.
- Domain complexity.
- Error uncertainty.
- Detectability.
- Reversibility.
- Verification burden.
- User expertise.
- Governance or oversight controls.

State that ADD estimates required oversight, not truth, correctness, or safety.

- [ ] **Step 4: Add the decision rule section**

Use this decision rule:

```markdown
## Decision Rule

ADD treats AI use as an oversight triage problem:

- **Low harm / low loss:** non-expert use may be acceptable with ordinary checks.
- **Moderate harm / moderate loss:** non-expert use should be bounded to ideation, drafting, exploration, or supervised assistance; trained or expert review is needed before action.
- **High or critical harm / loss:** AI use should be expert-led; AI output should not become autonomous authority.

The boundary is intentionally conservative. In moderate-or-higher harm contexts, false reassurance is more dangerous than conservative escalation.
```

- [ ] **Step 5: Add the reference model and numerical framework section**

Include the current formula as an early reference model:

```text
E_required = E_min
           + beta_h * harm^alpha_h
           + beta_c * complexity^alpha_c
           + beta_e * error_uncertainty^alpha_e
           + beta_i * (harm * complexity * error_uncertainty)^alpha_i
```

Then state:

- The formula is inspectable and falsifiable.
- It is not a final scientific model.
- The evolved framework combines Bayesian calibration, Monte Carlo uncertainty propagation, and Markov reliance workflow modeling.
- `docs/numerical-framework.md` contains the technical foundation.

- [ ] **Step 6: Add calibration, noise, and moving-target section**

Cover:

- Rate/lambda: how often meaningful errors or misuse events occur.
- Severity: how bad outcomes are when errors matter.
- Detectability: whether users can notice or contest wrong outputs.
- Reversibility: whether consequences can be corrected.
- Model drift and model improvement: current assumptions decay over time.
- Human overprediction of vivid negative events: anecdotes inform but should not dominate calibration.

Use this sentence:

```markdown
The goal is not perfect prediction. ADD is good enough when it reliably separates casual low-risk use from consequential use requiring trained or expert oversight.
```

- [ ] **Step 7: Add governance alignment section**

Mention these references as alignment targets:

- NIST AI Risk Management Framework.
- NIST Generative AI Profile.
- EU AI Act high-risk concepts.
- ISO/IEC 42001 AI management systems.
- AI incident taxonomy and incident repository work.

Do not claim compliance certification.

- [ ] **Step 8: Add limitations and builder agenda**

Limitations must include:

- No empirical calibration yet.
- Weights and priors are provisional.
- Transition probabilities need expert evaluation and empirical grounding.
- Error rates are difficult to estimate and change with model capabilities.
- This is not legal, medical, financial, safety, or compliance advice.

Builder agenda must include:

- calibrate priors and weights,
- test against documented cases,
- compare domain expert judgments,
- implement Monte Carlo and Markov simulation,
- build scenario classifiers only after the rubric is stable,
- version assumptions over time.

- [ ] **Step 9: Verify whitepaper content**

Run:

```bash
test -f docs/whitepaper.md
rg -n "Bayesian|Monte Carlo|Markov|good enough|false reassurance|not legal, medical, financial" docs/whitepaper.md
```

Expected:

- `test` exits 0.
- `rg` finds each major concept in `docs/whitepaper.md`.

- [ ] **Step 10: Commit the whitepaper**

```bash
git add docs/whitepaper.md
git commit -m "docs: add ADD whitepaper"
```

---

### Task 2: Create Model Rubric

**Files:**
- Create: `docs/model-rubric.md`

- [ ] **Step 1: Create rubric title and purpose**

Start with:

```markdown
# ADD Model Rubric

Date: 2026-05-23

## Purpose

This rubric defines the factors ADD uses to estimate the level of human expertise and oversight required for AI-assisted work. It is a provisional decision aid, not a validated safety instrument.
```

- [ ] **Step 2: Add factor table**

Add a table with these rows:

```markdown
| Factor | Meaning | Low | Moderate | High |
| --- | --- | --- | --- | --- |
| Harm/loss potential | Consequence if the AI output is wrong or misused | Annoyance, minor rework | Meaningful cost, delay, health, legal, educational, or financial consequence | Severe, irreversible, regulated, safety-critical, or rights-impacting consequence |
| Domain complexity | Expertise needed to understand the task and its failure modes | Everyday knowledge | Some trained judgment required | Specialized professional expertise required |
| Error uncertainty | Uncertainty about model accuracy in this task/context | Well-bounded and easy to check | Known limitations or uneven reliability | Sparse evidence, high hallucination risk, or context-sensitive failure modes |
| Detectability | Likelihood the user can notice a wrong output | Errors obvious to ordinary users | Errors require domain familiarity | Errors require expert knowledge or external validation |
| Reversibility | Ability to undo or correct harm | Easy to reverse | Costly or delayed reversal | Irreversible or only partially correctable |
| Verification burden | Effort required to check output before action | Quick independent check | Structured review or second source needed | Expert review, audit, testing, or formal validation needed |
| User expertise | User's ability to detect, contest, and mitigate errors | General user | Trained or domain-familiar user | Domain expert |
| Governance controls | Process controls around use | Informal use | Review or approval process | Required expert signoff, audit trail, monitoring, or prohibition |
```

- [ ] **Step 3: Add oversight bands**

Use these bands:

```markdown
| Band | Label | Meaning |
| --- | --- | --- |
| 1 | Casual/Exploratory | AI may be used for low-stakes exploration with ordinary user checks. |
| 2 | Assisted/Bounded | AI may support drafting, ideation, or preparation, but should not be final authority. |
| 3 | Trained Review Required | A trained or domain-familiar person should review before action. |
| 4 | Expert Review Required | A domain expert should review before action. |
| 5 | Expert-Led or No Autonomous Use | AI may assist an expert workflow but should not drive autonomous decisions. |
```

- [ ] **Step 4: Add scoring interpretation**

Explain:

- Scores are ordinal, not precise measurements.
- Weights represent assumptions to be tuned.
- Interaction terms matter because high harm plus low detectability is worse than either alone.
- Moderate harm with non-expert users should usually escalate to at least trained review.

- [ ] **Step 5: Add example scenarios**

Include at least these examples:

- Creative brainstorming for a personal story: casual/exploratory.
- Summarizing public meeting notes: assisted/bounded or trained review depending on publication consequences.
- Flu symptom advice for personal concern: trained review or professional consultation before action if symptoms are serious.
- Legal contract interpretation: expert review required.
- Financial trading recommendation: expert-led or no autonomous use.
- Statistical analysis for policy or business decision: trained or expert review depending on consequences.

- [ ] **Step 6: Verify rubric content**

Run:

```bash
test -f docs/model-rubric.md
rg -n "Harm/loss potential|Detectability|Reversibility|Expert Review Required|Financial trading" docs/model-rubric.md
```

Expected: all terms are found.

- [ ] **Step 7: Commit the rubric**

```bash
git add docs/model-rubric.md
git commit -m "docs: add ADD model rubric"
```

---

### Task 3: Create Numerical Framework

**Files:**
- Create: `docs/numerical-framework.md`

- [ ] **Step 1: Create numerical framework title and purpose**

Start with:

```markdown
# ADD Numerical Framework

Date: 2026-05-23

## Purpose

This document describes the numerical foundation for ADD: Bayesian calibration for uncertain assumptions, Monte Carlo simulation for propagating uncertainty, and Markov reliance workflow modeling for how AI-assisted decisions move through checking, escalation, action, correction, or harm.
```

- [ ] **Step 2: Add method overview**

Use this framing:

```markdown
ADD uses the three methods together:

1. **Bayesian calibration** describes how beliefs about error, severity, detectability, and reversibility should update as evidence arrives.
2. **Monte Carlo simulation** propagates uncertainty through the oversight model so outputs can be reported as ranges and threshold-crossing probabilities.
3. **Markov workflow modeling** represents AI reliance as a sequence of states rather than a one-time score.
```

- [ ] **Step 3: Add Bayesian calibration section**

Include these parameter examples:

```markdown
| Parameter | Meaning | Example prior form |
| --- | --- | --- |
| lambda_error | Rate of meaningful AI errors or misuse events | Beta distribution or bounded triangular distribution |
| severity | Magnitude of harm/loss if an error matters | Triangular, log-normal, or expert-elicited ordinal distribution |
| detectability | Probability a user detects a meaningful error before action | Beta distribution conditioned on expertise |
| reversibility | Probability harm can be corrected after action | Beta distribution conditioned on domain and action type |
| verification_burden | Effort or expertise required to check the output | Ordinal distribution mapped to oversight bands |
```

State that priors should be versioned by domain, model family, task type, and date.

- [ ] **Step 4: Add Monte Carlo simulation section**

Include this pseudocode:

```text
for simulation in 1..N:
    sample lambda_error from posterior or prior
    sample severity from posterior or prior
    sample detectability from posterior or prior
    sample reversibility from posterior or prior
    sample verification_burden from posterior or prior

    calculate oversight_score
    assign oversight_band
    record whether thresholds are crossed:
        trained_review_required
        expert_review_required
        expert_led_or_no_autonomous_use

report:
    median oversight_score
    uncertainty interval
    P(trained_review_required)
    P(expert_review_required)
    P(expert_led_or_no_autonomous_use)
```

- [ ] **Step 5: Add Markov state model**

Include this state table:

```markdown
| State | Label | Description |
| --- | --- | --- |
| S0 | AI use proposed | A user or workflow considers using AI for a task. |
| S1 | AI output generated | AI produces advice, content, analysis, recommendation, or code. |
| S2 | Non-expert acceptance | A non-expert accepts the output without meaningful verification. |
| S3 | Non-expert checking | A non-expert attempts to verify the output. |
| S4 | Expert review | A trained or expert reviewer evaluates the output. |
| S5 | Error detected/corrected | A meaningful error is caught, corrected, or escalated. |
| S6 | Action taken | The output materially informs a decision or action. |
| S7 | Harmless outcome | The action produces no meaningful harm or loss. |
| S8 | Realized harm/loss | The action produces meaningful harm, loss, rights impact, safety issue, or other adverse outcome. |
```

Explain that terminal states are `S7` and `S8`, while `S5` may loop back to `S1`, `S3`, `S4`, or exit the workflow.

- [ ] **Step 6: Add transition probability logic**

Include these transition rules:

- Higher user expertise increases `P(S3 -> S5)` and `P(S3 -> S4)`.
- Higher domain complexity decreases `P(S3 -> S5)` for non-experts.
- Strong governance increases `P(S1 -> S4)` and decreases `P(S1 -> S2)`.
- Higher detectability increases error correction before action.
- Lower reversibility increases expected loss once the workflow reaches `S6`.
- Model improvement can reduce `P(error at S1)`, but does not eliminate verification burden.

- [ ] **Step 7: Add example outputs section**

Include example outputs in prose:

```markdown
Under one set of provisional assumptions, ADD might report:

- 82% probability that trained review is required.
- 46% probability that expert review is required.
- 18% simulated probability of unverified action.
- 6% simulated probability of realized material harm/loss.

These numbers are illustrative, not calibrated. Their purpose is to show the reporting shape expected from a future implementation.
```

- [ ] **Step 8: Add limitations**

Mention:

- false precision,
- sparse evidence,
- expert disagreement,
- overfitting to dramatic incidents,
- changing model capability,
- transition probabilities requiring calibration,
- no claim of empirical validity yet.

- [ ] **Step 9: Verify numerical framework content**

Run:

```bash
test -f docs/numerical-framework.md
rg -n "Bayesian calibration|Monte Carlo simulation|Markov workflow|lambda_error|S8|transition probabilities" docs/numerical-framework.md
```

Expected: all terms are found.

- [ ] **Step 10: Commit the numerical framework**

```bash
git add docs/numerical-framework.md
git commit -m "docs: add ADD numerical framework"
```

---

### Task 4: Create Validation Agenda

**Files:**
- Create: `docs/validation-agenda.md`

- [ ] **Step 1: Create title and purpose**

Start with:

```markdown
# ADD Validation Agenda

Date: 2026-05-23

## Purpose

ADD is not validated by assertion. This document defines the evidence and evaluation work needed to tune, challenge, and improve the framework.
```

- [ ] **Step 2: Add evidence sources**

Include:

- documented incidents,
- domain expert panels,
- structured expert elicitation,
- benchmark and model evaluation results,
- deployment logs where ethically and legally available,
- user studies on error detection,
- case reviews from high-stakes domains.

- [ ] **Step 3: Add calibration risks**

Cover:

- human overprediction of vivid harms,
- underreporting of mundane failures,
- survivorship bias,
- domain skew,
- model-version drift,
- incentives to downplay risk,
- incentives to exaggerate risk.

- [ ] **Step 4: Add validation milestones**

Use these milestones:

```markdown
1. Face-validity review by domain experts.
2. Retrospective scoring of documented cases.
3. Sensitivity analysis over weights, priors, and transition probabilities.
4. Inter-rater reliability testing across reviewers.
5. Prospective use-case triage in low-risk organizational settings.
6. Periodic recalibration as models and evidence change.
```

- [ ] **Step 5: Add failure criteria**

State that ADD should be revised if:

- it repeatedly under-escalates moderate-or-higher harm cases,
- different reviewers cannot apply the rubric consistently,
- numerical outputs imply precision unsupported by evidence,
- Markov transition assumptions do not match observed workflow behavior,
- model improvements make old priors obsolete.

- [ ] **Step 6: Verify validation agenda content**

Run:

```bash
test -f docs/validation-agenda.md
rg -n "Face-validity|Inter-rater|overprediction|transition assumptions|recalibration" docs/validation-agenda.md
```

Expected: all terms are found.

- [ ] **Step 7: Commit the validation agenda**

```bash
git add docs/validation-agenda.md
git commit -m "docs: add ADD validation agenda"
```

---

### Task 5: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the opening positioning**

Reframe the README opening so it says:

```markdown
# AI Danger Detector (ADD)

AI Danger Detector is a prototype framework for estimating when AI-assisted work requires human expertise, trained review, expert oversight, or restraint.

The project began as a simple expertise-risk calculator. It is evolving into an inspectable oversight triage framework: part governance argument, part model rubric, and part numerical foundation for future simulation and validation.
```

- [ ] **Step 2: Add documentation links near the top**

Add:

```markdown
## Current Foundation

- [Whitepaper](docs/whitepaper.md)
- [Model Rubric](docs/model-rubric.md)
- [Numerical Framework](docs/numerical-framework.md)
- [Validation Agenda](docs/validation-agenda.md)
```

- [ ] **Step 3: Add status and limitations**

Add:

```markdown
## Status

This is not a validated safety, medical, legal, financial, or compliance tool. The current Python code is an early prototype. The whitepaper and supporting docs define the intended framework and the calibration work needed before stronger claims would be justified.
```

- [ ] **Step 4: Remove or soften overclaims**

Remove or rewrite claims that say:

- "detect when AI is safe",
- "scientific validation",
- "objectively classify",
- "simply describe your intended AI use in natural language" as an implemented feature.

Keep the project name, but make clear that "Danger Detector" is a shorthand for oversight triage, not an oracle.

- [ ] **Step 5: Verify README links and overclaim scan**

Run:

```bash
test -f README.md
rg -n "docs/whitepaper.md|docs/model-rubric.md|docs/numerical-framework.md|docs/validation-agenda.md" README.md
rg -n "detect when AI is safe|scientific validation|objectively classify|Simply describe your intended AI use" README.md || true
```

Expected:

- Documentation links are found.
- Overclaim scan returns no matches.

- [ ] **Step 6: Commit README update**

```bash
git add README.md
git commit -m "docs: reframe README around oversight triage"
```

---

### Task 6: Final Documentation QA

**Files:**
- Verify: `docs/whitepaper.md`
- Verify: `docs/model-rubric.md`
- Verify: `docs/numerical-framework.md`
- Verify: `docs/validation-agenda.md`
- Verify: `README.md`

- [ ] **Step 1: Confirm all expected files exist**

Run:

```bash
test -f docs/whitepaper.md
test -f docs/model-rubric.md
test -f docs/numerical-framework.md
test -f docs/validation-agenda.md
test -f README.md
```

Expected: exit 0.

- [ ] **Step 2: Confirm core terms are represented**

Run:

```bash
rg -n "Bayesian|Monte Carlo|Markov|expert review|non-expert|calibration|false reassurance" README.md docs/whitepaper.md docs/model-rubric.md docs/numerical-framework.md docs/validation-agenda.md
```

Expected: each term appears in at least one relevant document.

- [ ] **Step 3: Scan for unfinished markers**

Run:

```bash
rg -n "T[B]D|T[O]DO|F[IX]ME|draft marker|l[o]rem|fill[- ]in" README.md docs/whitepaper.md docs/model-rubric.md docs/numerical-framework.md docs/validation-agenda.md || true
```

Expected: no matches.

- [ ] **Step 4: Scan for strongest forbidden overclaims**

Run:

```bash
rg -n "scientifically validated|objective detector|guarantees safety|detects safety|safe to use" README.md docs/whitepaper.md docs/model-rubric.md docs/numerical-framework.md docs/validation-agenda.md || true
```

Expected: no matches.

- [ ] **Step 5: Review git status**

Run:

```bash
git status --short
```

Expected:

- no modified docs from this plan remain unstaged,
- pre-existing untracked files may still appear if unrelated.

- [ ] **Step 6: Create final QA commit if needed**

If Step 5 shows only documentation edits from QA, commit them:

```bash
git add README.md docs/whitepaper.md docs/model-rubric.md docs/numerical-framework.md docs/validation-agenda.md
git commit -m "docs: polish ADD foundation artifacts"
```

If there are no remaining edits, do not create an empty commit.

---

## Self-Review

Spec coverage:

- Whitepaper deliverable: Task 1.
- Model rubric deliverable: Task 2.
- Numerical framework deliverable: Task 3.
- Validation agenda deliverable: Task 4.
- README update: Task 5.
- Verification and overclaim checks: Task 6.

Unfinished-marker scan:

- The plan contains no unfinished content markers, dummy Latin text, or unresolved drafting instructions.

Scope check:

- The plan is documentation-only and intentionally excludes Python repair, CLI work, package cleanup, tests, and web UI work.

Type and naming consistency:

- Documentation filenames match the approved spec: `docs/whitepaper.md`, `docs/model-rubric.md`, `docs/numerical-framework.md`, `docs/validation-agenda.md`.

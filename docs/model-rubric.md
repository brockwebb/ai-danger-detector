# ADD Model Rubric

Date: 2026-05-23

## Purpose

This rubric defines the factors ADD uses to estimate the level of human expertise and oversight required for AI-assisted work. It is a provisional decision aid, not a validated safety instrument.

## Operational Factors

ADD treats risk as a practical oversight question: what kind of human review is needed before an AI-assisted output is used? The factors below are ordinal prompts for judgment. They should be applied conservatively when evidence is limited, users are non-experts, or consequences are hard to reverse.

| Factor | Meaning | Low | Moderate | High |
| --- | --- | --- | --- | --- |
| Harm/loss potential | Consequence if the AI output is wrong or misused | Annoyance or minor rework | Meaningful cost, delay, health, legal, educational, or financial consequence | Severe, irreversible, regulated, safety-critical, or rights-impacting consequence |
| Domain complexity | Expertise needed to understand the task and its failure modes | Everyday knowledge | Trained judgment required | Specialized professional expertise required |
| Error uncertainty | Uncertainty about model accuracy in this task/context | Well-bounded and easy to check | Known limitations or uneven reliability | Sparse evidence, high hallucination risk, or context-sensitive failure modes |
| Detectability | Likelihood the user can notice a wrong output | Errors obvious to ordinary users | Errors require domain familiarity | Errors require expert knowledge or external validation |
| Reversibility | Ability to undo or correct harm | Easy to reverse | Costly or delayed reversal | Irreversible or only partially correctable |
| Verification burden | Effort required to check output before action | Quick independent check | Structured review or second source needed | Expert review, audit, testing, or formal validation needed |
| User expertise | User's ability to detect, contest, and mitigate errors | General user | Trained or domain-familiar user | Domain expert |
| Governance controls | Process controls around use | Informal use | Review or approval process | Required expert signoff, audit trail, monitoring, or prohibition |

## Oversight Bands

The bands describe the minimum oversight posture suggested by the scenario. They are not claims that the AI output is correct, safe, lawful, or fit for use.

| Band | Label | Meaning |
| --- | --- | --- |
| 1 | Casual/Exploratory | AI may be used for low-stakes exploration with ordinary user checks. |
| 2 | Assisted/Bounded | AI may support drafting, ideation, or preparation, but should not be final authority. |
| 3 | Trained Review Required | A trained or domain-familiar person should review before action. |
| 4 | Expert Review Required | A domain expert should review before action. |
| 5 | Expert-Led or No Autonomous Use | AI may assist an expert workflow but should not drive autonomous decisions. |

## Scoring Interpretation

ADD scores are ordinal indicators, not precise measurements. A score of 4 is higher concern than a score of 3, but the distance between categories should not be treated as mathematically exact.

Weights represent assumptions about how much each factor should influence oversight. Those assumptions should be tuned over time using expert review, documented incidents, domain-specific cases, and empirical evaluation where available. Until that calibration exists, the model should be read as a structured reasoning aid rather than a validated predictor.

Interaction terms matter. High harm plus low detectability is worse than either factor alone because a damaging error that ordinary users cannot notice can move from suggestion to action without a meaningful checkpoint. Similar escalation may be warranted when high uncertainty combines with high verification burden, or when weak governance controls combine with non-expert users.

Moderate harm with non-expert users should usually escalate to at least Band 3, Trained Review Required. If the user cannot reliably detect, contest, or mitigate mistakes, ordinary review is not enough even when the task is not obviously high stakes.

## Example Scenarios

| Scenario | Typical band | Interpretation |
| --- | --- | --- |
| Creative brainstorming for a personal story | Band 1: Casual/Exploratory | The likely downside is annoyance, wasted time, or poor creative fit. Ordinary user judgment is usually enough. |
| Summarizing public meeting notes | Band 2: Assisted/Bounded, or Band 3: Trained Review Required | Internal preparation may be bounded use. Publication, official records, or decisions based on the summary raise consequences and review needs. |
| Flu symptom advice for personal concern | Band 3: Trained Review Required, or professional consultation before action if symptoms are serious | Basic information may help prepare questions, but worsening, severe, unusual, or high-risk symptoms should be handled through qualified medical advice. |
| Legal contract interpretation | Band 4: Expert Review Required | Legal meaning depends on jurisdiction, facts, drafting context, and consequences. AI may help organize questions, but a legal expert should review before action. |
| Financial trading recommendation | Band 5: Expert-Led or No Autonomous Use | Loss potential, uncertainty, and incentive-sensitive dynamics make autonomous AI-driven trading advice inappropriate for ordinary use. |
| Statistical analysis for policy or business decision | Band 3: Trained Review Required, or Band 4: Expert Review Required | Routine internal analysis may need trained review. Decisions with large financial, public, rights, employment, or safety consequences should escalate to expert review. |

## Practical Use

Use the rubric to document why a scenario was assigned an oversight band. Note the strongest escalation drivers, any assumptions about the user and context, and what verification would be needed before action. When in doubt, prefer a higher oversight band and make the reason explicit.

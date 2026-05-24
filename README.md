# AI Danger Detector (ADD)

AI Danger Detector is a prototype framework for estimating when AI-assisted work requires human expertise, trained review, expert oversight, or restraint.

The project began as a simple expertise-risk calculator. It is evolving into an inspectable oversight triage framework: part governance argument, part model rubric, and part numerical foundation for simulation and validation work.

## Current Foundation

- [Whitepaper](docs/whitepaper.md)
- [Model Rubric](docs/model-rubric.md)
- [Numerical Framework](docs/numerical-framework.md)
- [Validation Agenda](docs/validation-agenda.md)

## Status

This is not a validated safety, medical, legal, financial, or compliance tool. The active Python code is a first reference implementation of the numerical framework, not an empirically calibrated instrument. The whitepaper and supporting docs define the calibration work needed before stronger claims would be justified.

## What is AI Danger Detector?

AI Danger Detector, or ADD, is a working name for an oversight triage framework. "Danger Detector" is shorthand for identifying where AI-assisted work may need more human expertise or restraint; it is not an oracle and does not determine that a use case is safe.

The intended framework combines:

- A public-facing governance argument for why oversight should scale with expertise risk.
- A model rubric for translating scenario properties into review expectations.
- A numerical foundation for future simulation, calibration, and sensitivity analysis.
- A validation agenda describing the evidence needed before the framework can support stronger claims.

## Why It Matters

AI systems can be useful across everyday and high-stakes work, but the appropriate level of human involvement depends on context. A casual brainstorming task, a medical decision, a legal filing, and a financial recommendation should not receive the same oversight treatment.

ADD is being developed to make those distinctions inspectable. The current goal is to clarify what would need to be measured, reviewed, and validated before an AI-assisted task could be treated as lower risk.

## Current Implementation

The active Python implementation lives in `core_model/numerical_framework.py`. It provides a tested reference layer for sampling uncertain assumptions, evaluating a Markov reliance workflow, running Monte Carlo simulations, and summarizing oversight threshold probabilities.

Earlier exploratory calculators, scripts, and sensitivity experiments are preserved in `archive/exploratory-prototype/` for historical reference. They are not the active implementation.

## Project Structure

```
AI-Danger-Detector/
|
|-- docs/
|   |-- whitepaper.md               # Governance argument and framework overview
|   |-- model-rubric.md             # Scenario factors and oversight categories
|   |-- numerical-framework.md      # Mathematical foundation for modeling
|   `-- validation-agenda.md        # Calibration and validation work needed
|
|-- core_model/
|   |-- __init__.py                 # Active API exports
|   `-- numerical_framework.py      # Monte Carlo and Markov reference implementation
|
|-- tests/
|   `-- test_numerical_framework.py # Reference implementation tests
|
|-- evaluation/
|   |-- critical_review.md          # Project review
|   |-- market_landscape.md         # Publishing and landscape notes
|   |-- refresh_plan.md             # Recommended refresh path
|   `-- technical_health_check.md   # Codebase health review
|
`-- archive/
    `-- exploratory-prototype/      # Historical code, not active implementation
```

## Disclaimer

This project and all associated code and materials are intended solely for educational and research purposes. The contents of this repository reflect the authors' individual views and research and do not necessarily represent the official policies, positions, or views of any employer, organization, or institution associated with the authors.

## License

- MIT

## Contact

For questions or feedback, please reach out via [LinkedIn](https://www.linkedin.com/in/brockwebb/).

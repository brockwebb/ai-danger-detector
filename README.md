# AI Danger Detector (ADD)

AI Danger Detector is a prototype framework for estimating when AI-assisted work requires human expertise, trained review, expert oversight, or restraint.

The project began as a simple expertise-risk calculator. It is evolving into an inspectable oversight triage framework: part governance argument, part model rubric, and part numerical foundation for future simulation and validation.

## Current Foundation

- [Whitepaper](docs/whitepaper.md)
- [Model Rubric](docs/model-rubric.md)
- [Numerical Framework](docs/numerical-framework.md)
- [Validation Agenda](docs/validation-agenda.md)

## Status

This is not a validated safety, medical, legal, financial, or compliance tool. The current Python code is an early prototype. The whitepaper and supporting docs define the intended framework and the calibration work needed before stronger claims would be justified.

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

## Current Prototype

The existing Python code is an early calculator and exploration environment. It is useful for inspecting assumptions and experimenting with parameter sensitivity, but it should not be treated as a validated classifier or decision system.

The documentation now defines the intended direction of the project. Future code work should align the prototype with the whitepaper, rubric, numerical framework, and validation agenda.

## Project Structure

```
AI-Danger-Detector/
|
|-- docs/
|   |-- whitepaper.md               # Governance argument and framework overview
|   |-- model-rubric.md             # Scenario factors and oversight categories
|   |-- numerical-framework.md      # Mathematical foundation for future modeling
|   `-- validation-agenda.md        # Calibration and validation work needed
|
|-- core_model/
|   |-- model_definition.py         # Early equations, parameters, and prototype logic
|   `-- parameters.json             # Domain presets and parameter ranges
|
|-- model_validation/               # Exploratory model behavior checks
|   |-- parameter_sensitivity.py    # Parameter sweeps and sensitivity exploration
|   |-- domain_validation.py        # Domain profile checks
|   `-- edge_case_testing.py        # Boundary and extreme-case exploration
|
|-- data_generation/
|   |-- calculator.py               # Single-point expertise calculator
|   |-- batch_generator.py          # Scenario batch generator
|   `-- generate_large_dataset.py   # Synthetic datasets for experimentation
|
|-- visualization/
|   |-- plotting.py                 # Core visualization functions
|   `-- domain_comparisons.py       # Domain comparison visualizations
|
|-- machine_learning/
|   |-- preprocessing.py            # Data preparation experiments
|   |-- model_training.py           # Predictive model experiments
|   |-- validation.py               # ML evaluation experiments
|   `-- interpretability.py         # Feature importance and interpretation experiments
|
`-- applications/
    |-- sensitivity_tool.py         # Parameter sensitivity analyzer
    `-- interactive_assessment.py   # Early interactive assessment prototype
```

## Disclaimer

This project and all associated code and materials are intended solely for educational and research purposes. The contents of this repository reflect the authors' individual views and research and do not necessarily represent the official policies, positions, or views of any employer, organization, or institution associated with the authors.

## License

- MIT

## Contact

For questions or feedback, please reach out via [LinkedIn](https://www.linkedin.com/in/brockwebb/).

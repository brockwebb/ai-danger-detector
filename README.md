# AI Danger Detector (ADD)

**AI Danger Detector**... because it **"ADDs"** value by helping you quickly detect when AI is safe—and when it isn't.

Quickly assess how much expertise you need when using AI, so you can enjoy its benefits safely.

## What is AI Danger Detector?

AI Danger Detector is a practical, science-based tool designed to quantify and communicate the level of human expertise required to safely use AI in different contexts. It provides clear, actionable guidance—not preachy warnings—to help users understand the risks involved when interacting with AI systems.

## Why is it Important?

AI systems offer significant benefits, but their risks vary widely based on how they're used. Without a clear, objective way to assess these risks, people either underuse powerful tools or, worse, underestimate their dangers. AI Danger Detector bridges this gap by clearly defining the necessary expertise for safe usage, ensuring you can make informed decisions.

## What Difference Does It Make?

AI Danger Detector enables users—both experts and non-experts—to quickly understand:

- **The level of expertise required** for specific AI use cases.
- **The potential risks involved**, clearly and objectively assessed.
- **Practical insights** that let you decide how and when to safely use AI.

Whether you're using AI for everyday advice (like assessing flu symptoms) or engaging in high-stakes applications (such as medical diagnostics or stock trading), AI Danger Detector helps you understand the minimum expertise level necessary.

- Backed by rigorous analytical models and scientific validation.
- Provides friendly and engaging guidance, avoiding moralistic finger-wagging.
- You can always choose to accept the risk, but you'll do so informed—no surprises!

## Expected Use Scenario

Simply describe your intended AI use in natural language—for example:
- *"I'm going to use AI for advice on flu symptoms."*
- *"I'm using AI for stock trading recommendations."*

AI Danger Detector will objectively classify the scenario into clear risk/expertise categories (low, medium, high, critical), helping you understand exactly what level of human expertise is required.

## Project Structure

```
AI-Danger-Detector/
│
├── core_model/
│   ├── model_definition.py         # Equations, core parameters, and logic clearly defined
│   └── parameters.json             # Domain-specific presets and parameter ranges
│
├── model_validation/               # Early validation of theoretical model behavior
│   ├── parameter_sensitivity.py    # Test model behavior with parameter sweeps
│   ├── domain_validation.py        # Validate domain profiles (medical, legal, statistical, etc.)
│   └── edge_case_testing.py        # Evaluate behavior at parameter boundaries and extremes
│
├── data_generation/
│   ├── calculator.py               # Single-point expertise calculator
│   ├── batch_generator.py          # Scenario batch generator
│   └── generate_large_dataset.py   # Large synthetic datasets for ML
│
├── visualization/
│   ├── plotting.py                 # Core visualization functions (2D, 3D, sensitivity analyses)
│   └── domain_comparisons.py       # Domain-specific visualizations
│
├── machine_learning/
│   ├── preprocessing.py            # Data preparation for ML
│   ├── model_training.py           # Training predictive ML models
│   ├── validation.py               # ML model performance evaluation
│   └── interpretability.py         # Feature importance and model interpretation
│
└── applications/
    ├── sensitivity_tool.py         # Parameter sensitivity analyzer
    └── interactive_assessment.py   # User-friendly interactive risk/expertise assessment tool
```

## Disclaimer

This project and all associated code and materials are intended solely for educational and research purposes. The contents of this repository reflect the authors' individual views and research and do not necessarily represent the official policies, positions, or views of any employer, organization, or institution associated with the authors.

## License

- MIT 

## Contact

For questions or feedback, please reach out via LinkedIn (u: /brockwebb).

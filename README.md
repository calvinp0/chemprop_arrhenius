# Learning Rates: Arrhenius Modeling for Hydrogen Abstraction Reactions

This repository contains the implementation accompanying the article:

Learning Rates: Predicting Reaction Rate Coefficients for Hydrogen Abstraction Reactions

It provides a Chemprop-based reaction modeling framework extended with an Arrhenius-parameter prediction workflow tailored for hydrogen abstraction kinetics.

<details open>
<summary><strong>Overview</strong></summary>

The core objective of this project is to predict full Arrhenius parameter triplets

$k(T) = A T^n \exp\!\left(-\frac{E_a}{RT}\right)$

for forward and reverse hydrogen abstraction reactions using graph neural networks.

This repository introduces:

- Arrhenius-specific model heads and loss formulations
- Reaction-aware data preprocessing and role labeling
- Donor-acceptor ordered and invariant training modes
- Forward/reverse thermodynamic consistency handling
- Temperature-grid evaluation of log10 k(T)
- Custom cross-validation and splitting strategies for reaction kinetics

All Arrhenius-specific logic is isolated in:

`arrhenius/`

The upstream Chemprop framework is used as the message-passing backbone and training infrastructure.

</details>

<details>
<summary><strong>Repository Structure</strong></summary>

```text
arrhenius/
    preprocessing/      # Reaction-specific data preparation
    training/           # Training loops and HPO workflow
    models/             # Arrhenius heads and layers
    evaluation/         # Temperature-grid evaluation
    README.md           # Entry point for Arrhenius workflow
```

If you are interested specifically in the kinetics workflow, begin with:

- `arrhenius/README.md`
- `arrhenius/preprocessing/README.md`
- `arrhenius/training/hpo/WORKFLOW.md`

</details>

<details>
<summary><strong>Relationship to Chemprop</strong></summary>

This project builds on Chemprop v2, which provides:

- Directed message passing neural network backbone
- Molecular/reaction featurization
- Training infrastructure and CLI
- Lightning-based training utilities

The Arrhenius extension adds:

- Parameterized Arrhenius output heads
- Kinetics-specific objective functions
- Reaction-level splitters
- Hydrogen abstraction-specific preprocessing
- Thermodynamic coupling of forward/reverse directions

The Chemprop core remains structurally recognizable. All domain-specific logic is confined to `arrhenius/`.

Chemprop documentation:
https://chemprop.readthedocs.io/en/main/

</details>

<details>
<summary><strong>Scientific Scope</strong></summary>

This repository supports:

- Prediction of forward and reverse Arrhenius parameters
- Evaluation of log10 k(T) over 300-3000 K
- Analysis by donor-acceptor reactive site types
- Uncertainty-aware modeling (when enabled)
- Cross-validation protocols appropriate for reaction kinetics

The workflow is designed specifically for hydrogen abstraction reactions and is not a general-purpose property prediction package.

</details>

<details>
<summary><strong>Installation</strong></summary>

Installation follows standard Chemprop v2 setup procedures.

After installation, the Arrhenius workflow can be accessed via the modules under `arrhenius/`.

</details>

<details>
<summary><strong>Citation</strong></summary>

If this repository contributes to your work, please cite:

Learning Rates: Predicting Reaction Rate Coefficients for Hydrogen Abstraction Reactions

and the relevant Chemprop references:

Yang et al., Chemprop v2: An Efficient, Modular Machine Learning Package for Chemical Property Prediction, J. Chem. Inf. Model. (2025).

Yang et al., Chemprop: A Machine Learning Package for Chemical Property Prediction, J. Chem. Inf. Model. (2023).

</details>

<details>
<summary><strong>License</strong></summary>

Chemprop is distributed under the MIT License.
Arrhenius-specific extensions follow the same licensing structure.

</details>

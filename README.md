# Interpretable Iris Classification

**Python · pandas · NumPy · Matplotlib · scikit-learn**

An end-to-end exploration of how simple, inspectable rules compare with a learned decision tree on the Iris dataset. The workflow moves from descriptive statistics and visualization to species prediction, then examines what the evaluation actually supports.

[Explore the notebook](Demo_IrisSpeciesClassification.ipynb) · [Read the classifier](irisClassification.py) · [Watch the original demo](https://youtu.be/23Iec5DPY7k) · [Original project report](docs/original-report.md)

## Why this project matters

For technical product work, model selection is also a decision about explainability, evidence, and operational expectations. This project makes the path from observed feature patterns to classification rules visible. It also illustrates why a strong accuracy number needs a clear evaluation protocol before it becomes a product claim.

This is a small educational experiment, not a deployed prediction service or a production ML system.

## What it does

1. Loads 150 flower observations with four measurements and three balanced species classes.
2. Normalizes species labels and explores their distributions through grouped statistics, scatter plots, and histograms.
3. Applies the original team's fixed rules using petal width and petal length.
4. Fits a scikit-learn decision tree using all four measurement features.
5. Reports the manual classifier's full-dataset score and a separate learned model's training/test scores.

The identifier column is excluded from the learned model's features. Measurements are in centimeters.

## Design and code map

- `irisClassification.py`: `IrisClassifier`, containing the manual rule tree, grouped summaries, visualizations, and species-specific min/max reporting.
- `Demo_IrisSpeciesClassification.ipynb`: original exploratory narrative, plots, predictions, and unseeded model comparison. Original code cells and saved outputs are retained; the interpretation of the saved metrics has been corrected.
- `evaluate.py`: added deterministic evaluation with `random_state=42`, stratified sampling, a 60/40 split, and `max_depth=3`.
- `Iris.csv`: 150 observations, including an ID, four numeric measurements, and the species label.
- `tests/test_classifier.py`: regression checks covering input handling, original decision boundaries, dataset predictions, and plotting compatibility.

The manual rules are unchanged: width below 0.8 predicts setosa; width above 1.75 predicts virginica; otherwise length at most 4.95 predicts versicolor, with virginica as the final branch.

## Run locally

The maintenance version was validated with Python 3.12 and the versions in `requirements.txt`.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python evaluate.py
python -m unittest discover -s tests -v
```

To explore the notebook, install a Jupyter frontend of your choice, install `requirements-notebook.txt`, and select this Python environment. Open the notebook from the repository root so it can find `Iris.csv` and the local module.

Minimal use:

```python
import pandas as pd
from irisClassification import IrisClassifier

data = pd.read_csv("Iris.csv")
data["Species"] = data["Species"].str.removeprefix("Iris-")
classifier = IrisClassifier(data)
print(classifier.decision_tree(0.2, 1.4))  # setosa; width, then length
```

## Results and how to interpret them

The reproducible `evaluate.py` run produced:

- **Manual rules:** 146/150 correct, or **97.33%**, on the full dataset used to select the rules.
- **Learned tree:** **97.78%** training accuracy on 90 observations and **98.33%** held-out accuracy on 60 observations, for the fixed stratified split with seed 42.

The original notebook's saved, unseeded run instead reports **93.33%** test accuracy. These are different runs with different splits; neither result should replace the other without explaining the protocol.

The manual score is in-sample. It is not an independent generalization estimate and does not justify claiming that the manual model outperforms the learned model. One small test split also does not establish an absence of overfitting. Cross-validation, per-class errors, and genuinely unseen data would be needed for a stronger evaluation.

## Validation and limitations

- All **11 regression checks passed** in the maintenance environment.
- The original manual thresholds and 146/150 prediction result are preserved.
- Grouped-statistic validation and compatibility with modern pandas were corrected; details are in [CHANGELOG.md](CHANGELOG.md).
- The class retains the supplied dataframe by reference. The notebook performs label normalization before use.
- The original measurement method accepts Python `float` and `int` inputs; it is not a comprehensive physical-measurement validator.
- The original histogram method exits on an invalid feature. That behavior is retained and should be considered before embedding it in a service.
- No serving API, monitoring, external validation, or production deployment is claimed.

## Authors and provenance

Original contributors: **Alexis Pendleton, Andrew Dorado, Neira Ibrahimovic, and Lillian Gabrelian**. The original report credits lecture and discussion material by **Dr. Harlin Lee**. This portfolio presentation preserves those credits and does not assign the team's complete implementation to a single author.

Data: the Iris dataset, credited in the original report to Edgar Anderson and R. A. Fisher, distributed through the [UCI dataset page](https://archive.ics.uci.edu/ml/datasets/Iris) and the [Kaggle copy](https://www.kaggle.com/datasets/uciml/iris).

Documentation, targeted maintenance fixes, and regression checks were prepared with AI assistance. No new license is asserted; repository availability does not by itself grant redistribution rights. The original report and Git history preserve the project's provenance.

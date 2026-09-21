"""Repeatable evaluation alongside the original exploratory notebook.

The manual rules were selected from the complete dataset. Their full-dataset
score is descriptive and cannot be treated as an independent test result.
"""
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from irisClassification import IrisClassifier


def evaluate():
    data = pd.read_csv(Path(__file__).with_name("Iris.csv"))
    data["Species"] = data["Species"].str.removeprefix("Iris-")
    features = data.drop(columns=["Species", "Id"])
    labels = data["Species"]
    # Stratification preserves class representation; the seed makes the split
    # and learned tree reproducible for this environment and dependency set.
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.4, random_state=42, stratify=labels
    )
    learned = DecisionTreeClassifier(max_depth=3, random_state=42)
    learned.fit(x_train, y_train)
    manual = IrisClassifier(data)
    guesses = data.apply(
        lambda row: manual.decision_tree(row.PetalWidthCm, row.PetalLengthCm), axis=1
    )
    return {
        "samples": len(data), "train_samples": len(x_train), "test_samples": len(x_test),
        "seed": 42, "max_depth": 3,
        "manual_full_dataset_accuracy": float((guesses == labels).mean()),
        "learned_train_accuracy": float(learned.score(x_train, y_train)),
        "learned_test_accuracy": float(learned.score(x_test, y_test)),
        "caveat": "Manual thresholds were chosen using the full dataset; scores are not a fair held-out comparison."
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))

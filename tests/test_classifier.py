"""Regression checks for validation, predictions, and plotting compatibility."""
import contextlib
import io
import unittest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from irisClassification import IrisClassifier


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv("Iris.csv")
        self.data["Species"] = self.data["Species"].str.removeprefix("Iris-")
        self.classifier = IrisClassifier(self.data)

    def tearDown(self):
        plt.close("all")

    def test_reference_predictions(self):
        for width, length, expected in [(0.2, 1.4, "setosa"), (1.3, 4.0, "versicolor"), (1.9, 5.0, "virginica")]:
            self.assertEqual(self.classifier.decision_tree(width, length), expected)

    def test_threshold_boundaries_preserved(self):
        self.assertEqual(self.classifier.decision_tree(0.8, 4.95), "versicolor")
        self.assertEqual(self.classifier.decision_tree(1.75, 4.96), "virginica")

    def test_full_dataset_reference_score(self):
        predictions = self.data.apply(lambda r: self.classifier.decision_tree(r.PetalWidthCm, r.PetalLengthCm), axis=1)
        self.assertEqual(int((predictions == self.data.Species).sum()), 146)

    def test_invalid_group_rejected(self):
        with self.assertRaises(NameError):
            self.classifier.summary_table("not_a_column", "PetalWidthCm")

    def test_invalid_value_rejected(self):
        with self.assertRaises(NameError):
            self.classifier.summary_table("Species", "not_a_column")

    def test_list_arguments(self):
        summary = self.classifier.summary_table(["Species"], ["PetalWidthCm", "PetalLengthCm"])
        self.assertEqual(summary.shape, (3, 4))

    def test_empty_columns_rejected(self):
        with self.assertRaises(NameError):
            self.classifier.summary_table([], ["PetalWidthCm"])

    def test_scatter_uses_species_labels(self):
        self.classifier.visualize_scatter("PetalLengthCm", "PetalWidthCm")
        self.assertEqual(plt.gca().get_legend_handles_labels()[1], ["setosa", "versicolor", "virginica"])

    def test_constructor_rejects_non_dataframe(self):
        with self.assertRaises(TypeError):
            IrisClassifier("Iris.csv")

    def test_invalid_measurement_type(self):
        with self.assertRaises(TypeError):
            self.classifier.decision_tree("0.2", 1.4)

    def test_min_max_output(self):
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.classifier.max_min("setosa", "PetalWidthCm")
        self.assertIn("0.1", stream.getvalue())
        self.assertIn("0.6", stream.getvalue())


if __name__ == "__main__":
    unittest.main()

"""Drift monitoring for the heart disease model using Evidently.

Compares training data (reference) against simulated production data,
reports which features drifted, saves an HTML report, and exits non-zero
if drift exceeds a configurable threshold.
"""

import argparse
import os
import numpy as np

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.data_preprocessing import load_data, remove_duplicates


def make_production_data(reference):
    """Simulate production data by shifting a few features, imitating an
    incoming population that is older with higher cholesterol and lower
    peak heart rate than the training population.
    """
    prod = reference.copy()
    rng = np.random.default_rng(42)
    prod["age"] = prod["age"] + rng.integers(8, 15, size=len(prod))
    prod["chol"] = prod["chol"] * 1.2
    prod["thalach"] = prod["thalach"] - rng.integers(10, 20, size=len(prod))
    return prod


def main(reference_path, threshold):
    # reference = the data the model was trained on, features only
    reference = remove_duplicates(load_data(reference_path)).drop(columns=["target"])
    production = make_production_data(reference)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=production)

    os.makedirs("reports", exist_ok=True)
    report_path = "reports/drift_report.html"
    report.save_html(report_path)

    result = report.as_dict()
    summary = result["metrics"][0]["result"]
    drift_share = summary["share_of_drifted_columns"]
    n_drifted = summary["number_of_drifted_columns"]

    by_column = result["metrics"][1]["result"]["drift_by_columns"]
    drifted = [c for c, info in by_column.items() if info["drift_detected"]]

    print(f"Overall drift share: {drift_share:.2f}")
    print(f"Drifted features ({n_drifted}): {drifted}")
    print(f"HTML report saved to {report_path}")

    if drift_share > threshold:
        print(f"DRIFT ALERT: share {drift_share:.2f} exceeds threshold {threshold}")
        raise SystemExit(1)

    print(f"OK: drift share {drift_share:.2f} within threshold {threshold}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/heart.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args.reference, args.threshold)
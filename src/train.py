"""Train a heart disease classifier and log the run to MLflow.

Exits non-zero if the model misses the minimum thresholds in the config,
which makes this script usable as a gate in the CI pipeline.
"""

import argparse
import sys
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.data_preprocessing import (
    load_data,
    validate_columns,
    handle_missing_values,
    remove_duplicates,
    split_features_target,
    build_preprocessor,
)
from src.evaluate import compute_metrics


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(model_config):
    model_type = model_config["type"]
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=model_config["n_estimators"],
            max_depth=model_config["max_depth"],
            random_state=model_config["random_state"],
        )
    elif model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=model_config["random_state"])
    elif model_type == "svm":
        return SVC(probability=True, random_state=model_config["random_state"])
    raise ValueError(f"Unknown model type: {model_type}")


def main(config_path):
    config = load_config(config_path)

    df = load_data(config["data"]["raw_path"])
    all_features = config["features"]["numeric"] + config["features"]["categorical"]
    validate_columns(df, all_features + [config["data"]["target_column"]])
    df = remove_duplicates(df)
    df = handle_missing_values(df)

    X, y = split_features_target(df, config["data"]["target_column"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    preprocessor = build_preprocessor(
        config["features"]["numeric"],
        config["features"]["categorical"],
    )
    model = build_model(config["model"])
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    mlflow.set_experiment("heart-disease-prediction")
    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)

        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_param("n_estimators", config["model"].get("n_estimators"))
        mlflow.log_param("max_depth", config["model"].get("max_depth"))
        mlflow.log_param("test_size", config["data"]["test_size"])
        mlflow.log_param("data_version", "raw_v1")

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        mlflow.sklearn.log_model(pipeline, "model")

        print("Run complete. Metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.3f}")

    # Performance gate: fail loudly if the model is below threshold
    min_accuracy = config["evaluation"]["min_accuracy"]
    min_recall = config["evaluation"]["min_recall"]
    if metrics["accuracy"] < min_accuracy or metrics["recall"] < min_recall:
        print(
            f"FAIL: accuracy {metrics['accuracy']:.3f} (min {min_accuracy}) "
            f"or recall {metrics['recall']:.3f} (min {min_recall}) below threshold."
        )
        sys.exit(1)

    print("PASS: model meets minimum thresholds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
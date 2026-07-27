"""Query all MLflow runs and report the best one by the primary metric."""

import yaml
import mlflow


def load_primary_metric(config_path="configs/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["evaluation"]["primary_metric"]


def main():
    primary_metric = load_primary_metric()
    metric_column = f"metrics.{primary_metric}"

    runs = mlflow.search_runs(experiment_names=["heart-disease-prediction"])
    if runs.empty:
        print("No runs found. Run training first.")
        return

    runs = runs.sort_values(by=metric_column, ascending=False)

    print(f"Total runs: {len(runs)}")
    print(f"Ranking by {primary_metric}:\n")
    cols = ["run_id", "params.model_type", metric_column,
            "metrics.accuracy", "metrics.precision", "metrics.f1"]
    cols = [c for c in cols if c in runs.columns]
    print(runs[cols].to_string(index=False))

    best = runs.iloc[0]
    print(f"\nBest run: {best['run_id']}")
    print(f"  model_type: {best.get('params.model_type')}")
    print(f"  {primary_metric}: {best[metric_column]:.3f}")


if __name__ == "__main__":
    main()

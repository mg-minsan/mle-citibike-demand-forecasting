"""Held-out permutation importance in original pipeline-input space."""

import mlflow
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error


def measure_feature_importance(model, features, labels, seed, max_rows=1000, n_repeats=3):
    # Never refit here: use the train+validation model on its held-out test month.
    if len(features) == 0:
        raise ValueError("feature importance requires non-empty test data")
    rows = np.random.default_rng(seed).choice(len(features), min(max_rows, len(features)), replace=False)
    sample = features.iloc[rows].copy()
    target = np.asarray(labels)[rows]
    result = permutation_importance(
        model, sample, target, scoring="neg_mean_absolute_error",
        n_repeats=n_repeats, random_state=seed, n_jobs=1,
    )
    table = pd.DataFrame({
        "feature": sample.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False, kind="stable")
    return {
        "method": "permutation",
        "units": "increase_in_mae",
        "model_scope": "train_validation_refit_before_final_refit",
        "sample_rows": len(sample),
        "test_rows": len(features),
        "n_repeats": n_repeats,
        "seed": seed,
        "sample_baseline_mae": float(mean_absolute_error(target, model.predict(sample))),
        "features": table.to_dict(orient="records"),
    }


def log_feature_importance(report):
    """Attach the report, ranked table, chart and comparison metrics to the active run."""
    from matplotlib.figure import Figure

    table = pd.DataFrame(report["features"])
    mlflow.log_dict(report, "feature_importance/importance.json")
    mlflow.log_text(table.to_csv(index=False), "feature_importance/importance.csv")
    mlflow.log_metrics({
        "feature_importance/" + row["feature"]: row["importance_mean"]
        for row in report["features"]
    })
    figure = Figure(figsize=(10, max(4, len(table) * 0.3)), layout="constrained")
    axis = figure.subplots()
    axis.barh(table["feature"], table["importance_mean"], xerr=table["importance_std"])
    axis.invert_yaxis()
    axis.axvline(0, color="grey", linewidth=0.8)
    axis.set_xlabel("Increase in held-out MAE when shuffled (mean ± std)")
    axis.set_title("Permutation importance — pre-final-refit model")
    try:
        mlflow.log_figure(figure, "feature_importance/importance.png")
    finally:
        figure.clear()

"""Evaluate saved station/day forecasts after their actual labels arrive."""
import numpy as np
import pandas as pd


def daily_performance(predictions, labels):
    keys = ["snapshot_date", "station_id"]
    predictions, labels = predictions.copy(), labels.copy()
    for table in (predictions, labels):
        table["snapshot_date"] = pd.to_datetime(table["snapshot_date"], errors="raise").dt.normalize()
        if table[keys].isna().any().any() or table.duplicated(keys).any():
            raise ValueError("Expected unique, non-null station/day keys")
    joined = predictions.merge(labels[keys + ["label"]], on=keys, how="outer", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Prediction and label station/day coverage differs; do not score partial days")
    if joined.empty:
        raise ValueError("No matched predictions and labels")
    columns = ["label", "model_prediction", "baseline_prediction"]
    values = joined[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Actuals and forecasts must be finite and nonnegative")
    if joined["model_version"].isna().any() or joined.groupby("snapshot_date")["model_version"].nunique().gt(1).any():
        raise ValueError("Expected exactly one model version per day")
    joined["model_absolute_error"] = (joined["model_prediction"] - joined["label"]).abs()
    joined["baseline_absolute_error"] = (joined["baseline_prediction"] - joined["label"]).abs()
    daily = joined.groupby("snapshot_date", as_index=False).agg(
        station_count=("station_id", "size"), actual_demand=("label", "sum"),
        model_absolute_error=("model_absolute_error", "sum"),
        baseline_absolute_error=("baseline_absolute_error", "sum"),
        model_version=("model_version", "first"),
    ).sort_values("snapshot_date").reset_index(drop=True)
    daily["model_mae"] = daily["model_absolute_error"] / daily["station_count"]
    daily["baseline_mae"] = daily["baseline_absolute_error"] / daily["station_count"]
    denominator = daily["actual_demand"].where(daily["actual_demand"] > 0)
    daily["model_wmape_pct"] = 100 * daily["model_absolute_error"] / denominator
    daily["baseline_wmape_pct"] = 100 * daily["baseline_absolute_error"] / denominator
    daily["improvement_pct"] = 100 * (1 - daily["model_mae"] / daily["baseline_mae"].where(daily["baseline_mae"] > 0))
    daily["model_changed"] = daily["model_version"].ne(daily["model_version"].shift())
    daily.loc[0, "model_changed"] = False
    return daily

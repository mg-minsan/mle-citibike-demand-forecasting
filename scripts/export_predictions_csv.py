"""Export a daily prediction partition as the business station forecast CSV."""

import argparse
import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def export_predictions(snapshotdate, predictions_dir="datamart/gold/predictions",
                       stations_file="data/stations.csv",
                       output_dir="datamart/business/predictions"):
    snapshotdate = date.fromisoformat(snapshotdate).isoformat()
    filename = "predictions_" + snapshotdate.replace("-", "_")
    predictions = pd.read_parquet(
        Path(predictions_dir) / (filename + ".parquet"),
        columns=["snapshot_date", "model_prediction", "station_id"],
    )
    stations = pd.read_csv(stations_file, dtype="string", keep_default_na=False,
                           usecols=["station_id", "station_name"])
    if predictions.empty:
        raise ValueError("No predictions for " + snapshotdate)
    if predictions.isna().any().any():
        raise ValueError("Missing values in predictions")
    predictions["station_id"] = predictions["station_id"].astype("string").str.strip()
    stations["station_id"] = stations["station_id"].str.strip()
    stations["station_name"] = stations["station_name"].str.strip()
    if predictions["station_id"].eq("").any() or predictions["station_id"].duplicated().any():
        raise ValueError("Expected one prediction per nonempty station ID")
    if stations["station_id"].eq("").any() or stations["station_id"].duplicated().any():
        raise ValueError("Station lookup must have unique, nonempty station IDs")
    predictions["date"] = pd.to_datetime(predictions["snapshot_date"], errors="raise").dt.strftime("%Y-%m-%d")
    if not predictions["date"].eq(snapshotdate).all():
        raise ValueError("Prediction dates do not match " + snapshotdate)
    values = pd.to_numeric(predictions["model_prediction"], errors="raise")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Predictions must be finite and nonnegative")
    # Business exports use whole trips, with .5 rounded up.
    # The source Parquet retains the original model predictions for evaluation.
    predictions["model_prediction"] = np.floor(values + 0.5).map(int)
    result = predictions.merge(stations, on="station_id", how="left", validate="one_to_one")
    missing = result["station_name"].isna() | result["station_name"].eq("")
    if missing.any():
        raise ValueError("Missing station names for IDs: " + ", ".join(result.loc[missing, "station_id"]))
    result = result[["station_id", "station_name", "model_prediction", "date"]].sort_values("station_id")

    destination = Path(output_dir) / (filename + ".csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Readers see a complete CSV, including when an Airflow retry replaces it.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            result.to_csv(handle, index=False)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Exported {len(result):,} station predictions to {destination}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshotdate", required=True, help="YYYY-MM-DD")
    parser.add_argument("--predictions-dir", default="datamart/gold/predictions")
    parser.add_argument("--stations-file", default="data/stations.csv")
    parser.add_argument("--output-dir", default="datamart/business/predictions")
    args = parser.parse_args()
    export_predictions(args.snapshotdate, args.predictions_dir, args.stations_file, args.output_dir)

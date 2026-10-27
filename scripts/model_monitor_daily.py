import argparse
import os
import shutil
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyspark

import utils.model_monitoring as monitoring
from utils.model_store import ModelStore
from utils.monitoring_store import log_monitoring

# PURPOSE
# Daily monitoring job. Runs after the daily prediction job and checks YESTERDAY, the latest day whose real trip
# counts (labels) exist. Writes one row per monitored day to datamart/gold/monitoring/model_monitoring_YYYY_MM_DD.parquet
# for the dashboard, and prints an ALERT line when something needs attention.
#
# WHAT IT DOES (for the monitored day D = snapshot date - 1)
# 1. MODEL QUALITY: joins the predictions for D with the labels for D -> model MAE and RMSE vs the lag-7 baseline,
#    bias (over/under-predicting), MAE for busy vs quiet stations, and the day's type (weekend, holiday, rain day)
# 2. ALERTS: 7-day rolling MAE vs the production model's expected MAE: its test MAE at first, then refreshed
#    every month with its MAE on the newest month (each monthly retrain re-scores the champion, see MLflow training metadata)
#      ok        rolling MAE <= 1.2 x expected MAE
#      warning   rolling MAE >  1.2 x expected MAE
#      critical  rolling MAE >= rolling lag-7 baseline MAE (the model is no better than the simple guess)
# 3. DATA DRIFT: PSI of each per-station trip history feature, last 30 days vs the model's training data
#      none < 0.1 <= moderate <= 0.25 < significant
# 4. DATA QUALITY: row counts of predictions / labels / features, stations without a label, nulls, negative
#    predictions, missing feature partitions in the drift window -> ok or warning
# If there are no predictions for D (e.g. the first day of the DAG), nothing is written.
#
# to call this script (from the repo root): python scripts/model_monitor_daily.py --snapshotdate "2016-07-02"
# to replay a range in one go:              python scripts/model_monitor_daily.py --snapshotdate "2016-07-02" --enddate "2017-11-01"


def monitor_one_day(snapshotdate, spark):
    monitored_date = (datetime.strptime(snapshotdate, "%Y-%m-%d").date() - timedelta(days=1)).strftime("%Y-%m-%d")

    predictions = monitoring.read_partition(spark, monitoring.PREDICTIONS_DIR, "predictions_", monitored_date)
    if predictions is None:
        print(monitored_date, "no predictions for this day, nothing to monitor")
        return None
    labels = monitoring.read_partition(spark, monitoring.LABEL_DIR, "gold_label_store_", monitored_date)
    features = monitoring.read_partition(spark, monitoring.FEATURE_DIR, "gold_feature_store_", monitored_date)
    store = ModelStore()
    if predictions.empty:
        raise ValueError("Empty prediction partition for " + monitored_date)
    for column in ("model_version", "mlflow_run_id"):
        if predictions[column].isna().any() or predictions[column].nunique() != 1:
            raise ValueError("Expected exactly one " + column + " per prediction day")
    saved = store.for_run(predictions["mlflow_run_id"].iloc[0])
    model_version = saved.version
    if not predictions["model_version"].eq(model_version).all():
        raise ValueError("Prediction model_version disagrees with MLflow source run")
    if saved.trained_through >= monitored_date or saved.effective_from > monitored_date:
        raise ValueError("Prediction source was not eligible on the monitored date")
    model_info = store.info(saved)
    row = {"monitored_date": monitored_date, "model_version": model_version, "mlflow_run_id": saved.run_id}
    # Stable schema, including days whose late/invalid labels prevent performance scoring.
    for name in ("station_count", "actual_demand", "model_absolute_error", "baseline_absolute_error",
                 "stations", "actual_trips", "predicted_trips", "model_mae", "baseline_mae", "model_rmse",
                 "baseline_rmse", "model_bias", "busy_stations_mae", "quiet_stations_mae",
                 "model_wmape_pct", "baseline_wmape_pct", "improvement_pct"):
        row[name] = np.nan
    row.update(is_weekend=None, is_holiday=None, is_rain_day=None, precipitation_mm=np.nan)
    problems = []
    if labels is None or features is None:
        problems.append("labels or features missing/incomplete")
    else:
        try:
            for table in (predictions, labels, features):
                dates = pd.to_datetime(table["snapshot_date"], errors="raise")
                if table.empty or not dates.eq(pd.Timestamp(monitored_date)).all():
                    raise ValueError("Empty partition or unexpected snapshot_date")
            scored = monitoring.score_day(predictions, labels, features)
            row.update(monitoring.day_segments(features))
            row.update(scored)
        except ValueError as error:
            problems.append(str(error))

    previous = monitoring.previous_monitoring_rows(spark, monitored_date, monitoring.ROLLING_DAYS)
    yesterday = (pd.Timestamp(monitored_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    prior = [part for part in previous if pd.to_datetime(part["monitored_date"]).eq(pd.Timestamp(yesterday)).any()]
    row["model_changed"] = bool(prior and prior[0]["model_version"].iloc[0] != model_version)
    expected = monitoring.expected_mae(store, model_version, model_info, monitored_date)
    row.update(monitoring.rolling_status(row, previous, model_info, expected))
    reference = model_info.get("psi_reference") or monitoring.build_psi_reference(spark, saved.run_id, model_info)
    row.update(monitoring.drift_status(spark, monitored_date, reference))
    empty_labels = pd.DataFrame(columns=["station_id", "label"])
    empty_features = pd.DataFrame(columns=["station_id"])
    row.update(monitoring.data_quality(predictions, labels if labels is not None else empty_labels,
                                      features if features is not None else empty_features,
                                      row["missing_feature_partitions"]))
    if row["drift_status"] == "unavailable":
        problems.append("drift features empty or nonfinite")
    if problems:
        row["data_quality_status"] = "warning"
        row["data_quality_reason"] = "; ".join(problems + [row["data_quality_reason"]])
    row["score_status"] = "scored" if np.isfinite(row["model_mae"]) else "not_scored"

    print(monitored_date, "monitoring | model", model_version,
          "| MAE", round(row["model_mae"], 2), "vs baseline", round(row["baseline_mae"], 2),
          "| 7-day MAE", round(row["rolling_model_mae"], 2), "(warning above", round(row["warning_threshold_mae"], 2), ")",
          "|", row["performance_status"],
          "| max PSI", round(row["max_psi"], 3), row["max_psi_feature"], "|", row["drift_status"],
          "| data quality", row["data_quality_status"])
    if row["performance_status"] != "ok":
        print("ALERT", monitored_date, row["performance_status"] + ":", row["performance_reason"])
    if row["drift_status"] == "significant":
        print("ALERT", monitored_date, "data drift:", row["features_significant_drift"], "feature(s) with PSI above",
              monitoring.PSI_SIGNIFICANT, "- largest", row["max_psi_feature"])
    if row["data_quality_status"] != "ok":
        print("ALERT", monitored_date, "data quality:", row["data_quality_reason"])

    # save the day's monitoring row (overwrite so re-running is safe) - IRL write to a database
    os.makedirs(monitoring.MONITORING_DIR, exist_ok=True)
    filepath = monitoring.partition_path(monitoring.MONITORING_DIR, "model_monitoring_", monitored_date)
    result = pd.DataFrame([row])
    result["monitored_date"] = pd.to_datetime(result["monitored_date"]).dt.date
    for name in ("is_weekend", "is_holiday", "is_rain_day"):
        result[name] = result[name].astype("boolean")
    # Pandas preserves nullable columns even when this day's entire metric column is missing.
    # Keep Spark-compatible partition directories for downstream readers.
    with tempfile.TemporaryDirectory(dir=monitoring.MONITORING_DIR) as temporary:
        result.to_parquet(os.path.join(temporary, "part-00000.parquet"), index=False)
        open(os.path.join(temporary, "_SUCCESS"), "w").close()
        if os.path.isdir(filepath):
            shutil.rmtree(filepath)
        os.replace(temporary, filepath)
    log_monitoring(store, "daily", monitored_date, saved.run_id, model_version, row,
                   tags={key: row[key] for key in ("performance_status", "drift_status", "data_quality_status", "score_status")},
                   table=result, artifact_path=filepath)
    print("saved to:", filepath)
    return row


def main(snapshotdate, enddate):
    print('\n\n---starting job---\n\n')

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # one day (Airflow) or a range of days in order (replay), so each day's 7-day window sees the days before it
    day = datetime.strptime(snapshotdate, "%Y-%m-%d").date()
    last = datetime.strptime(enddate or snapshotdate, "%Y-%m-%d").date()
    while day <= last:
        monitor_one_day(day.strftime("%Y-%m-%d"), spark)
        day += timedelta(days=1)

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD, monitors the day before")
    parser.add_argument("--enddate", type=str, default=None, help="optional YYYY-MM-DD, replays every snapshot date up to this one")
    args = parser.parse_args()
    main(args.snapshotdate, args.enddate)

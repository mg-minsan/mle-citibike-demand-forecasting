import glob
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import utils.data_loading_model_train as loader

# PURPOSE
# Helpers for the daily monitoring job (scripts/model_monitor_daily.py). Monitoring looks at a day whose real trip
# counts are known (the label arrives after the day ends) and answers four questions:
#   1. MODEL QUALITY: how far off were the model's predictions, compared with the lag-7 baseline? Also per segment:
#      busy vs quiet stations, and the day's type (weekend, holiday, rain) so errors can be grouped by them later
#   2. ALERTS: is the 7-day rolling MAE worse than the production model's expected MAE allows?
#   3. DATA DRIFT: do the last 30 days of features still look like the data the model was trained on? (PSI per feature)
#   4. DATA QUALITY: do predictions, labels and features line up (row counts, nulls, missing partitions)?
# Everything here works on small pandas tables: one day is about 600-750 stations.

PREDICTIONS_DIR = "datamart/gold/predictions/"
LABEL_DIR = "datamart/gold/label_store/"
FEATURE_DIR = "datamart/gold/feature_store/"
MODEL_BANK_DIR = "datamart/model_bank/"
MONITORING_DIR = "datamart/gold/monitoring/"
REFERENCE_DIR = os.path.join(MONITORING_DIR, "psi_reference/")

ROLLING_DAYS = 7               # alerts look at the last 7 monitored days, so one odd day (a storm) does not alert alone
WARNING_FACTOR = 1.2           # warning when the 7-day MAE is more than 20% worse than the model's expected MAE
BUSY_SHARE = 0.2               # the busiest 20% of stations (by 28-day average) form the "busy" segment
RAIN_MM = 1.0                  # a day with at least 1 mm of precipitation counts as a rain day
DRIFT_DAYS = 30                # drift compares the last 30 days of features with the model's training data
PSI_BINS = 10
PSI_MODERATE = 0.1             # PSI < 0.1 no change, 0.1-0.25 moderate change, > 0.25 significant change
PSI_SIGNIFICANT = 0.25

# drift is checked on the per-station trip history features. Calendar flags and weather are left out: they are the
# same for every station on a day and swing with the season by definition (a week in July has no January
# temperatures in it), so their PSI would be "significant" almost all year without anything being wrong
PSI_FEATURES = ["fe_lag_1", "fe_lag_7", "fe_lag_14", "fe_rolling_mean_7", "fe_rolling_mean_28"]


def partition_path(directory, prefix, date_str):
    return os.path.join(directory, prefix + date_str.replace("-", "_") + ".parquet")


def read_partition(spark, directory, prefix, date_str):
    # one dated partition as pandas, or None if that day was not written
    path = partition_path(directory, prefix, date_str)
    if not os.path.isdir(path):
        return None
    return spark.read.parquet(path).toPandas()


def read_model_info(model_version):
    with open(os.path.join(MODEL_BANK_DIR, model_version + ".json")) as handle:
        return json.load(handle)


def expected_mae(model_version, model_info, monitored_date):
    # the MAE the production model is expected to have, measured on the most recent month it was scored on.
    # At first that is its own test month. Each monthly retrain also re-scores the champion on the newest month
    # (champion_test_mae in the candidate's json), so while the champion stays, its expected MAE is refreshed every
    # month. Without this, a model tested on a quiet winter month would be judged against winter errors all summer.
    best = (model_info["retrained_on"], model_info["test_model_mae"], model_info["test_window"])
    for path in glob.glob(os.path.join(MODEL_BANK_DIR, "*.json")):
        if os.path.basename(path) == "champion_history.json":
            continue
        with open(path) as handle:
            info = json.load(handle)
        if info.get("champion_version") != model_version or info.get("champion_test_mae") is None:
            continue
        if best[0] < info["retrained_on"] <= monitored_date:
            best = (info["retrained_on"], info["champion_test_mae"], info["test_window"])
    return {"expected_mae": float(best[1]), "expected_mae_window": best[2][0] + " to " + best[2][1]}


# ---------- 1. model quality ----------

def score_day(predictions, labels, features):
    # join predictions with the real trip counts and compute the day's errors (stations without a label are counted
    # by the data quality check instead)
    df = predictions.merge(labels[["station_id", "label"]], on="station_id", how="inner")
    if len(df) == 0:
        raise ValueError("no prediction matches a label, nothing to score")
    model_error = df["model_prediction"] - df["label"]
    baseline_error = df["baseline_prediction"] - df["label"]

    # busy vs quiet stations: is the model equally good for both? (bias / fairness check across station groups)
    df = df.merge(features[["station_id", "fe_rolling_mean_28"]], on="station_id", how="left")
    cutoff = df["fe_rolling_mean_28"].quantile(1 - BUSY_SHARE)
    busy = df["fe_rolling_mean_28"] >= cutoff

    return {
        "stations": int(len(df)),
        "actual_trips": int(df["label"].sum()),
        "predicted_trips": round(float(df["model_prediction"].sum()), 1),
        "model_mae": float(model_error.abs().mean()),
        "baseline_mae": float(baseline_error.abs().mean()),
        "model_rmse": float(np.sqrt((model_error ** 2).mean())),
        "baseline_rmse": float(np.sqrt((baseline_error ** 2).mean())),
        "model_bias": float(model_error.mean()),           # > 0 over-predicts, < 0 under-predicts
        "busy_stations_mae": float(model_error[busy].abs().mean()),
        "quiet_stations_mae": float(model_error[~busy].abs().mean()),
    }


def day_segments(features):
    # the day's type, the same for every station; the summary groups the daily MAEs by these
    precipitation = float(features["total_precipitation_mm"].iloc[0])
    return {
        "is_weekend": bool(features["fe_is_weekend"].iloc[0] == 1),
        "is_holiday": bool(features["fe_is_holiday"].max() == 1),
        "precipitation_mm": precipitation,
        "is_rain_day": bool(precipitation >= RAIN_MM),
    }


# ---------- 4. data quality ----------

def data_quality(predictions, labels, features, missing_feature_partitions):
    # simple checks that the three tables of the day line up; a warning here means the MAE of the day is less reliable
    feature_columns = [c for c in loader.FEATURE_COLUMNS if c in features.columns]
    unmatched = len(set(predictions["station_id"]) - set(labels["station_id"]))
    checks = {
        "prediction_rows": int(len(predictions)),
        "label_rows": int(len(labels)),
        "feature_rows": int(len(features)),
        "predictions_without_label": int(unmatched),
        "duplicate_prediction_stations": int(predictions["station_id"].duplicated().sum()),
        "null_feature_values": int(features[feature_columns].isna().sum().sum()),
        "null_or_negative_predictions": int((predictions["model_prediction"].isna() | (predictions["model_prediction"] < 0)).sum()),
        "missing_feature_partitions": int(missing_feature_partitions),
    }
    problems = []
    if not checks["prediction_rows"] == checks["label_rows"] == checks["feature_rows"]:
        problems.append("row counts differ")
    for name in ["predictions_without_label", "duplicate_prediction_stations", "null_feature_values",
                 "null_or_negative_predictions", "missing_feature_partitions"]:
        if checks[name] > 0:
            problems.append(str(checks[name]) + " " + name.replace("_", " "))
    checks["data_quality_status"] = "warning" if len(problems) > 0 else "ok"
    checks["data_quality_reason"] = "; ".join(problems) if len(problems) > 0 else "all checks passed"
    return checks


# ---------- 2. alerts ----------

def previous_monitoring_rows(spark, monitored_date, days):
    # the monitoring rows already written for the days before the monitored date (the rolling window)
    rows = []
    day = datetime.strptime(monitored_date, "%Y-%m-%d").date()
    for offset in range(1, days):
        found = read_partition(spark, MONITORING_DIR, "model_monitoring_", (day - timedelta(days=offset)).strftime("%Y-%m-%d"))
        if found is not None:
            rows.append(found)
    return rows


def rolling_status(today, previous_rows, model_info, expected):
    # 7-day rolling MAE (today + up to 6 earlier days) and the alert level
    window = pd.concat([pd.DataFrame([today])] + previous_rows, ignore_index=True)
    rolling_model_mae = float(window["model_mae"].mean())
    rolling_baseline_mae = float(window["baseline_mae"].mean())
    threshold = WARNING_FACTOR * expected["expected_mae"]

    if rolling_model_mae >= rolling_baseline_mae:
        status, reason = "critical", "the model is no better than the lag-7 baseline over the last days"
    elif rolling_model_mae > threshold:
        status, reason = "warning", "7-day MAE is more than " + str(round((WARNING_FACTOR - 1) * 100)) + "% above the model's expected MAE"
    else:
        status, reason = "ok", "within limits"

    return {
        "rolling_days": int(len(window)),
        "rolling_model_mae": rolling_model_mae,
        "rolling_baseline_mae": rolling_baseline_mae,
        "model_test_mae": float(model_info["test_model_mae"]),
        "expected_mae": expected["expected_mae"],
        "expected_mae_window": expected["expected_mae_window"],
        "warning_threshold_mae": threshold,
        "performance_status": status,
        "performance_reason": reason,
    }


# ---------- 3. data drift (PSI) ----------

def build_psi_reference(spark, model_version, model_info):
    # bin edges and shares of each feature over the model's training data, saved once per model version
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    path = os.path.join(REFERENCE_DIR, model_version + ".json")
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)

    print("building PSI reference from the training data of", model_version)
    train = loader.load_feature_dataset(model_info["trained_from"], model_info["trained_through"],
                                        FEATURE_DIR, spark).select(*PSI_FEATURES).toPandas()
    reference = {}
    for feature in PSI_FEATURES:
        values = train[feature].astype(float).values
        # quantile edges; repeated edges (features with many zeros) are merged, so some features get fewer bins
        edges = np.unique(np.quantile(values, np.linspace(0, 1, PSI_BINS + 1)[1:-1]))
        shares = bin_shares(values, edges)
        reference[feature] = {"edges": edges.tolist(), "shares": shares.tolist()}

    with open(path + ".tmp", "w") as handle:
        json.dump(reference, handle)
    os.replace(path + ".tmp", path)
    return reference


def bin_shares(values, edges):
    # share of values in each bin: (-inf, e1], (e1, e2], ..., (e_last, inf)
    counts = np.bincount(np.searchsorted(edges, values, side="left"), minlength=len(edges) + 1)
    return counts / max(len(values), 1)


def psi(reference_shares, current_shares):
    # population stability index; a tiny floor stops empty bins from dividing by zero
    ref = np.clip(np.asarray(reference_shares), 1e-4, None)
    cur = np.clip(np.asarray(current_shares), 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift_status(spark, monitored_date, reference):
    # PSI of each feature: last 30 days of features vs the training data. Missing days are skipped (and counted for
    # the data quality check) instead of stopping the job
    end = datetime.strptime(monitored_date, "%Y-%m-%d").date()
    days = [(end - timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(DRIFT_DAYS)]
    paths = [partition_path(FEATURE_DIR, "gold_feature_store_", day) for day in days]
    found = [path for path in paths if os.path.isdir(path)]
    recent = spark.read.parquet(*found).select(*PSI_FEATURES).toPandas()

    result = {"drift_window_days": len(found), "missing_feature_partitions": len(paths) - len(found)}
    for feature in PSI_FEATURES:
        edges = np.asarray(reference[feature]["edges"])
        result["psi_" + feature] = psi(reference[feature]["shares"], bin_shares(recent[feature].astype(float).values, edges))

    worst = max(PSI_FEATURES, key=lambda f: result["psi_" + f])
    max_psi = result["psi_" + worst]
    if max_psi > PSI_SIGNIFICANT:
        status = "significant"
    elif max_psi > PSI_MODERATE:
        status = "moderate"
    else:
        status = "none"
    result.update({"max_psi": max_psi, "max_psi_feature": worst, "drift_status": status,
                   "features_significant_drift": int(sum(result["psi_" + f] > PSI_SIGNIFICANT for f in PSI_FEATURES))})
    return result

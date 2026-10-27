import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import utils.data_loading_model_train as loader
from utils.prediction_monitoring import daily_performance, rolling_performance

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
    if not os.path.isfile(os.path.join(path, "_SUCCESS")):
        return None
    return spark.read.parquet(path).toPandas()


def expected_mae(store, model_version, model_info, monitored_date):
    # Only information available by the historical monitored date may refresh the expectation.
    best = (model_info["retrained_on"], model_info["test_model_mae"], model_info["test_window"])
    for saved in store.models():
        if not best[0] < saved.effective_from <= monitored_date:
            continue
        info = store.info(saved)
        if info.get("champion_version") == model_version and info.get("champion_test_mae") is not None:
            best = (info["retrained_on"], info["champion_test_mae"], info["test_window"])
    return {"expected_mae": float(best[1]), "expected_mae_window": best[2][0] + " to " + best[2][1]}


# ---------- 1. model quality ----------

def score_day(predictions, labels, features):
    daily = daily_performance(predictions, labels)
    if len(daily) != 1:
        raise ValueError("Expected one day per monitoring partition")
    keys = ["snapshot_date", "station_id"]
    frames = [table.copy() for table in (predictions, labels, features)]
    for table in frames:
        table["snapshot_date"] = pd.to_datetime(table["snapshot_date"]).dt.normalize()
    predictions, labels, features = frames
    if features[keys].isna().any().any() or features.duplicated(keys).any():
        raise ValueError("Expected unique, non-null feature station/day keys")
    df = predictions.merge(labels[keys + ["label"]], on=keys, validate="one_to_one")
    df = df.merge(features[keys + ["fe_rolling_mean_28"]], on=keys, how="outer",
                  validate="one_to_one", indicator=True)
    if not df["_merge"].eq("both").all() or not np.isfinite(df["fe_rolling_mean_28"]).all():
        raise ValueError("Feature station/day coverage differs or station history is invalid")
    errors = (df["model_prediction"] - df["label"]).abs()
    busy = df["fe_rolling_mean_28"] >= df["fe_rolling_mean_28"].quantile(1 - BUSY_SHARE)
    row = daily.iloc[0].drop(labels=["snapshot_date", "model_version", "model_changed"]).to_dict()
    row.update(stations=int(row["station_count"]), actual_trips=float(row["actual_demand"]),
               busy_stations_mae=float(errors[busy].mean()), quiet_stations_mae=float(errors[~busy].mean()))
    return row


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
        "null_or_negative_predictions": int((~np.isfinite(predictions["model_prediction"]) | (predictions["model_prediction"] < 0)).sum()),
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
    end = pd.Timestamp(today["monitored_date"])
    dates = pd.to_datetime(window["monitored_date"])
    window = window[(dates > end - pd.Timedelta(days=ROLLING_DAYS)) & (dates <= end)].copy()
    window["snapshot_date"] = window["monitored_date"]
    rolling = rolling_performance(window, ROLLING_DAYS).iloc[-1]
    rolling_model_mae = float(rolling.get("rolling_model_mae", np.nan))
    rolling_baseline_mae = float(rolling.get("rolling_baseline_mae", np.nan))
    threshold = WARNING_FACTOR * expected["expected_mae"]

    if not np.isfinite(today["model_mae"]):
        status, reason = "not_scored", "incomplete or invalid station/day data"
    elif rolling_model_mae >= rolling_baseline_mae:
        status, reason = "critical", "the model is no better than the lag-7 baseline over the last days"
    elif rolling_model_mae > threshold:
        status, reason = "warning", "7-day MAE is more than " + str(round((WARNING_FACTOR - 1) * 100)) + "% above the model's expected MAE"
    else:
        status, reason = "ok", "within limits"

    return {
        "rolling_days": int(window["model_mae"].notna().sum()),
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
    reference = psi_reference(train)

    with open(path + ".tmp", "w") as handle:
        json.dump(reference, handle)
    os.replace(path + ".tmp", path)
    return reference


def psi_reference(train):
    """Construct the frozen reference from final training inputs."""
    reference = {}
    for feature in PSI_FEATURES:
        values = train[feature].to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("PSI reference requires non-empty finite training features")
        edges = np.unique(np.quantile(values, np.linspace(0, 1, PSI_BINS + 1)[1:-1]))
        reference[feature] = {"edges": edges.tolist(), "shares": bin_shares(values, edges).tolist()}
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
    found = [path for path in paths if os.path.isfile(os.path.join(path, "_SUCCESS"))]
    recent = (spark.read.parquet(*found).select(*PSI_FEATURES).toPandas()
              if found else pd.DataFrame(columns=PSI_FEATURES))
    if recent.empty or not np.isfinite(recent.to_numpy(dtype=float)).all():
        return {"drift_window_days": len(found), "missing_feature_partitions": len(paths) - len(found),
                "max_psi": np.nan, "max_psi_feature": "", "drift_status": "unavailable",
                "features_significant_drift": 0}

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

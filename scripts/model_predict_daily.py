import argparse
import glob
import json
import os
import re

import joblib
import pyspark

import utils.data_loading_model_train as loader

# PURPOSE
# Daily prediction job: predicts trip starts for every station for ONE day and saves the result as a dated
# gold partition (datamart/gold/predictions/predictions_YYYY_MM_DD.parquet).
#
# WHAT IT DOES
# 1. picks the latest model in the model bank that was trained only on data BEFORE the snapshot date
#    (when replaying history this stops a day from being predicted by a model that had already seen it)
# 2. builds the features for that day with the same loader training uses (same partitions, holiday flags, columns)
# 3. predicts, and saves model_prediction next to baseline_prediction (= fe_lag_7), so monitoring can later
#    compare model vs baseline from this table plus the label that arrives after the day ends
#
# to call this script (from inside scripts/): python model_predict_daily.py --snapshotdate "2016-07-01"

FEATURE_DIR = "datamart/gold/feature_store/"
MODEL_BANK_DIR = "datamart/model_bank/"
PREDICTIONS_DIR = "datamart/gold/predictions/"


HISTORY_PATH = os.path.join(MODEL_BANK_DIR, "champion_history.json")


def pick_model_path(snapshot_date_str):
    # 1. if a champion history exists: use the champion that was in place on the snapshot date
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as handle:
            history = json.load(handle)
        in_place = [entry for entry in history if entry["effective_from"] <= snapshot_date_str]
        if len(in_place) > 0:
            latest = max(in_place, key=lambda entry: entry["effective_from"])
            path = os.path.join(MODEL_BANK_DIR, latest["version"] + ".pkl")
            if not (os.path.exists(path) and os.path.exists(path[:-len(".pkl")] + ".json")):
                raise FileNotFoundError("champion " + latest["version"] + " is in the history but its files are incomplete")
            return path
    # 2. no history yet: use the latest model trained before the snapshot date
    candidates = []
    for path in glob.glob(os.path.join(MODEL_BANK_DIR, "*_trained_through_*.pkl")):
        trained_through = re.search(r"_trained_through_(\d{4}-\d{2}-\d{2})\.pkl$", path).group(1)
        if trained_through < snapshot_date_str and os.path.exists(path[:-len(".pkl")] + ".json"):
            candidates.append((trained_through, path))
    if len(candidates) == 0:
        raise FileNotFoundError("no model in " + MODEL_BANK_DIR + " trained before " + snapshot_date_str)
    return max(candidates)[1]


def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # load the model and the info saved with it (the feature list guarantees the same columns, same order as training)
    model_path = pick_model_path(snapshotdate)
    model_version = os.path.basename(model_path)[:-len(".pkl")]
    model = joblib.load(model_path)
    with open(model_path[:-len(".pkl")] + ".json") as handle:
        info = json.load(handle)
    print("model used:", model_version)

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # features for the day (no label exists yet) and predictions
    features_df = loader.load_feature_dataset(snapshotdate, snapshotdate, FEATURE_DIR, spark).toPandas()
    if len(features_df) == 0:
        raise ValueError("no feature rows for " + snapshotdate)
    predictions = model.predict(features_df[info["features"]])

    result = features_df[["snapshot_date", "station_id"]].copy()
    result["model_prediction"] = predictions.astype(float)
    result["baseline_prediction"] = features_df["fe_lag_7"].astype(float)
    result["model_version"] = model_version

    # data checks: one prediction per station, no nulls, no negative predictions
    if len(result) != len(features_df) or result["station_id"].duplicated().any():
        raise ValueError("expected exactly one prediction per station")
    if result.isnull().any().any():
        raise ValueError("null values in predictions")
    if (result["model_prediction"] < 0).any():
        raise ValueError("negative predictions")
    print(snapshotdate, "predictions | stations:", len(result), "| mean prediction:", round(result["model_prediction"].mean(), 2),
          "| mean baseline:", round(result["baseline_prediction"].mean(), 2))

    # save the day's predictions (one small partition per day, overwrite so re-running is safe) - IRL write to a database
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    filepath = os.path.join(PREDICTIONS_DIR, "predictions_" + snapshotdate.replace("-", "_") + ".parquet")
    spark.createDataFrame(result).coalesce(1).write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyspark

import utils.data_loading_model_train as loader
from utils.model_store import ModelStore
from utils.monitoring_store import log_monitoring, monitoring_tables

# PURPOSE
# Monthly explanation job (SHAP). Runs after the monthly retrain and explains the model that retrain saved, on its
# former test month (now included in the final refit): which inputs drive the predictions, and did that change since the previous retrain (attribution drift)?
#
# WHAT IT DOES (for the model retrained on the snapshot date)
# 1. loads the model and its metadata from MLflow, and the features of its test month
# 2. runs the pipeline's own preprocessing first, so the final estimator is explained on the columns it really sees:
#    StationProfile turns station_id into station_mean and station_weekend_ratio (poisson regression also one-hot
#    encodes the weekday and scales the rest; the weekday columns are added back up into fe_day_of_week)
# 3. SHAP values on a fixed sample of the test month: TreeExplainer for xgboost and random forest (approximate for
#    random forest, exact would take hours on its deep trees), LinearExplainer for poisson regression.
#    importance = mean |SHAP value| per column, share = importance / sum of importances.
#    Raw importances are in log trips for xgboost and poisson regression (they predict log counts) and in trips
#    for random forest, so compare models by share
# 4. ATTRIBUTION CHANGE (an investigation signal, not a concept-drift test) vs the previous retrain: shift = half the sum of the absolute share changes
#    (0 = the model relies on the same inputs in the same proportions, 1 = completely different inputs)
# Writes one table per retrain to datamart/gold/monitoring/shap/model_shap_YYYY_MM_DD.parquet (one row per input).
# Days without a retrain are skipped, so the job can be replayed over a range of days.
#
# to call this script (from the repo root): python scripts/model_explain.py --snapshotdate "2016-07-01"
# to replay every retrain in a range:      python scripts/model_explain.py --snapshotdate "2016-07-01" --enddate "2017-10-01"

FEATURE_DIR = "datamart/gold/feature_store/"
SHAP_DIR = "datamart/gold/monitoring/shap/"
SAMPLE_ROWS = 2000             # SHAP on a random forest is slow, a fixed sample of the test month is enough for averages
SHIFT_WARNING = 0.2            # heuristic investigation signal, not a calibrated drift threshold


def transformed_inputs(pipeline, X):
    # everything before the final estimator, with readable column names
    Xt = pipeline[:-1].transform(X)
    if isinstance(Xt, pd.DataFrame):
        return Xt, list(Xt.columns)
    # poisson regression: the ColumnTransformer output, named after its input columns (one-hot columns get the value)
    column_transformer = pipeline[-2]
    names = []
    for _, transformer, columns in column_transformer.transformers_:
        if isinstance(transformer, str) and transformer == "drop":
            continue
        columns = [column_transformer.feature_names_in_[c] if isinstance(c, (int, np.integer)) else c for c in columns]
        if hasattr(transformer, "categories_"):
            for column, values in zip(columns, transformer.categories_):
                names += [column + "_" + str(value) for value in values]
        else:
            names += list(columns)
    return pd.DataFrame(Xt, columns=names, index=X.index), names


def shap_values(estimator, Xt):
    import shap
    if hasattr(estimator, "coef_"):   # poisson regression: a linear model of the log of the prediction
        return np.asarray(shap.LinearExplainer(estimator, Xt).shap_values(Xt))
    if hasattr(estimator, "estimators_"):
        # random forest: its trees are grown deep on all rows, so exact SHAP would take hours; the approximate
        # method (each split's change in prediction along the row's path) takes seconds and ranks inputs the same way
        return np.asarray(shap.TreeExplainer(estimator).shap_values(Xt, approximate=True, check_additivity=False))
    return np.asarray(shap.TreeExplainer(estimator).shap_values(Xt, check_additivity=False))


def explain(snapshotdate, spark):
    store = ModelStore()
    cutoff = (datetime.strptime(snapshotdate, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    saved = store.for_cutoff(cutoff)
    if saved is None or saved.effective_from != snapshotdate:
        return None
    version, info = saved.version, store.info(saved)
    pipeline = store.load(saved)

    # 1. the test month of that retrain, a fixed sample
    test_start, test_end = info["test_window"]
    features = loader.load_feature_dataset(test_start, test_end, FEATURE_DIR, spark).toPandas()
    features = features.sort_values(["snapshot_date", "station_id"]).reset_index(drop=True)
    X = features[info["features"]].sample(n=min(SAMPLE_ROWS, len(features)), random_state=42)

    # 2. the columns the final estimator sees
    Xt, names = transformed_inputs(pipeline, X)

    # 3. SHAP importances; one-hot weekday columns are summed back into fe_day_of_week before averaging
    values = pd.DataFrame(shap_values(pipeline[-1], Xt), columns=names)
    groups = [("fe_day_of_week" if name.startswith("fe_day_of_week_") else name) for name in names]
    values = values.T.groupby(groups, sort=False).sum().T
    importance = values.abs().mean()
    table = pd.DataFrame({"feature": importance.index, "mean_abs_shap": importance.values})
    total = table["mean_abs_shap"].sum()
    table["share"] = table["mean_abs_shap"] / total if total > 0 else 0.0
    table = table.sort_values("share", ascending=False).reset_index(drop=True)
    table["rank"] = table.index + 1

    # 4. attribution drift vs the previous retrain's explanation
    previous = previous_explanation(snapshotdate, store)
    if previous is None or set(previous["feature"]) != set(table["feature"]):
        table["previous_share"] = np.nan
        shift, top_changed = np.nan, False
    else:
        table = table.merge(previous[["feature", "share"]].rename(columns={"share": "previous_share"}), on="feature", how="left")
        joined = pd.concat([table.set_index("feature")["share"], previous.set_index("feature")["share"]], axis=1).fillna(0)
        shift = float((joined.iloc[:, 0] - joined.iloc[:, 1]).abs().sum() / 2)
        top_changed = bool(table["feature"].iloc[0] != previous.sort_values("share", ascending=False)["feature"].iloc[0])

    table.insert(0, "retrained_on", datetime.strptime(snapshotdate, "%Y-%m-%d").date())
    table.insert(1, "model_version", version)
    table.insert(2, "model", info["model"])
    table.insert(3, "promoted", bool(info["promoted"]))
    table["source_run_id"] = saved.run_id
    table["explanation_scope"] = "final_refit_on_seen_training_month"
    table["method"] = "shap"
    table["test_window"] = test_start + " to " + test_end
    table["explained_rows"] = len(X)
    table["attribution_shift"] = shift
    table["top_feature_changed"] = top_changed

    print(snapshotdate, "SHAP |", version, "| promoted" if info["promoted"] else "| not promoted",
          "| test month", test_start, "to", test_end, "|", len(X), "rows")
    for _, row in table.head(8).iterrows():
        print(f"   {row['rank']:>2}. {row['feature']:<24} share {row['share']:6.1%}")
    if not np.isfinite(shift):
        print("   attribution shift: no comparable earlier explanation")
    else:
        print(f"   attribution shift vs previous retrain: {shift:.3f}", "| top feature changed" if top_changed else "")
        if shift > SHIFT_WARNING:
            print("INVESTIGATE", snapshotdate, "attribution change across models/months; not a concept-drift test (shift",
                  round(shift, 3), ")")

    os.makedirs(SHAP_DIR, exist_ok=True)
    filepath = os.path.join(SHAP_DIR, "model_shap_" + snapshotdate.replace("-", "_") + ".parquet")
    spark.createDataFrame(table).coalesce(1).write.mode("overwrite").parquet(filepath)
    log_monitoring(store, "shap", snapshotdate, saved.run_id, version,
                   {"attribution_shift": shift},
                   tags={"explanation_scope": "final_refit_on_seen_training_month", "method": "shap"},
                   table=table, artifact_path=filepath)
    print("saved to:", filepath)
    return table


def previous_explanation(snapshotdate, store):
    earlier = [table for table in monitoring_tables(store, "shap")
               if str(table["retrained_on"].iloc[0]) < snapshotdate]
    return max(earlier, key=lambda table: str(table["retrained_on"].iloc[0]), default=None)


def main(snapshotdate, enddate):
    print('\n\n---starting job---\n\n')

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # one day (Airflow, on the 1st) or a range of days in order (replay), so each retrain compares with the one before
    day = datetime.strptime(snapshotdate, "%Y-%m-%d").date()
    last = datetime.strptime(enddate or snapshotdate, "%Y-%m-%d").date()
    explained = 0
    while day <= last:
        if explain(day.strftime("%Y-%m-%d"), spark) is not None:
            explained += 1
        day += timedelta(days=1)
    if explained == 0:
        print("no retrain on", snapshotdate if enddate is None else snapshotdate + " to " + enddate, "- nothing to explain")

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD, the day of the retrain")
    parser.add_argument("--enddate", type=str, default=None, help="optional YYYY-MM-DD, explains every retrain up to this day")
    args = parser.parse_args()
    main(args.snapshotdate, args.enddate)

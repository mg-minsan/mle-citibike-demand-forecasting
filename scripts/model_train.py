import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyspark
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

import utils.data_loading_model_train as loader
import utils.model_tuning as tuning
from utils.feature_importance import measure_feature_importance
from utils.model_store import ModelStore
from utils.model_monitoring import psi_reference
from utils.station_profile import StationProfile

# PURPOSE
# Monthly retraining job. Each run compares three candidate models (poisson regression, random forest, xgboost),
# scores the winner on a separate test month, and saves the winner to MLflow.
#   1. split the data by walking back from the day before the snapshot date:
#        test       = the last month (only used to record a score)
#        validation = the month before that (only used to pick the winner)
#        train      = everything from 30 Jan 2016 up to the validation month
#   2. VALIDATION: fit all three candidates on train, score each on the validation month (and the baseline
#      fe_lag_7), pick the lowest MAE
#   2b. TUNING: search settings for the winning model type only (about 10 trials, walk-forward folds on data before
#       the test month, last month's settings kept unless clearly beaten)
#   3. TEST: score the winner (fitted on train + validation) once on the test month, record that MAE
#   4. FINAL: refit the winner on ALL data up to the day before the snapshot date and log the full
#      pipeline, training info, parameters and metrics to MLflow; register it as a model version.
#   5. PROMOTION: the new model only becomes the champion if its test MAE beats the current champion's
#      on the same test month. Immutable MLflow run tags preserve the date-aware champion history.
# Example: snapshot 2017-11-01 -> train 30 Jan 2016 to 31 Aug 2017, validate on Sep, test on Oct,
#          then refit the winner through 31 Oct 2017.
# Daily prediction uses the champion effective on its date, regardless of model type.
# If a completed model for that cutoff is already in MLflow nothing is done, so reruns do not overwrite.
# to call this script (from inside scripts/): python model_train.py --snapshotdate "2017-11-01"

TRAIN_START = "2016-01-30"   # first 28 days dropped as warm-up since fe_rolling_mean_28 is not available
FEATURE_DIR = "datamart/gold/feature_store/"
LABEL_DIR = "datamart/gold/label_store/"
PROMOTION_MARGIN = 0.01   # the candidate's test MAE must beat the champion's by at least this fraction (0.01 = 1%)

# fe_month is left out on purpose: increasing integer is not a desired signal
MODEL_FEATURES = [c for c in loader.FEATURE_COLUMNS if c != "fe_month"]
MODEL_INPUTS = MODEL_FEATURES + ["station_id"]   # StationProfile turns station_id into 2 features
COUNT_COLUMNS = ["fe_lag_1", "fe_lag_7", "fe_lag_14", "fe_rolling_mean_7", "fe_rolling_mean_28"]


def rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def build_candidates():
    # every candidate starts with StationProfile (two per-station features learned from the training rows only)
    poisson = make_pipeline(
        StationProfile(),
        ColumnTransformer([
            ("day", OneHotEncoder(handle_unknown="ignore"), ["fe_day_of_week"]),
            ("counts", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), COUNT_COLUMNS + ["station_mean"]),
        ], remainder=StandardScaler()),
        PoissonRegressor(alpha=1.0, max_iter=500),
    )
    forest = make_pipeline(
        StationProfile(),
        RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42),
    )
    xgboost = make_pipeline(
        StationProfile(),
        XGBRegressor(objective="count:poisson", tree_method="hist", n_estimators=300, learning_rate=0.05,
                     max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=42),
    )
    return {"poisson regression": poisson, "random forest": forest, "xgboost (poisson)": xgboost}


def score_champion(store, champion, test_df, label):
    # Use the complete pipeline and feature order saved with this historical champion.
    features = store.info(champion)["features"]
    model = store.load(champion)
    return mean_absolute_error(test_df[label], model.predict(test_df[features]))


def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    snapshot = datetime.strptime(snapshotdate, "%Y-%m-%d").date()
    trained_through = snapshot - timedelta(days=1)
    through_str = trained_through.strftime("%Y-%m-%d")

    store = ModelStore()
    if store.for_cutoff(through_str) is not None:
        store.sync_champion_alias()
        print("a complete model trained through", through_str, "is already in MLflow, nothing to do")
        return

    # three windows, walking back from the cutoff:
    # test = the last month, validation = the month before it, train = everything earlier
    test_start = trained_through.replace(day=1)
    valid_end = test_start - timedelta(days=1)
    valid_start = valid_end.replace(day=1)
    train_end = valid_start - timedelta(days=1)
    fmt = lambda d: d.strftime("%Y-%m-%d")

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    train_df = loader.load_model_dataset(TRAIN_START, fmt(train_end), FEATURE_DIR, LABEL_DIR, spark).toPandas()
    valid_df = loader.load_model_dataset(fmt(valid_start), fmt(valid_end), FEATURE_DIR, LABEL_DIR, spark).toPandas()
    test_df = loader.load_model_dataset(fmt(test_start), through_str, FEATURE_DIR, LABEL_DIR, spark).toPandas()
    spark.stop()

    label = loader.LABEL_COLUMN
    candidates = build_candidates()

    print(f"\ntrain {TRAIN_START} to {fmt(train_end)} ({len(train_df)} rows)")
    print(f"validate {fmt(valid_start)} to {fmt(valid_end)} ({len(valid_df)} rows)")
    print(f"test {fmt(test_start)} to {through_str} ({len(test_df)} rows)")

    # 1. VALIDATION: fit each candidate on train, score on the validation month, pick the lowest MAE
    valid_baseline_mae = mean_absolute_error(valid_df[label], valid_df["fe_lag_7"])
    valid_baseline_rmse = rmse(valid_df[label], valid_df["fe_lag_7"])
    print(f"\n{'validation':<22} baseline MAE {valid_baseline_mae:7.3f} | RMSE {valid_baseline_rmse:7.3f}")
    valid_mae = {}
    valid_rmse = {}
    for name, model in candidates.items():
        model.fit(train_df[MODEL_INPUTS], train_df[label])
        valid_pred = model.predict(valid_df[MODEL_INPUTS])
        valid_mae[name] = mean_absolute_error(valid_df[label], valid_pred)
        valid_rmse[name] = rmse(valid_df[label], valid_pred)
        improvement = (valid_baseline_mae - valid_mae[name]) / valid_baseline_mae * 100
        print(f"{name:<22} MAE {valid_mae[name]:7.3f} | RMSE {valid_rmse[name]:7.3f} | vs baseline {improvement:+6.1f}%")
    best_name = min(valid_mae, key=valid_mae.get)
    print("picked:", best_name)

    # 2. TUNING: search settings for the winning model type, using only data before the test month
    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    seed = int(through_str.replace("-", ""))
    print(f"\ntuning {best_name} (walk-forward folds ending at the validation month)")
    params, tuning_summary = tuning.tune(best_name, candidates[best_name], train_valid_df, MODEL_INPUTS, label,
                                         fmt(valid_start), seed, store.last_best_params(best_name, through_str))
    print(f"chosen: {tuning_summary['chosen_kind']} | tuning MAE {tuning_summary['chosen_mae']:.3f} vs defaults {tuning_summary['default_mae']:.3f}")
    unfitted_winner = clone(candidates[best_name]).set_params(**params)
    del candidates, model                        # free the fitted models

    # 3. TEST: refit the winner on train + validation (the same data the champion had) and score it once on the test month
    scored_model = clone(unfitted_winner)
    scored_model.fit(train_valid_df[MODEL_INPUTS], train_valid_df[label])
    test_pred = scored_model.predict(test_df[MODEL_INPUTS])
    del train_valid_df
    test_mae = mean_absolute_error(test_df[label], test_pred)
    test_rmse = rmse(test_df[label], test_pred)
    test_baseline_mae = mean_absolute_error(test_df[label], test_df["fe_lag_7"])
    test_baseline_rmse = rmse(test_df[label], test_df["fe_lag_7"])
    test_improvement = (test_baseline_mae - test_mae) / test_baseline_mae * 100
    print(f"\ntest month | {best_name} MAE {test_mae:7.3f} RMSE {test_rmse:7.3f} | baseline MAE {test_baseline_mae:7.3f} RMSE {test_baseline_rmse:7.3f} | MAE vs baseline {test_improvement:+6.1f}%")

    # Explain the evaluated model, not the final model that will have seen the test rows.
    print("\nmeasuring held-out permutation importance (up to 1000 rows, 3 repeats)")
    importance = measure_feature_importance(scored_model, test_df[MODEL_INPUTS], test_df[label], seed)
    importance["test_window"] = [fmt(test_start), through_str]
    del scored_model

    # PROMOTION: the candidate only becomes the champion if it beats the current champion on this same test month
    champion = store.champion(snapshotdate)
    champion_mae = None
    if champion is None:
        promote, reason = True, "no champion yet"
    elif champion.trained_through >= fmt(test_start):
        promote, reason = False, "the champion has already seen the test month, so it cannot be compared fairly"
    else:
        champion_mae = score_champion(store, champion, test_df, label)
        promote = test_mae < champion_mae * (1 - PROMOTION_MARGIN)
        reason = f"candidate test MAE {test_mae:.3f} vs champion {champion.version} {champion_mae:.3f}"
    print(f"promote: {promote} ({reason})")

    # FINAL: refit through the cutoff and save to MLflow, whether promoted or not.
    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    final_model = clone(unfitted_winner)
    final_model.fit(all_df[MODEL_INPUTS], all_df[label])

    # Save explicit settings alongside the model ({} in "params" means the defaults).
    hyperparameters = {k: v for k, v in final_model.steps[-1][1].get_params().items()
                       if isinstance(v, (int, float, str, bool)) and v == v}   # v == v drops NaN
    version = best_name.replace(" ", "_").replace("(", "").replace(")", "") + "_trained_through_" + through_str
    info = {
        "model": best_name,
        "trained_from": TRAIN_START,
        "trained_through": through_str,
        "retrained_on": snapshotdate,
        "train_rows": len(all_df),
        "validation_window": [fmt(valid_start), fmt(valid_end)],
        "validation_mae": {name: round(value, 3) for name, value in valid_mae.items()},
        "validation_baseline_mae": round(valid_baseline_mae, 3),
        "validation_rmse": {name: round(value, 3) for name, value in valid_rmse.items()},
        "validation_baseline_rmse": round(valid_baseline_rmse, 3),
        "test_window": [fmt(test_start), through_str],
        "test_model_mae": round(test_mae, 3),
        "test_baseline_mae": round(test_baseline_mae, 3),
        "test_model_rmse": round(test_rmse, 3),
        "test_baseline_rmse": round(test_baseline_rmse, 3),
        "promoted": promote,
        "promotion_reason": reason,
        "champion_version": None if champion is None else champion.version,
        "champion_test_mae": None if champion_mae is None else round(champion_mae, 3),
        "features": MODEL_INPUTS,
        "label": label,
        "params": params,
        "tuning": tuning_summary,
        "tuning_seed": seed,
        "hyperparameters": hyperparameters,
        "feature_importance": importance,
        "psi_reference": psi_reference(all_df),
    }
    saved = store.save(final_model, info, version, input_example=all_df[MODEL_INPUTS].head(5))
    print("saved model to:", saved.model_uri, "| run:", saved.run_id)

    if promote:
        print("promoted to champion from", snapshotdate)
    else:
        print("not promoted, the champion stays")

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)

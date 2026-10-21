import argparse
import glob
import json
import os
from datetime import datetime, timedelta

import joblib
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
from utils.station_profile import StationProfile

# PURPOSE
# Monthly retraining job. Each run compares three candidate models (poisson regression, random forest, xgboost),
# scores the winner on a separate test month, and saves the winner to the model bank.
#   1. split the data by walking back from the day before the snapshot date:
#        test       = the last month (only used to record a score)
#        validation = the month before that (only used to pick the winner)
#        train      = everything from 30 Jan 2016 up to the validation month
#   2. VALIDATION: fit all three candidates on train, score each on the validation month (and the baseline
#      fe_lag_7), pick the lowest MAE
#   2b. TUNING: search settings for the winning model type only (about 10 trials, walk-forward folds on data before
#       the test month, last month's settings kept unless clearly beaten)
#   3. TEST: score the winner (fitted on train only) once on the test month, record that MAE
#   4. FINAL: refit the winner on ALL data up to the day before the snapshot date and save it as
#      <model>_trained_through_<day before snapshot>.pkl (+ a .json with training info, all three validation
#      scores and the test score)
#   5. PROMOTION: the new model only becomes the champion if its test MAE beats the current champion's on the same test month (champion_history.json).
# Example: snapshot 2017-11-01 -> train 30 Jan 2016 to 31 Aug 2017, validate on Sep, test on Oct,
#          then refit the winner through 31 Oct 2017.
# The daily prediction job uses the latest model trained before its date, whichever model type won.
# If a model for that cutoff is already in the bank nothing is done, so re-running does not overwrite.
# to call this script (from inside scripts/): python model_train.py --snapshotdate "2017-11-01"

TRAIN_START = "2016-01-30"   # first 28 days dropped as warm-up since fe_rolling_mean_28 is not available
FEATURE_DIR = "datamart/gold/feature_store/"
LABEL_DIR = "datamart/gold/label_store/"
MODEL_BANK_DIR = "datamart/model_bank/"
HISTORY_PATH = os.path.join(MODEL_BANK_DIR, "champion_history.json")
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

def read_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH) as handle:
        return sorted(json.load(handle), key=lambda entry: entry["effective_from"])


def score_champion(champion, test_df, label):
    # score the current champion on the same test rows, using the feature list saved with it
    with open(os.path.join(MODEL_BANK_DIR, champion["version"] + ".json")) as handle:
        features = json.load(handle)["features"]
    model = joblib.load(os.path.join(MODEL_BANK_DIR, champion["version"] + ".pkl"))
    return mean_absolute_error(test_df[label], model.predict(test_df[features]))


def last_best_params(model_name, through_str):
    # the settings chosen the last time this model type was tuned (None the first time), read from the newest json
    # in the bank that was trained before this cutoff, so a rerun of an old month never sees later months
    found = None
    for path in sorted(glob.glob(os.path.join(MODEL_BANK_DIR, "*.json"))):
        if os.path.basename(path) == "champion_history.json":
            continue
        with open(path) as handle:
            info = json.load(handle)
        if info.get("model") == model_name and info.get("params") and info["trained_through"] < through_str:
            found = info["params"]
    return found


def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    snapshot = datetime.strptime(snapshotdate, "%Y-%m-%d").date()
    trained_through = snapshot - timedelta(days=1)
    through_str = trained_through.strftime("%Y-%m-%d")

    if len(glob.glob(os.path.join(MODEL_BANK_DIR, "*_trained_through_" + through_str + ".pkl"))) > 0:
        print("a model trained through", through_str, "is already in the model bank, nothing to do")
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
                                         fmt(valid_start), seed, last_best_params(best_name, through_str))
    print(f"chosen: {tuning_summary['chosen_kind']} | tuning MAE {tuning_summary['chosen_mae']:.3f} vs defaults {tuning_summary['default_mae']:.3f}")
    unfitted_winner = clone(candidates[best_name]).set_params(**params)
    del candidates, model                        # free the fitted models

    # 3. TEST: refit the winner on train + validation (the same data the champion had) and score it once on the test month
    scored_model = clone(unfitted_winner)
    scored_model.fit(train_valid_df[MODEL_INPUTS], train_valid_df[label])
    test_pred = scored_model.predict(test_df[MODEL_INPUTS])
    del scored_model, train_valid_df
    test_mae = mean_absolute_error(test_df[label], test_pred)
    test_rmse = rmse(test_df[label], test_pred)
    test_baseline_mae = mean_absolute_error(test_df[label], test_df["fe_lag_7"])
    test_baseline_rmse = rmse(test_df[label], test_df["fe_lag_7"])
    test_improvement = (test_baseline_mae - test_mae) / test_baseline_mae * 100
    print(f"\ntest month | {best_name} MAE {test_mae:7.3f} RMSE {test_rmse:7.3f} | baseline MAE {test_baseline_mae:7.3f} RMSE {test_baseline_rmse:7.3f} | MAE vs baseline {test_improvement:+6.1f}%")

    # PROMOTION: the candidate only becomes the champion if it beats the current champion on this same test month
    earlier = [entry for entry in read_history() if entry["effective_from"] < snapshotdate]
    champion = earlier[-1] if len(earlier) > 0 else None
    champion_mae = None
    if champion is None:
        promote, reason = True, "no champion yet"
    elif champion["trained_through"] >= fmt(test_start):
        promote, reason = False, "the champion has already seen the test month, so it cannot be compared fairly"
    else:
        champion_mae = score_champion(champion, test_df, label)
        promote = test_mae < champion_mae * (1 - PROMOTION_MARGIN)
        reason = f"candidate test MAE {test_mae:.3f} vs champion {champion['version']} {champion_mae:.3f}"
    print(f"promote: {promote} ({reason})")

    # 3. FINAL: refit the winner on everything up to the cutoff and save it (promoted or not, it stays in the bank)
    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    final_model = clone(unfitted_winner)
    final_model.fit(all_df[MODEL_INPUTS], all_df[label])

    # the explicit settings of the saved model, so the json stands on its own ({} in "params" means the defaults)
    hyperparameters = {k: v for k, v in final_model.steps[-1][1].get_params().items()
                       if isinstance(v, (int, float, str, bool)) and v == v}   # v == v drops NaN
    version = best_name.replace(" ", "_").replace("(", "").replace(")", "") + "_trained_through_" + through_str
    os.makedirs(MODEL_BANK_DIR, exist_ok=True)
    joblib.dump(final_model, os.path.join(MODEL_BANK_DIR, version + ".pkl"))
    with open(os.path.join(MODEL_BANK_DIR, version + ".json"), "w") as handle:
        json.dump({
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
            "champion_version": None if champion is None else champion["version"],
            "champion_test_mae": None if champion_mae is None else round(champion_mae, 3),
            "features": MODEL_INPUTS,
            "label": label,
            "params": params,
            "tuning": tuning_summary,
            "tuning_seed": seed,
            "hyperparameters": hyperparameters,
        }, handle, indent=2)
    print("saved model to:", os.path.join(MODEL_BANK_DIR, version + ".pkl"))

    if promote:
        history = read_history() + [{"effective_from": snapshotdate, "version": version,
                                     "trained_through": through_str, "test_mae": round(test_mae, 3)}]
        with open(HISTORY_PATH, "w") as handle:
            json.dump(history, handle, indent=2)
        print("promoted to champion from", snapshotdate)
    else:
        print("not promoted, the champion stays")

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterSampler

# PURPOSE
# Monthly hyperparameter search for the model type that won validation. Every trial is scored on walk-forward folds
# that use only data BEFORE the test month, so the test month stays a clean score.
#   - fold k validates on one month and trains on every row before it (3 folds, the last one is the validation month)
#   - trial 0 = the current default settings, trial 1 = last month's best settings (if any), the rest are random
#   - selection: keep the incumbent (last month's best, else the defaults) unless another trial is better by more than
#     one standard error of its fold scores; this stops the settings flipping from month to month on noise

# parameter names are step name + __ + parameter name of the pipelines built in build_candidates()
SEARCH_SPACES = {
    "xgboost (poisson)": {
        "xgbregressor__learning_rate": loguniform(0.03, 0.10),
        "xgbregressor__n_estimators": randint(200, 501),
        "xgbregressor__max_depth": randint(4, 9),
        "xgbregressor__min_child_weight": loguniform(1, 20),
        "xgbregressor__subsample": uniform(0.7, 0.3),
        "xgbregressor__colsample_bytree": uniform(0.6, 0.4),
        "xgbregressor__reg_lambda": loguniform(0.5, 10),
        "xgbregressor__reg_alpha": loguniform(0.001, 1),
        "xgbregressor__gamma": uniform(0, 1),
    },
    "random forest": {
        "randomforestregressor__min_samples_leaf": randint(2, 30),
        "randomforestregressor__max_features": uniform(0.3, 0.7),
        "randomforestregressor__max_depth": [None, 12, 20, 30],
    },
    "poisson regression": {
        "poissonregressor__alpha": loguniform(0.001, 30),
    },
}
N_TRIALS = {"xgboost (poisson)": 12, "random forest": 5, "poisson regression": 8}   # random forest is slow, so fewer


def plain(params):
    # numpy numbers -> python numbers so the settings can be saved in the json
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}


def month_folds(df, last_month_start, n_folds=3):
    # walk-forward folds: validate on the month starting at last_month_start and the months before it,
    # each fold trains on every row before its validation month
    dates = pd.to_datetime(df["snapshot_date"]).to_numpy()
    folds = []
    month = pd.Timestamp(last_month_start)
    for _ in range(n_folds):
        next_month = month + pd.offsets.MonthBegin(1)
        train_idx = np.where(dates < month.to_datetime64())[0]
        valid_idx = np.where((dates >= month.to_datetime64()) & (dates < next_month.to_datetime64()))[0]
        if len(train_idx) > 0 and len(valid_idx) > 0:
            folds.append((train_idx, valid_idx))
        month = month - pd.offsets.MonthBegin(1)
    return folds


def tune(name, base_model, df, inputs, label, last_month_start, seed, last_best=None, n_folds=3):
    # returns the chosen settings and the score summary (mean MAE over the folds) for the defaults and the chosen trial
    folds = month_folds(df, last_month_start, n_folds)
    trials = [("default", {})]
    if last_best:
        trials.append(("last month", last_best))
    for params in ParameterSampler(SEARCH_SPACES[name], n_iter=N_TRIALS[name], random_state=seed):
        trials.append(("random", plain(params)))

    results = []
    for kind, params in trials:
        maes = []
        for train_idx, valid_idx in folds:
            model = clone(base_model).set_params(**params)
            model.fit(df.iloc[train_idx][inputs], df.iloc[train_idx][label])
            maes.append(mean_absolute_error(df.iloc[valid_idx][label], model.predict(df.iloc[valid_idx][inputs])))
            del model
        results.append({"kind": kind, "params": params, "mae": float(np.mean(maes)),
                        "se": float(np.std(maes) / np.sqrt(len(maes)))})
        print(f"  trial {len(results) - 1:2d} ({kind:<10}) tuning MAE {results[-1]['mae']:7.3f}")

    best = min(results, key=lambda r: r["mae"])
    incumbent = results[1] if last_best else results[0]
    chosen = incumbent if incumbent["mae"] <= best["mae"] + best["se"] else best
    summary = {"folds": len(folds), "trials": len(results), "default_mae": results[0]["mae"],
               "chosen_mae": chosen["mae"], "chosen_kind": chosen["kind"]}
    return chosen["params"], summary
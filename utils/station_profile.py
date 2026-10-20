# PURPOSE
# Adds two per-station features to a model: station_mean (average daily trips) and station_weekend_ratio
# (average weekend trips / average weekday trips). Both are learned from the rows it is fitted on only
# (fit), so validation and test months never leak into them, and then looked up for any rows (transform).
# Stations it has never seen get the overall average and a neutral ratio of 1.0.
# It is meant to be the first step of a pipeline: make_pipeline(StationProfile(), model)
# The input must contain the columns station_id and fe_is_weekend; station_id is dropped in the output.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class StationProfile(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        df = pd.DataFrame({
            "station_id": X["station_id"].astype(str).values,
            "is_weekend": X["fe_is_weekend"].values,
            "label": np.asarray(y),
        })
        mean = df.groupby("station_id")["label"].mean()
        weekend = df[df["is_weekend"] == 1].groupby("station_id")["label"].mean()
        weekday = df[df["is_weekend"] == 0].groupby("station_id")["label"].mean()
        ratio = (weekend / weekday).replace([np.inf, -np.inf], np.nan)
        self.profile_ = pd.DataFrame({"station_mean": mean, "station_weekend_ratio": ratio})
        self.overall_mean_ = float(df["label"].mean())
        return self

    def transform(self, X):
        X = X.copy()
        ids = X["station_id"].astype(str)
        X["station_mean"] = ids.map(self.profile_["station_mean"]).fillna(self.overall_mean_).values
        X["station_weekend_ratio"] = ids.map(self.profile_["station_weekend_ratio"]).fillna(1.0).values
        return X.drop(columns=["station_id"])
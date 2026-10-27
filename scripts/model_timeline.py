import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils.model_store import ModelStore
from utils.monitoring_store import monitoring_tables

# PURPOSE
# Model timeline for the report and dashboard: one row per monthly retrain from the MLflow training runs, with the test
# MAE of the retrained model vs the lag-7 baseline on the same test month, whether it was promoted, and (when
# model_explain.py has run) its top SHAP input and attribution shift. Also compares the same test months a year
# apart (Jun-Sep 2016 vs Jun-Sep 2017). Read-only on MLflow.
# Writes datamart/gold/monitoring/model_timeline.parquet (the table) and model_timeline.png (the chart).
#
# to call this script (from the repo root): python scripts/model_timeline.py

MONITORING_DIR = "datamart/gold/monitoring/"


def read_retrains(store):
    rows = []
    for saved in store.models():
        info = store.info(saved)
        rows.append({
            "retrained_on": info["retrained_on"],
            "model_version": saved.version,
            "source_run_id": saved.run_id,
            "model": info["model"],
            "test_month": info["test_window"][0][:7],
            "test_model_mae": info["test_model_mae"],
            "test_baseline_mae": info["test_baseline_mae"],
            "champion_test_mae": info.get("champion_test_mae"),
            "promoted": bool(info["promoted"]),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("retrained_on").reset_index(drop=True)
    df["vs_baseline"] = ((1 - df["test_model_mae"] / df["test_baseline_mae"]) * 100).round(1)
    return df


def add_shap(df, store):
    tables = monitoring_tables(store, "shap")
    if not tables:
        return df
    shap = pd.concat(tables, ignore_index=True)
    top = shap[shap["rank"] == 1][["source_run_id", "feature", "share", "attribution_shift"]].rename(
        columns={"feature": "top_shap_feature", "share": "top_shap_share"})
    return df.merge(top, on="source_run_id", how="left", validate="one_to_one")


def plot(df, path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = pd.to_datetime(df["retrained_on"])
    ax.plot(x, df["test_baseline_mae"], color="grey", linestyle="--", marker="s", markersize=4, label="lag-7 baseline")
    ax.plot(x, df["test_model_mae"], color="tab:blue", label="retrained model")
    promoted = df["promoted"]
    ax.scatter(x[promoted], df.loc[promoted, "test_model_mae"], color="tab:blue", s=60, zorder=3, label="promoted to champion")
    ax.scatter(x[~promoted], df.loc[~promoted, "test_model_mae"], facecolors="white", edgecolors="tab:blue", s=60,
               zorder=3, label="not promoted (champion kept)")
    for xi, yi, model in zip(x, df["test_model_mae"], df["model"]):
        if model != "xgboost (poisson)":
            ax.annotate(model, (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_title("Test MAE of each monthly retrain (on the month before the retrain)")
    ax.set_xlabel("retrained on")
    ax.set_ylabel("MAE (trips per station per day)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    store = ModelStore()
    df = read_retrains(store)
    if df.empty:
        print("no completed published training runs in MLflow")
        return
    df = add_shap(df, store)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(len(df), "retrains,", int(df["promoted"].sum()), "promoted\n")
    print(df.drop(columns=["model_version"]).to_string(index=False))

    # the same test months a year apart
    df["month"] = df["test_month"].str[5:]
    df["year"] = df["test_month"].str[:4]
    same = df[df["month"].isin(["06", "07", "08", "09"])]
    if same["year"].nunique() > 1:
        print("\nsame test months a year apart (Jun-Sep): test MAE of the retrained model and of the baseline")
        print(same.pivot_table(index="month", columns="year", values=["test_model_mae", "test_baseline_mae"]).round(2).to_string())

    os.makedirs(MONITORING_DIR, exist_ok=True)
    table_path = os.path.join(MONITORING_DIR, "model_timeline.parquet")
    df.drop(columns=["month", "year"]).to_parquet(table_path, index=False)
    chart_path = os.path.join(MONITORING_DIR, "model_timeline.png")
    plot(df, chart_path)
    print("\nsaved to:", table_path, "and", chart_path)


if __name__ == "__main__":
    main()

import glob
import os

import pandas as pd

# PURPOSE
# Summary of the daily monitoring rows in datamart/gold/monitoring/ (written by model_monitor_daily.py), used to check
# the alert pattern after a backfill and for the report. Read-only, writes nothing.
#   1. month by month: MAE vs baseline, alert days, drift days, data quality warnings, model in production
#   2. MAE by segment: busy vs quiet stations, weekday vs weekend, holiday vs not, rain day vs dry day
#   3. the same months a year apart (Jun-Sep 2016 vs Jun-Sep 2017)
#
# to call this script (from the repo root): python scripts/model_monitor_summary.py

MONITORING_DIR = "datamart/gold/monitoring/"


def segment_table(df):
    # mean daily MAE of the model and the baseline for each segment
    df = df.copy()
    for column in ("is_weekend", "is_holiday", "is_rain_day"):
        df[column] = df[column].astype("boolean")
    rows = []
    for name, mask in [("weekday", ~df["is_weekend"]), ("weekend", df["is_weekend"]),
                       ("not a holiday", ~df["is_holiday"]), ("holiday", df["is_holiday"]),
                       ("dry day", ~df["is_rain_day"]), ("rain day", df["is_rain_day"])]:
        part = df[mask]
        rows.append({"segment": name, "days": len(part), "model_mae": part["model_mae"].mean(),
                     "baseline_mae": part["baseline_mae"].mean()})
    rows.append({"segment": "busy stations (top 20%)", "days": len(df), "model_mae": df["busy_stations_mae"].mean(),
                 "baseline_mae": None})
    rows.append({"segment": "quiet stations", "days": len(df), "model_mae": df["quiet_stations_mae"].mean(),
                 "baseline_mae": None})
    table = pd.DataFrame(rows).set_index("segment")
    table["vs_baseline"] = ((1 - table["model_mae"] / table["baseline_mae"]) * 100).round(1)
    return table.round({"model_mae": 2, "baseline_mae": 2})


def main():
    paths = sorted(glob.glob(os.path.join(MONITORING_DIR, "model_monitoring_*.parquet")))
    if len(paths) == 0:
        print("no monitoring rows found in", MONITORING_DIR)
        return
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["month"] = pd.to_datetime(df["monitored_date"]).dt.strftime("%Y-%m")
    print(len(df), "monitored days from", df["monitored_date"].min(), "to", df["monitored_date"].max(), "\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)

    # 1. month by month
    summary = df.groupby("month").agg(
        days=("monitored_date", "count"),
        scored_days=("model_mae", "count"),
        model_mae=("model_mae", "mean"),
        baseline_mae=("baseline_mae", "mean"),
        expected_mae=("expected_mae", "last"),
        ok=("performance_status", lambda s: int((s == "ok").sum())),
        warning=("performance_status", lambda s: int((s == "warning").sum())),
        critical=("performance_status", lambda s: int((s == "critical").sum())),
        drift_significant=("drift_status", lambda s: int((s == "significant").sum())),
        max_psi=("max_psi", "max"),
        dq_warnings=("data_quality_status", lambda s: int((s != "ok").sum())),
        model_version=("model_version", "last"),
    )
    summary["vs_baseline"] = ((1 - summary["model_mae"] / summary["baseline_mae"]) * 100).round(1).astype(str) + "%"
    summary["model_version"] = summary["model_version"].str.replace("_trained_through_", " to ")
    print(summary.round({"model_mae": 2, "baseline_mae": 2, "expected_mae": 2, "max_psi": 3}).to_string())

    print("\nwhole period | model MAE", round(df["model_mae"].mean(), 2), "| baseline MAE", round(df["baseline_mae"].mean(), 2),
          "| model RMSE", round(df["model_rmse"].mean(), 2), "| baseline RMSE", round(df["baseline_rmse"].mean(), 2),
          "\n             | days ok / warning / critical:", (df["performance_status"] == "ok").sum(), "/",
          (df["performance_status"] == "warning").sum(), "/", (df["performance_status"] == "critical").sum(),
          "| days with significant drift:", (df["drift_status"] == "significant").sum(),
          "| data quality warnings:", (df["data_quality_status"] != "ok").sum())
    for reason, count in df.loc[df["data_quality_status"] != "ok", "data_quality_reason"].value_counts().items():
        print("  data quality:", count, "day(s) with", reason)

    # 2. segments
    print("\nMAE by segment (mean of the daily MAEs)")
    print(segment_table(df).to_string())

    # 3. the same months a year apart
    same_months = df[pd.to_datetime(df["monitored_date"]).dt.month.isin([6, 7, 8, 9])]
    if same_months["month"].str[:4].nunique() > 1:
        print("\nsame months a year apart (Jun-Sep)")
        year_table = same_months.groupby([same_months["month"].str[5:].rename("month"), same_months["month"].str[:4].rename("year")]) \
            [["model_mae", "baseline_mae"]].mean().unstack("year").round(2)
        print(year_table.to_string())


if __name__ == "__main__":
    main()

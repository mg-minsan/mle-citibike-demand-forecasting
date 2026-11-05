import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# PURPOSE
# Charts of the monitoring results for the report and slides. Read-only on the monitoring outputs:
#   1. monitoring_mae.png       7-day rolling MAE of the model vs the lag-7 baseline, the warning line, warning and
#                               critical days shaded, and the days the production model changed
#   2. monitoring_drift.png     PSI drift over time (the highest PSI of the 5 trip-history features each day)
#   3. monitoring_segments.png  mean daily MAE of the model and the baseline per segment (bias drift)
#   4. monitoring_shap.png      SHAP share of the top inputs per retrain, and the attribution shift vs the previous retrain
# Reads datamart/gold/monitoring/model_monitoring_*.parquet (model_monitor_daily.py) and
# datamart/gold/monitoring/shap/model_shap_*.parquet (model_explain.py). Writes the PNGs to datamart/gold/monitoring/charts/.
#
# to call this script (from the repo root): python scripts/model_monitor_charts.py

MONITORING_DIR = "datamart/gold/monitoring/"
SHAP_DIR = os.path.join(MONITORING_DIR, "shap/")
CHART_DIR = os.path.join(MONITORING_DIR, "charts/")

MODEL_COLOR = "#2a78d6"        # same blue as model_timeline.png
BASELINE_COLOR = "#898781"     # grey, dashed, as in model_timeline.png
WARNING_COLOR = "#fab219"
CRITICAL_COLOR = "#d03b3b"
INK = "#52514e"
TOP_SHAP_FEATURES = 8          # rows of the SHAP heatmap
SHIFT_WARNING = 0.2            # same threshold as model_explain.py
PSI_MODERATE = 0.1             # same thresholds as utils/model_monitoring.py
PSI_SIGNIFICANT = 0.25

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})


def read_monitoring():
    paths = sorted(glob.glob(os.path.join(MONITORING_DIR, "model_monitoring_*.parquet")))
    if len(paths) == 0:
        return None
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["monitored_date"] = pd.to_datetime(df["monitored_date"])
    return df.sort_values("monitored_date").reset_index(drop=True)


def short_model(version):
    # "xgboost_poisson_trained_through_2016-06-30" -> "XGB", "random_forest_..." -> "RF"
    for prefix, name in [("xgboost", "XGB"), ("random_forest", "RF"), ("poisson", "Poisson")]:
        if version.startswith(prefix):
            return name
    return version.split("_")[0]


def model_changes(df):
    # the days the production model changed (the first day of each new model_version)
    changed = df["model_version"] != df["model_version"].shift()
    return df.loc[changed, ["monitored_date", "model_version"]].iloc[1:]


def shade_status(ax, df):
    # one band per warning / critical day
    for _, row in df[df["performance_status"] != "ok"].iterrows():
        color = CRITICAL_COLOR if row["performance_status"] == "critical" else WARNING_COLOR
        ax.axvspan(row["monitored_date"] - pd.Timedelta(hours=12), row["monitored_date"] + pd.Timedelta(hours=12),
                   color=color, alpha=0.25 if color == WARNING_COLOR else 0.35, linewidth=0, zorder=0)


def mark_model_changes(ax, df, y_text):
    for _, row in model_changes(df).iterrows():
        ax.axvline(row["monitored_date"], color=INK, linewidth=0.6, linestyle=":", zorder=1)
        ax.text(row["monitored_date"], y_text, " " + short_model(row["model_version"]), fontsize=7, color=INK,
                va="top", ha="left")


def date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)


def chart_mae(df, path):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    shade_status(ax, df)
    ax.plot(df["monitored_date"], df["rolling_baseline_mae"], color=BASELINE_COLOR, linestyle="--", linewidth=1.5,
            label="lag-7 baseline (7-day MAE)")
    ax.plot(df["monitored_date"], df["rolling_model_mae"], color=MODEL_COLOR, linewidth=2, label="model (7-day MAE)")
    ax.step(df["monitored_date"], df["warning_threshold_mae"], where="post", color=INK, linewidth=0.9,
            label="warning line (1.2 x expected MAE)")
    top = max(df["rolling_baseline_mae"].max(), df["rolling_model_mae"].max()) * 1.12
    ax.set_ylim(0, top)
    mark_model_changes(ax, df, top * 0.99)
    # legend entries for the shaded days
    ax.fill_between([], [], color=WARNING_COLOR, alpha=0.25, label="warning day")
    ax.fill_between([], [], color=CRITICAL_COLOR, alpha=0.35, label="critical day")
    counts = df["performance_status"].value_counts()
    ax.set_title(f"Daily model monitoring: 7-day rolling MAE ({counts.get('ok', 0)} ok, {counts.get('warning', 0)} warning, "
                 f"{counts.get('critical', 0)} critical days)", fontsize=10, loc="left")
    ax.set_ylabel("MAE (trips per station per day)")
    date_axis(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=5, fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def chart_drift(df, path):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axhspan(PSI_SIGNIFICANT, 100, color=CRITICAL_COLOR, alpha=0.07, linewidth=0, zorder=0)
    ax.axhspan(PSI_MODERATE, PSI_SIGNIFICANT, color=WARNING_COLOR, alpha=0.12, linewidth=0, zorder=0)
    ax.plot(df["monitored_date"], df["max_psi"].clip(lower=0.01), color=MODEL_COLOR, linewidth=1.8,
            label="highest PSI of the 5 trip-history features")
    ax.set_yscale("log")
    ticks = [0.05, 0.1, 0.25, 0.5, 1, 2]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    low = min(0.04, df["max_psi"].min() * 0.8)
    high = max(2.5, df["max_psi"].max() * 1.3)
    ax.set_ylim(low, high)
    # threshold labels just outside the right edge, so they never sit on the line
    edge = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.01, PSI_SIGNIFICANT * 1.6, "significant\n> 0.25", transform=edge, fontsize=7, color=INK, va="center")
    ax.text(1.01, (PSI_MODERATE * PSI_SIGNIFICANT) ** 0.5, "moderate\n0.1-0.25", transform=edge, fontsize=7, color=INK, va="center")
    ax.text(1.01, PSI_MODERATE * 0.6, "no change\n< 0.1", transform=edge, fontsize=7, color=INK, va="center")
    mark_model_changes(ax, df, high * 0.95)
    significant = int((df["drift_status"] == "significant").sum())
    ax.set_title(f"Feature drift: PSI of the last 30 days vs the production model's training window "
                 f"({significant} of {len(df)} days significant)", fontsize=10, loc="left")
    ax.set_ylabel("PSI (log scale)")
    date_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def chart_segments(df, path):
    groups = [("Weekday", ~df["is_weekend"]), ("Weekend", df["is_weekend"]),
              ("Not a holiday", ~df["is_holiday"]), ("Holiday", df["is_holiday"]),
              ("Dry day", ~df["is_rain_day"]), ("Rain day", df["is_rain_day"])]
    names, model, baseline, days = [], [], [], []
    for name, mask in groups:
        names.append(name)
        model.append(df.loc[mask, "model_mae"].mean())
        baseline.append(df.loc[mask, "baseline_mae"].mean())
        days.append(int(mask.sum()))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [3, 1]})
    x = np.arange(len(names))
    width = 0.38
    ax.bar(x - width / 2 - 0.01, model, width, color=MODEL_COLOR, label="model")
    ax.bar(x + width / 2 + 0.01, baseline, width, color=BASELINE_COLOR, label="lag-7 baseline")
    for xi, m, b in zip(x, model, baseline):
        ax.text(xi - width / 2, m, f"{m:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.text(xi + width / 2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.text(xi, max(m, b) * 1.12, f"{(1 - m / b) * 100:.0f}% better", ha="center", fontsize=7.5, color="#0b0b0b")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({d} days)" for n, d in zip(names, days)], fontsize=7.5)
    ax.set_ylim(0, max(baseline) * 1.3)
    ax.set_ylabel("mean daily MAE (trips)")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2)
    ax.set_title("MAE by type of day", fontsize=10, loc="left")

    # busy vs quiet stations: the model only (the baseline is not split by station size)
    station = [df["busy_stations_mae"].mean(), df["quiet_stations_mae"].mean()]
    ax2.bar([0, 1], station, 0.55, color=MODEL_COLOR)
    for xi, v in zip([0, 1], station):
        ax2.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Busy\n(top 20%)", "Other\nstations"], fontsize=7.5)
    ax2.set_ylim(0, max(station) * 1.25)
    ax2.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.set_title("Model MAE by station size", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def chart_shap(path):
    paths = sorted(glob.glob(os.path.join(SHAP_DIR, "model_shap_*.parquet")))
    if len(paths) == 0:
        print("no SHAP tables in", SHAP_DIR, "- skipping the SHAP chart (run model_explain.py first)")
        return False
    shap = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    shap["retrained_on"] = pd.to_datetime(shap["retrained_on"])
    retrains = shap.drop_duplicates("retrained_on").sort_values("retrained_on")
    top = shap.groupby("feature")["share"].mean().sort_values(ascending=False).head(TOP_SHAP_FEATURES).index
    grid = shap[shap["feature"].isin(top)].pivot_table(index="feature", columns="retrained_on", values="share") \
        .reindex(index=top, columns=retrains["retrained_on"]).fillna(0)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 4.8), gridspec_kw={"height_ratios": [3, 1.2]}, sharex=True)
    image = ax.imshow(grid.values * 100, aspect="auto", cmap="Blues", vmin=0, vmax=max(45, grid.values.max() * 100))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.values[i, j] * 100
            if value >= 5:
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if value >= 25 else "#0b0b0b")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top, fontsize=7.5)
    ax.tick_params(length=0)
    for side in ["left", "bottom"]:
        ax.spines[side].set_visible(False)
    bar = fig.colorbar(image, ax=[ax, ax2], fraction=0.025, pad=0.01)
    bar.set_label("share of SHAP importance (%)", fontsize=7.5)
    bar.ax.tick_params(labelsize=7)
    ax.set_title("SHAP share of the top inputs per monthly retrain (on its test month)", fontsize=10, loc="left")

    shift = retrains["attribution_shift"].values
    colors = [CRITICAL_COLOR if s > SHIFT_WARNING else MODEL_COLOR for s in np.nan_to_num(shift)]
    ax2.bar(range(len(shift)), np.nan_to_num(shift), 0.6, color=colors)
    ax2.axhline(SHIFT_WARNING, color=INK, linewidth=0.8, linestyle="--")
    ax2.text(len(shift) - 0.5, SHIFT_WARNING, "alert > 0.2", fontsize=7, color=INK, ha="right", va="bottom")
    ax2.set_ylabel("attribution\nshift", fontsize=8)
    ax2.set_ylim(0, max(0.3, np.nanmax(shift) * 1.2) if np.isfinite(np.nanmax(shift)) else 0.3)
    ax2.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    ax2.set_axisbelow(True)
    labels = [d.strftime("%b %y") + "\n" + short_model(v) + ("*" if p else "")
              for d, v, p in zip(retrains["retrained_on"], retrains["model_version"], retrains["promoted"])]
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=6.5)
    ax2.set_xlabel("retrained on (model type, * = promoted to production)", fontsize=7.5)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    df = read_monitoring()
    if df is None:
        print("no monitoring rows found in", MONITORING_DIR, "- run model_monitor_daily.py first")
        return
    os.makedirs(CHART_DIR, exist_ok=True)
    print(len(df), "monitored days from", df["monitored_date"].min().date(), "to", df["monitored_date"].max().date())

    saved = []
    for name, chart in [("monitoring_mae.png", chart_mae), ("monitoring_drift.png", chart_drift),
                        ("monitoring_segments.png", chart_segments)]:
        chart(df, os.path.join(CHART_DIR, name))
        saved.append(name)
    if chart_shap(os.path.join(CHART_DIR, "monitoring_shap.png")):
        saved.append("monitoring_shap.png")
    for name in saved:
        print("saved to:", os.path.join(CHART_DIR, name))


if __name__ == "__main__":
    main()

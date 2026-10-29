"""Business dashboard for the daily prediction CSV exports."""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["station_id", "station_name", "model_prediction", "date"]


def read_predictions(path, expected_date):
    frame = pd.read_csv(path, dtype={"station_id": "string", "station_name": "string"})
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    frame = frame[COLUMNS].copy()
    if frame.empty:
        raise ValueError("The CSV contains no predictions.")
    if frame.isna().any().any():
        raise ValueError("The CSV contains missing values.")
    for column in ("station_id", "station_name"):
        frame[column] = frame[column].str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"The CSV contains blank {column} values.")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    if not frame["date"].eq(expected_date).all():
        raise ValueError("CSV dates do not match the selected forecast date.")
    if frame["station_id"].duplicated().any():
        raise ValueError("The CSV contains duplicate station IDs.")
    frame["model_prediction"] = pd.to_numeric(frame["model_prediction"], errors="raise")
    if not np.isfinite(frame["model_prediction"]).all() or frame["model_prediction"].lt(0).any():
        raise ValueError("Predictions must be finite, nonnegative numbers.")
    return frame.sort_values("model_prediction", ascending=False)


def main():
    st.set_page_config(page_title="Citi Bike | Demand forecast", page_icon="🚲", layout="wide")
    styles = Path(__file__).with_name("styles.css").read_text()
    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)
    st.markdown('''<section class="forecast-hero">
        <div class="eyebrow">New York City · Demand planning</div>
        <h1>Ahead of the next ride.</h1>
        <p>Your daily view of Citi Bike station demand. Explore forecasts,
        spot the busiest stations, and plan for the day ahead.</p>
        </section>''', unsafe_allow_html=True)

    directory = Path(os.getenv("PREDICTIONS_CSV_DIR", str(ROOT / "datamart/business/predictions")))
    if not directory.is_absolute():
        directory = ROOT / directory
    st.sidebar.markdown('<div class="bike-brand"><span>citi</span> bike</div>'
                        '<div class="brand-caption">Station demand dashboard</div>', unsafe_allow_html=True)
    st.sidebar.title("Plan your view")
    st.sidebar.button("Refresh forecasts", help="Reload the latest batch exports from disk.")
    available = {}
    for path in sorted(directory.glob("predictions_*.csv")):
        try:
            day = datetime.strptime(path.stem, "predictions_%Y_%m_%d").date().isoformat()
        except ValueError:
            continue
        available[day] = path
    if not available:
        st.info("No prediction CSVs are available yet. Export a prediction batch to get started.")
        st.code("python scripts/export_predictions_csv.py --snapshotdate 2017-03-22", language="bash")
        st.caption(f"Looking in: {directory}")
        return

    day = st.sidebar.selectbox("Forecast date", sorted(available, reverse=True))
    try:
        frame = read_predictions(available[day], day)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Unable to load {available[day].name}: {exc}")
        return

    options = frame.sort_values(["station_name", "station_id"])
    labels = dict(zip(options["station_id"], options["station_name"] + " · " + options["station_id"]))
    selected = st.sidebar.multiselect("Stations", list(labels), format_func=labels.get,
                                      placeholder="All stations", help="Search by station name or ID.")
    filtered = frame[frame["station_id"].isin(selected)] if selected else frame
    st.sidebar.caption(f"{len(filtered):,} of {len(frame):,} stations selected")
    st.sidebar.caption("Forecasts are model estimates of trip starts, not live bike availability.")

    st.caption("DAILY FORECAST")
    st.subheader(datetime.fromisoformat(day).strftime("%A, %d %B %Y"))
    total, stations, average = st.columns(3)
    total.metric("Predicted trip starts", f"{filtered['model_prediction'].sum():,.1f}")
    stations.metric("Stations", f"{len(filtered):,}")
    average.metric("Average per station", f"{filtered['model_prediction'].mean():,.1f}")
    st.caption("Your selected stations · Expected trip starts for the forecast date")

    top = filtered.head(15).copy()
    top["Station"] = top["station_name"] + " · " + top["station_id"]
    chart, insight = st.columns([3, 1])
    with chart:
        with st.container(border=True):
            st.subheader("Where demand is heading")
            st.caption(f"Top {len(top)} stations by predicted trip starts")
            st.bar_chart(top, x="Station", y="model_prediction", horizontal=True,
                         sort="-model_prediction", color="#0074C8", height=440,
                         x_label="Predicted trip starts", y_label="Station")
    with insight:
        with st.container(border=True):
            st.caption("STATION SPOTLIGHT")
            busiest = filtered.iloc[0]
            st.subheader(busiest["station_name"])
            st.caption(f"Station {busiest['station_id']}")
            st.write(f"**{busiest['model_prediction']:,.0f} predicted trip starts**")
            st.divider()
            total_trips = filtered["model_prediction"].sum()
            share = busiest["model_prediction"] / total_trips if total_trips else 0
            st.write(f"This station accounts for **{share:.1%}** of predicted demand in your selection.")
            st.caption("Use the station filter to focus on a specific part of the network.")

    st.subheader("Every station. One clear view.")
    st.caption("Browse the forecast or download your selected stations for planning.")
    st.download_button("Download selected predictions", filtered[COLUMNS].to_csv(index=False).encode("utf-8"),
                       file_name=f"predictions_{day.replace('-', '_')}_selected.csv", mime="text/csv")
    st.dataframe(filtered[COLUMNS], hide_index=True, width="stretch",
                 column_config={
                     "date": st.column_config.TextColumn("Forecast date"),
                     "model_prediction": st.column_config.NumberColumn("Predicted trip starts", format="%.1f"),
                     "station_id": st.column_config.TextColumn("Station ID"),
                     "station_name": st.column_config.TextColumn("Station name"),
                 })
    st.caption(f"Source: {available[day].name} · Downloads preserve the values in the business CSV.")


if __name__ == "__main__":
    main()

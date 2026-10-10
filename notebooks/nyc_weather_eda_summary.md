# NYC weather EDA and daily features — 2016–2017

The executed analysis is in [`nyc_weather_eda.ipynb`](nyc_weather_eda.ipynb). Input: `data/NYC_Weather_2016_2022.csv`, filtered to 2016 and 2017 before all analysis. The source file is unchanged.

## Coverage and quality

- 17,544 hourly records covering January 1, 2016 through December 31, 2017.
- No duplicate timestamps, missing hourly timestamps, or missing weather measurements.
- No violations of the notebook's broad physical bounds; rain never exceeds precipitation.
- All plots and numerical summaries now use only 2016–2017.

## Daily output

**Path:** `data/processed/nyc_weather_daily_2016_2017.csv`

**Shape:** 731 rows × 10 columns: 366 days in 2016 and 365 days in 2017.

| Column | Aggregation |
|---|---|
| date | Supplied-clock calendar date, YYYY-MM-DD |
| temperature_mean_c | Mean hourly temperature |
| temperature_min_c | Minimum hourly temperature |
| temperature_max_c | Maximum hourly temperature |
| precipitation_total_mm | Sum of hourly precipitation |
| rain_total_mm | Sum of hourly rain |
| wet_hours | Count of hours with precipitation > 0 |
| cloud_mean_pct | Mean hourly total cloud cover |
| wind_mean_kmh | Mean hourly wind speed |
| wind_max_kmh | Maximum hourly wind speed, not a gust measurement |

Assertions verify complete hourly coverage, unique daily dates, absence of missing daily features, and preservation of precipitation/rain totals. Annual precipitation sums are 898.5 mm for 2016 and 1,118.8 mm for 2017.

Rain is optional because it may be redundant with total precipitation. Wind direction is omitted from daily aggregation because an ordinary mean is inappropriate for circular angles.

## Joining and forecasting caveats

- The notebook includes a daily join template with `validate="one_to_one"` and unmatched-date checks.
- Confirm timezone and day boundaries before joining. The source timestamps are timezone-naive, and aggregation uses their supplied clock; no timezone conversion is assumed.
- Confirm precipitation/rain are hourly increments before interpreting their sums as daily totals.
- Use chronological validation and train-only preprocessing. Full-day realized weather cannot be used to forecast that day before those observations are available; use available weather forecasts instead.
- This is weather-only analysis, not an estimate of weather–demand effects.

## Reproduce

Run all cells in `notebooks/nyc_weather_eda.ipynb` with pandas, NumPy, Matplotlib, seaborn, and a Jupyter Python kernel. The notebook recreates the daily CSV. Generated data lives under the repository's ignored `data/` directory.

Validation: all code cells executed successfully, plots are embedded, the notebook schema validated, and the exported CSV passed date, row-count, missingness, and annual-total checks.

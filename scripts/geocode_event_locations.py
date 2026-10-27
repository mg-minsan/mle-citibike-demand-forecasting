"""Geocode relevant 2016–2017 event locations with resumable GeoSearch requests.

Run a sample: python scripts/geocode_event_locations.py --limit 100
Resume all:   python scripts/geocode_event_locations.py
"""

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
from pathlib import Path
import sqlite3
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://geosearch.planninglabs.nyc/v2/search"
RELEVANT_EVENT_TYPES = [
    "Athletic Race / Tour", "Parade", "Street Festival", "Single Block Festival",
    "Block Party", "Street Event", "Play Streets", "Farmers Market",
]
OUTPUT_COLUMNS = [
    "event_location", "borough", "event_count", "query", "status", "latitude",
    "longitude", "matched_label", "matched_borough", "postalcode", "confidence",
    "layer", "candidate_count", "needs_review", "review_reason", "error",
]


def load_locations(path):
    """Use the raw events so boroughs and the requested filters are preserved."""
    columns = ["Start Date/Time", "Event Type", "Event Location", "Event Borough"]
    events = pd.read_csv(path, usecols=columns, dtype=str)
    starts = pd.to_datetime(
        events["Start Date/Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    events = events.loc[
        starts.dt.year.isin([2016, 2017])
        & events["Event Type"].str.strip().isin(RELEVANT_EVENT_TYPES)
    ].copy()
    events["Event Location"] = events["Event Location"].fillna("").str.strip()
    events["Event Borough"] = events["Event Borough"].fillna("").str.strip()
    events = events.loc[events["Event Location"].ne("")]
    return events.groupby(["Event Location", "Event Borough"]).size().reset_index(
        name="event_count"
    ).rename(columns={"Event Location": "event_location", "Event Borough": "borough"})


def retry_delay(value, attempt):
    """Honor Retry-After in either seconds or HTTP-date form."""
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError):
                pass
    return min(60, 2 ** attempt)


class GeoSearchClient:
    def __init__(self, interval=1.0, timeout=30, retries=3):
        self.interval, self.timeout, self.retries = interval, timeout, retries
        self.last_request = None

    def search(self, query):
        for attempt in range(self.retries + 1):
            if self.last_request is not None:
                time.sleep(max(0, self.interval - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            request = Request(
                ENDPOINT + "?" + urlencode({"text": query, "size": 3}),
                headers={"User-Agent": "citibike-event-geocoder/1.0", "Accept": "application/json"},
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                if not isinstance(payload, dict) or not isinstance(payload.get("features"), list):
                    raise ValueError("Invalid GeoJSON response")
                return payload
            except HTTPError as exc:
                if exc.code in (401, 403):
                    raise RuntimeError(f"GeoSearch returned HTTP {exc.code}; stopping. Progress is saved.") from exc
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if exc.code == 429 and attempt == self.retries:
                    raise RuntimeError("GeoSearch is rate limiting requests; stopping. Resume later.") from exc
                if retryable and attempt < self.retries:
                    time.sleep(retry_delay(exc.headers.get("Retry-After"), attempt))
                    continue
                return {"error": f"HTTP {exc.code}: {exc.reason}"}
            except (URLError, TimeoutError, OSError, ValueError) as exc:
                if attempt < self.retries:
                    time.sleep(retry_delay(None, attempt))
                    continue
                return {"error": str(exc)}


def summarize(payload, location, borough, query):
    row = {"query": query, "status": "no_match", "needs_review": True,
           "review_reason": "No results"}
    if "error" in payload:
        return {**row, "status": "error", "error": payload["error"], "review_reason": "Request failed"}
    features = payload.get("features", [])
    if not features:
        return row
    feature = features[0]
    props, geometry = feature.get("properties", {}), feature.get("geometry", {})
    coordinates = geometry.get("coordinates", [])
    if geometry.get("type") != "Point" or len(coordinates) < 2:
        return {**row, "status": "error", "error": "Top candidate has no point coordinates"}
    try:
        longitude, latitude = map(float, coordinates[:2])
        if not (math.isfinite(latitude) and math.isfinite(longitude)
                and -90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("Invalid coordinates")
    except (TypeError, ValueError) as exc:
        return {**row, "status": "error", "error": str(exc)}
    reasons = []
    if not borough or props.get("borough", "").casefold() != borough.casefold():
        reasons.append("Borough missing or different")
    if "," in location or " between " in location.casefold():
        reasons.append("Multi-location or street segment: point may be approximate")
    confidence = props.get("confidence")
    if not isinstance(confidence, (int, float)) or confidence < 0.8:
        reasons.append("Confidence missing or below 0.8")
    if props.get("layer") not in ("address", "venue"):
        reasons.append("Result is not an address or venue")
    return {
        "query": query, "status": "candidate", "latitude": latitude, "longitude": longitude,
        "matched_label": props.get("label", ""), "matched_borough": props.get("borough", ""),
        "postalcode": props.get("postalcode", ""), "confidence": confidence,
        "layer": props.get("layer", ""), "candidate_count": len(features),
        "needs_review": bool(reasons), "review_reason": "; ".join(reasons),
    }


def export_results(connection, locations, output):
    cached = {(loc, borough): json.loads(result) for loc, borough, result in connection.execute(
        "SELECT location, borough, result FROM geocodes"
    )}
    rows = []
    for item in locations.to_dict("records"):
        result = cached.get((item["event_location"], item["borough"]))
        if result is not None:
            rows.append({**item, **result})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(temporary, index=False)
    temporary.replace(output)
    print(f"Exported {len(rows):,} saved results to {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/NYC_Permitted_Event_Information_-_Historical_20260927.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/event_locations_geocoded.csv")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/processed/event_geocoding.sqlite3")
    parser.add_argument("--limit", type=int, help="Maximum new location queries this run (e.g. 100).")
    parser.add_argument("--interval", type=float, default=1.0, help="Minimum seconds between requests (default 1).")
    args = parser.parse_args(argv)
    if args.interval < 1 or not math.isfinite(args.interval) or (args.limit is not None and args.limit < 1):
        parser.error("--interval must be at least 1 second; --limit must be positive.")
    if len({args.input.resolve(), args.output.resolve(), args.cache.resolve()}) != 3:
        parser.error("Input, output, and cache paths must be different.")
    if not args.input.is_file():
        parser.error(f"Source event CSV not found: {args.input}")
    locations = load_locations(args.input)
    print(f"{len(locations):,} unique location + borough pairs for relevant 2016–2017 events.", flush=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.cache)
    connection.execute("""CREATE TABLE IF NOT EXISTS geocodes (
        location TEXT, borough TEXT, result TEXT NOT NULL, response TEXT NOT NULL,
        updated_at TEXT NOT NULL, PRIMARY KEY (location, borough))""")
    client, queried = GeoSearchClient(interval=args.interval), 0
    failure = None
    try:
        for item in locations.to_dict("records"):
            location, borough = item["event_location"], item["borough"]
            saved = connection.execute(
                "SELECT result FROM geocodes WHERE location=? AND borough=?", (location, borough)
            ).fetchone()
            if saved and json.loads(saved[0])["status"] != "error":
                continue
            if args.limit is not None and queried >= args.limit:
                break
            query = f"{location}, {borough}, New York, NY" if borough else f"{location}, New York, NY"
            payload = client.search(query)
            result = summarize(payload, location, borough, query)
            with connection:
                connection.execute("INSERT OR REPLACE INTO geocodes VALUES (?, ?, ?, ?, ?)", (
                    location, borough, json.dumps(result), json.dumps(payload),
                    datetime.now(timezone.utc).isoformat(),
                ))
            queried += 1
            print(f"[{queried}] {result['status']}: {location}", flush=True)
            if queried % 100 == 0:
                export_results(connection, locations, args.output)
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to resume.", flush=True)
    except RuntimeError as exc:
        failure = str(exc)
    finally:
        try:
            export_results(connection, locations, args.output)
        finally:
            connection.close()
    if failure:
        parser.exit(1, failure + "\n")


if __name__ == "__main__":
    main()

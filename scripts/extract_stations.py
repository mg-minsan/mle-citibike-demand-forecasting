"""Extract a station lookup from both endpoints of historical NYC trip CSVs."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def extract_stations(input_dir=Path("data"), output=Path("data/stations.csv")):
    input_dir, output = Path(input_dir), Path(output)
    files = sorted(p for p in input_dir.glob("*-citibike-tripdata/*/*.csv")
                   if not p.name.startswith(("JC", "._")))
    if not files:
        raise FileNotFoundError(f"No NYC trip CSVs in {input_dir}")
    if output.resolve() in {p.resolve() for p in files}:
        raise ValueError("Output must not overwrite a source trip CSV")
    variants = defaultdict(Counter)
    for path in files:
        print(f"Reading {path}", flush=True)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            columns = {name.lower().replace(" ", "").replace("_", ""): i
                       for i, name in enumerate(next(reader))}
            endpoints = [tuple(columns[side + "station" + field]
                               for field in ("id", "name", "latitude", "longitude"))
                         for side in ("start", "end")]
            for row in reader:
                if not row:
                    continue
                for id_col, name_col, lat_col, lon_col in endpoints:
                    station_id, name = row[id_col].strip(), row[name_col].strip()
                    if station_id.lower() in ("", "null", "nan") or not name:
                        continue
                    variants[station_id][(name, row[lat_col], row[lon_col])] += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["station_id", "station_name", "latitude", "longitude"])
        for station_id, records in sorted(variants.items()):
            # Keep a consistent name/coordinate combination when stations changed.
            # Ties are broken lexicographically for reproducible results.
            record = max(records, key=lambda value: (records[value], value))
            writer.writerow((station_id, *record))
    print(f"Saved {len(variants):,} stations to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/stations.csv"))
    args = parser.parse_args()
    extract_stations(args.input_dir, args.output)

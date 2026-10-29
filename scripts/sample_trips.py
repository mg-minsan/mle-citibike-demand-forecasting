"""Stream raw Citi Bike CSVs and select a reproducible random sample per day."""

import argparse
import csv
import random
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


def trip_day(value):
    day = value.strip().split(" ")[0].split("T")[0]
    if "/" in day:
        month, day_number, year = map(int, day.split("/"))
        return date(year, month, day_number).isoformat()
    return date.fromisoformat(day).isoformat()


def sample_trips(input_dir, output_dir, trips_per_day=50, seed=42, overwrite=True):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if trips_per_day < 1:
        raise ValueError("trips_per_day must be positive")
    if (output_dir == input_dir or input_dir in output_dir.parents
            or output_dir in input_dir.parents):
        raise ValueError("Input and output must be separate, non-nested folders")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; use --overwrite or choose a new folder: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    files = sorted(
        p for p in input_dir.glob("*-citibike-tripdata/*/*.csv")
        if not p.name.startswith(("JC", "._"))
    )
    if not files:
        raise FileNotFoundError(f"No NYC trip CSVs found under {input_dir}")

    rng = random.Random(seed)
    counts, samples = Counter(), defaultdict(list)
    headers = {}
    # Cache parsed date tokens, not complete timestamps, to keep memory small.
    dates = {}
    for path in files:
        print(f"Reading {path}", flush=True)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            headers[path] = header
            start_index = next((i for i, column in enumerate(header)
                                if column.lower().replace(" ", "").replace("_", "")
                                in ("starttime", "startedat")), None)
            if start_index is None:
                raise ValueError(f"No start time column in {path}")
            for row in reader:
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError(f"Wrong column count: {path}, line {reader.line_num}")
                token = row[start_index].strip().split(" ")[0].split("T")[0]
                if token not in dates:
                    try:
                        dates[token] = trip_day(token)
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid start date: {path}, line {reader.line_num}: {token!r}"
                        ) from exc
                day = dates[token]
                counts[day] += 1
                # Reservoir sampling gives every trip that day equal probability,
                # including when a day is spread across multiple source CSVs.
                index = (counts[day] - 1 if counts[day] <= trips_per_day
                         else rng.randrange(counts[day]))
                if index < trips_per_day:
                    record = (path, row)
                    if len(samples[day]) < trips_per_day:
                        samples[day].append(record)
                    else:
                        samples[day][index] = record

    if not counts:
        raise ValueError("Source CSVs contain no trips")
    output_dir.mkdir(parents=True, exist_ok=overwrite)
    if overwrite:
        # Remove old sample CSVs only after the source has been read successfully.
        # This also removes the previous daily-file layout and obsolete files.
        for old_file in output_dir.glob("*-citibike-tripdata/*/*.csv"):
            old_file.unlink()
        for old_folder in output_dir.glob("*-citibike-tripdata/*"):
            if old_folder.is_dir() and not any(old_folder.iterdir()):
                old_folder.rmdir()
    by_source = defaultdict(list)
    for day, records in sorted(samples.items()):
        for path, row in records:
            by_source[path].append(row)
    for path in files:
        destination = output_dir / path.relative_to(input_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers[path])
            writer.writerows(by_source[path])
    with (output_dir / "sample_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "source_trips", "sampled_trips"])
        writer.writerows((day, counts[day], len(samples[day])) for day in sorted(counts))
    total = sum(map(len, samples.values()))
    (output_dir / "README.txt").write_text(
        f"Citi Bike daily trip sample\nSource: {input_dir}\n"
        f"Random seed: {seed}\nMaximum trips per day: {trips_per_day}\n"
        f"Days: {len(counts)}\nSampled trips: {total}\n"
        "Uniform reservoir sampling without replacement across all NYC input CSVs.\n"
        "Original columns and values preserved. Days with fewer trips keep all trips.\n"
        "Original trip CSV filenames and relative directories preserved.\n"
        "Each selected trip stays in its source file; 50 is the daily total across files.\n"
        "Only dates present in the source are included; JC and macOS metadata files excluded.\n"
        "This sample is for demonstration; counts do not represent full station demand.\n",
        encoding="utf-8",
    )
    print(f"Saved {total:,} trips across {len(counts):,} days to {output_dir}")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_sample"))
    parser.add_argument("--trips-per-day", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True,
                        help="Replace existing sample trip CSVs and summary files (default: enabled)")
    args = parser.parse_args()
    sample_trips(args.input_dir, args.output_dir, args.trips_per_day, args.seed,
                 overwrite=args.overwrite)


if __name__ == "__main__":
    main()

import os
import glob

# quick look at the raw trip files: which csv per month, header, first start time
# to call this script (from inside scripts/): python inspect_trip_csvs.py

for f in sorted(glob.glob("data/*-citibike-tripdata/*/*.csv")):
    with open(f, encoding="utf-8-sig") as fh:
        header = fh.readline().strip()
        first_row = fh.readline().strip()
    size_mb = os.path.getsize(f) / 1e6
    print(f, "| %.0f MB" % size_mb)
    print("   header:   ", header)
    print("   first row:", first_row[:120])

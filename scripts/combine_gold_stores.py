"""Concatenate monthly gold partitions into two standalone Parquet files.

Run from the repository root:
    python scripts/combine_gold_stores.py

Rows are stacked across months; features and labels remain separate tables.
"""

import argparse
import os
from pathlib import Path
import tempfile

import pyarrow.dataset as ds
import pyarrow.parquet as pq


def combine_store(input_directory, prefix, output_path):
    """Stream Spark Parquet partitions into one file without loading all rows."""
    partitions = sorted(Path(input_directory).glob(f"{prefix}_*.parquet"))
    if not partitions:
        raise FileNotFoundError(f"No {prefix}_*.parquet partitions in {input_directory}")
    output_path = Path(output_path)
    if any(output_path.resolve().is_relative_to(path.resolve()) for path in partitions):
        raise ValueError("Output must be outside the input partitions")

    schema = pq.ParquetDataset(partitions[0]).schema
    # Reject incompatible partitions before replacing any existing export.
    for partition in partitions:
        if not pq.ParquetDataset(partition).schema.equals(schema, check_metadata=False):
            raise ValueError(f"Schema mismatch in {partition}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(dir=output_path.parent, suffix=".parquet.tmp")
    os.close(descriptor)
    rows = 0
    try:
        with pq.ParquetWriter(temporary_path, schema, compression="snappy") as writer:
            for partition in partitions:
                dataset = ds.dataset(partition, format="parquet")
                for batch in dataset.to_batches(batch_size=65536):
                    writer.write_batch(batch)
                    rows += batch.num_rows
        if pq.ParquetFile(temporary_path).metadata.num_rows != rows:
            raise RuntimeError("Combined Parquet row count does not match input rows")
        os.replace(temporary_path, output_path)
    finally:
        Path(temporary_path).unlink(missing_ok=True)
    print(f"{len(partitions)} partitions, {rows:,} rows -> {output_path}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-directory", type=Path, default=Path("datamart_monthly/gold"))
    parser.add_argument("--output-directory", type=Path, default=None,
                        help="Default: <gold-directory>/combined")
    args = parser.parse_args(argv)
    output_directory = args.output_directory or args.gold_directory / "combined"
    for table in ("feature_store", "label_store"):
        combine_store(args.gold_directory / table, f"gold_{table}",
                      output_directory / f"gold_{table}.parquet")


if __name__ == "__main__":
    main()

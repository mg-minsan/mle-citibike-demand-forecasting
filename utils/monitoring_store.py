"""Publish monitoring snapshots without changing training runs or registry aliases."""
import json
import math
import numbers
import os

import mlflow
import pandas as pd


def log_monitoring(store, purpose, date, source_run_id, model_version, metrics, tags=None,
                   table=None, artifact_path=None):
    experiment_name = os.getenv("MLFLOW_MONITORING_EXPERIMENT_NAME", "citibike-model-monitoring")
    experiment = store.client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else store.client.create_experiment(experiment_name)
    if experiment_id == store.experiment_id:
        raise ValueError("Monitoring must use a separate experiment from training")
    # Each execution is a snapshot. Retries never modify completed training or monitoring runs.
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{purpose}-{date}") as run:
        mlflow.set_tags({"purpose": purpose, "monitored_date": date, "source_run_id": source_run_id,
                         "model_version": model_version, **(tags or {})})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()
                            if isinstance(value, numbers.Real) and math.isfinite(value)})
        if table is not None:
            mlflow.log_text(table.to_csv(index=False), "monitoring/report.csv")
            mlflow.log_dict(json.loads(table.to_json(orient="records", date_format="iso")),
                            "monitoring/report.json")
        if artifact_path:
            if os.path.isdir(artifact_path):
                mlflow.log_artifacts(artifact_path, "monitoring/parquet")
            else:
                mlflow.log_artifact(artifact_path, "monitoring")
        return run.info.run_id


def monitoring_tables(store, purpose):
    """Newest completed snapshot per source model/date; paginated and retry aware."""
    import tempfile
    experiment = store.client.get_experiment_by_name(
        os.getenv("MLFLOW_MONITORING_EXPERIMENT_NAME", "citibike-model-monitoring"))
    if experiment is None:
        return []
    if experiment.experiment_id == store.experiment_id:
        raise ValueError("Monitoring must use a separate experiment from training")
    tables, seen, token = [], set(), None
    while True:
        page = store.client.search_runs(
            [experiment.experiment_id],
            filter_string=f"attributes.status = 'FINISHED' AND tags.purpose = '{purpose}'",
            order_by=["attributes.start_time DESC"], max_results=1000, page_token=token)
        for run in page:
            tags = run.data.tags
            key = (tags["source_run_id"], tags["monitored_date"])
            if key in seen:
                continue
            seen.add(key)
            with tempfile.TemporaryDirectory() as directory:
                path = store.client.download_artifacts(run.info.run_id, "monitoring/report.csv", directory)
                tables.append(pd.read_csv(path))
        token = page.token
        if not token:
            return tables

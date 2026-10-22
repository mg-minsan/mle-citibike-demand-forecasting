"""MLflow-backed models and date-aware champion history.

Only successfully finished, fully published runs are visible to readers. Promotion
is immutable metadata on each run, not a mutable 'champion' alias: backfills must
select the champion that was effective on their snapshot date.
"""

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from utils.feature_importance import log_feature_importance


@dataclass(frozen=True)
class SavedModel:
    run_id: str
    version: str
    model_uri: str
    model_name: str
    trained_through: str
    effective_from: str
    promoted: bool


class ModelStore:
    def __init__(self, tracking_uri=None, experiment_name=None, registered_model_name=None):
        tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        self.registered_model_name = registered_model_name or os.getenv(
            "MLFLOW_REGISTERED_MODEL_NAME", "citibike-demand"
        )
        experiment_name = experiment_name or os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "citibike-demand-forecasting"
        )
        self.experiment_id = mlflow.set_experiment(experiment_name).experiment_id

    def _models(self):
        token = None
        while True:
            page = self.client.search_runs(
                [self.experiment_id],
                filter_string="attributes.status = 'FINISHED' AND tags.`citibike.ready` = 'true'",
                max_results=1000,
                page_token=token,
            )
            for run in page:
                tags = run.data.tags
                yield SavedModel(
                    run.info.run_id, tags["version"], tags["model_uri"], tags["model_type"],
                    tags["trained_through"], tags["effective_from"], tags["promoted"] == "true",
                )
            token = page.token
            if not token:
                break

    def for_cutoff(self, trained_through):
        return next((model for model in self._models() if model.trained_through == trained_through), None)

    def champion(self, snapshotdate, inclusive=False):
        eligible = [
            model for model in self._models()
            if model.promoted and model.trained_through < snapshotdate
            and (model.effective_from <= snapshotdate if inclusive else model.effective_from < snapshotdate)
        ]
        return max(eligible, key=lambda model: (model.effective_from, model.trained_through), default=None)

    def for_prediction(self, snapshotdate):
        champion = self.champion(snapshotdate, inclusive=True)
        if champion is not None:
            return champion
        eligible = [
            model for model in self._models()
            if model.trained_through < snapshotdate and model.effective_from <= snapshotdate
        ]
        if not eligible:
            raise FileNotFoundError("no MLflow model trained before and effective on " + snapshotdate)
        return max(eligible, key=lambda model: model.trained_through)

    def info(self, model):
        with tempfile.TemporaryDirectory() as directory:
            path = self.client.download_artifacts(model.run_id, "training_info.json", directory)
            return json.loads(Path(path).read_text())

    def load(self, model):
        return mlflow.sklearn.load_model(model.model_uri)

    def last_best_params(self, model_name, trained_through):
        eligible = sorted(
            (model for model in self._models()
             if model.model_name == model_name and model.trained_through < trained_through),
            key=lambda model: model.trained_through,
            reverse=True,
        )
        for model in eligible:
            params = self.info(model).get("params")
            if params:
                return params
        return None

    def save(self, model, info, version, input_example=None):
        # A retry ignores failed/incomplete runs; a completed cutoff is never overwritten.
        existing = self.for_cutoff(info["trained_through"])
        if existing is not None:
            return existing
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=version) as run:
            mlflow.set_tags({
                "version": version,
                "model_type": info["model"],
                "trained_through": info["trained_through"],
                "effective_from": info["retrained_on"],
                "promoted": str(info["promoted"]).lower(),
                "test_metrics_scope": "train_validation_refit_before_final_refit",
            })
            mlflow.log_params(info.get("hyperparameters", {}))
            mlflow.log_params({"tuned." + key: value for key, value in info.get("params", {}).items()})
            metrics = {
                key: info[key] for key in (
                    "test_model_mae", "test_model_rmse", "test_baseline_mae", "test_baseline_rmse",
                    "validation_baseline_mae", "validation_baseline_rmse", "champion_test_mae",
                ) if info.get(key) is not None
            }
            for metric in ("validation_mae", "validation_rmse"):
                for name, value in info.get(metric, {}).items():
                    metrics[metric + "/" + re.sub(r"[^a-zA-Z0-9_-]", "_", name)] = value
            mlflow.log_metrics(metrics)
            mlflow.log_dict(info, "training_info.json")
            if "feature_importance" in info:
                log_feature_importance(info["feature_importance"])
            signature = None
            if input_example is not None:
                signature = infer_signature(input_example, model.predict(input_example))
            # Keep objectives in parameters; display the winning algorithm and cutoff.
            algorithm = re.sub(r"[^a-z0-9]+", "-", info["model"].split(" (", 1)[0].lower()).strip("-")
            logged = mlflow.sklearn.log_model(
                sk_model=model,
                name=f"{algorithm}-{info['trained_through']}",
                input_example=input_example,
                signature=signature,
                registered_model_name=self.registered_model_name,
                # StationProfile contains pandas state unsupported by skops.
                # Load only models from our trusted tracking server.
                serialization_format="cloudpickle",
                # Include our custom sklearn transformer, also for loading outside /app.
                code_paths=[str(Path(__file__).resolve().parent)],
            )
            # Last write is the publication marker. FINISHED status is also required,
            # so failures anywhere above cannot become prediction candidates.
            mlflow.set_tags({"model_uri": logged.model_uri, "citibike.ready": "true"})
            saved = SavedModel(
                run.info.run_id, version, logged.model_uri, info["model"],
                info["trained_through"], info["retrained_on"], info["promoted"],
            )
        return saved

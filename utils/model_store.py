"""MLflow-backed models and date-aware champion history.

Only successfully finished, fully published runs are visible to readers. The
champion alias points to the latest promoted model. Immutable promotion tags
preserve the champion effective on each snapshot date for historical backfills.
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

    def for_prediction(self, snapshotdate, mode="historical"):
        if mode == "live":
            saved = self.latest_champion()
            if saved.trained_through >= snapshotdate or saved.effective_from > snapshotdate:
                raise ValueError("champion alias is not eligible for prediction on " + snapshotdate)
            return saved
        if mode != "historical":
            raise ValueError("model selection must be historical or live")
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

    def latest_champion(self):
        """Resolve the live alias once and pin its version for this prediction job."""
        version = self.client.get_model_version_by_alias(self.registered_model_name, "champion")
        run = self.client.get_run(version.run_id)
        tags = run.data.tags
        if (run.info.status != "FINISHED" or tags.get("citibike.ready") != "true"
                or tags.get("promoted") != "true" or run.info.experiment_id != self.experiment_id):
            raise ValueError("champion alias must point to a completed, published promotion in this experiment")
        return SavedModel(
            run.info.run_id, tags["version"],
            f"models:/{self.registered_model_name}/{version.version}", tags["model_type"],
            tags["trained_through"], tags["effective_from"], True,
        )

    def info(self, model):
        with tempfile.TemporaryDirectory() as directory:
            path = self.client.download_artifacts(model.run_id, "training_info.json", directory)
            return json.loads(Path(path).read_text())

    def load(self, model):
        return mlflow.sklearn.load_model(model.model_uri)

    def sync_champion_alias(self):
        """Publish the latest completed promotion, including after a retry.

        Choose by effective date so replaying an older cutoff cannot move the
        live alias back to an earlier champion.
        """
        champion = max(
            (model for model in self._models() if model.promoted),
            key=lambda model: (model.effective_from, model.trained_through),
            default=None,
        )
        if champion is None:
            return None
        token = None
        matches = []
        while True:
            page = self.client.search_model_versions(
                filter_string=f"name = '{self.registered_model_name}'",
                max_results=1000,
                page_token=token,
            )
            matches.extend(version for version in page if version.run_id == champion.run_id)
            token = page.token
            if not token:
                break
        if not matches:
            raise LookupError(f"No registered model version for champion run {champion.run_id}")
        version = max(matches, key=lambda item: int(item.version)).version
        self.client.set_registered_model_alias(self.registered_model_name, "champion", version)
        return version

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
            self.sync_champion_alias()
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
            mlflow.set_tags({
                "promotion_reason": info.get("promotion_reason", ""),
                "compared_champion_version": info.get("champion_version") or "none",
            })
            metadata = {key: info[key] for key in ("trained_from", "train_rows") if key in info}
            for window in ("validation_window", "test_window"):
                if window in info:
                    metadata[window + "_start"], metadata[window + "_end"] = info[window]
            mlflow.log_params(metadata)
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
        # Publish the alias only after the run has successfully finished. A
        # failure here can be repaired without retraining the completed model.
        if saved.promoted:
            self.sync_champion_alias()
        return saved

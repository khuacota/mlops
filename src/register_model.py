"""Registra los 2 mejores runs como versiones del modelo y marca el champion."""
import mlflow
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "late-delivery"
MODEL_NAME = "late-delivery-classifier"
METRIC = "roc_auc"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([exp.experiment_id], order_by=[f"metrics.{METRIC} DESC"], max_results=2)

    versions = []
    for run in runs:
        mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
        versions.append((mv.version, run))
        print(f"Versión {mv.version} <- run {run.info.run_id} ({run.info.run_name}) {METRIC}={run.data.metrics[METRIC]:.4f}")

    best_version, best_run = versions[0]
    client.set_registered_model_alias(MODEL_NAME, "champion", best_version)
    client.set_model_version_tag(MODEL_NAME, best_version, "run_id", best_run.info.run_id)
    print(f"\nAlias 'champion' -> versión {best_version} (run {best_run.info.run_id})")
    print("ANOTA este run_id y esta versión: van en el documento de arquitectura.")


if __name__ == "__main__":
    main()
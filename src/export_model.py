"""Descarga el modelo con alias 'champion' del Model Registry a la carpeta model/."""
import json
import shutil
from pathlib import Path

import mlflow
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "late-delivery-classifier"
ALIAS = "champion"
OUT_DIR = Path("model")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{MODEL_NAME}@{ALIAS}",
        dst_path=str(OUT_DIR / "artifacts"),
    )

    info = {
        "model_name": MODEL_NAME,
        "alias": ALIAS,
        "version": mv.version,
        "run_id": mv.run_id,
    }
    (OUT_DIR / "model_info.json").write_text(json.dumps(info, indent=2))
    print("Modelo exportado en model/artifacts")
    print(info)


if __name__ == "__main__":
    main()
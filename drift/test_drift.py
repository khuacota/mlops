"""Pruebas automatizadas: verde con datos normales, rojo con datos derivados."""
import mlflow.sklearn
import pandas as pd
import pytest

from drift.monitoring import (
    baseline_auc,
    data_drift_report,
    has_data_drift,
    retrain_alarm,
    run_concept_batches,
)
from src.generate_data import generate


@pytest.fixture(scope="module")
def reference():
    return pd.read_csv("data/reference.csv")


@pytest.fixture(scope="module")
def model():
    return mlflow.sklearn.load_model("model/artifacts")


# ---------- Data drift ----------
def test_data_drift_verde_con_datos_normales(reference):
    report = data_drift_report(reference, generate(1000, seed=9))
    assert not has_data_drift(report), report


def test_data_drift_rojo_con_datos_derivados(reference):
    report = data_drift_report(reference, generate(1000, seed=8, data_drift=True))
    assert has_data_drift(report)
    assert report.loc[report["variable"] == "distance_km", "drift"].item()


# ---------- Concept drift ----------
def test_concept_drift_verde_sin_cambio(model):
    df = run_concept_batches(model, change_at=99)  # nunca cambia el concepto
    assert retrain_alarm(df["roc_auc"], baseline_auc(model)) is None


def test_concept_drift_rojo_con_cambio(model):
    df = run_concept_batches(model, change_at=6)
    alarm = retrain_alarm(df["roc_auc"], baseline_auc(model))
    assert alarm is not None and alarm >= 6
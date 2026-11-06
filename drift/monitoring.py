"""Lógica de detección de data drift y concept drift."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.generate_data import generate

NUMERIC = ["distance_km", "weight_kg", "num_items", "dispatch_hour", "warehouse_load", "weather_risk"]
CATEGORICAL = ["carrier", "shipping_type"]
TARGET = "late"

# ---------- Data drift ----------
ALPHA = 0.01                                   # nivel de significancia global
N_TESTS = len(NUMERIC) + len(CATEGORICAL)      # 8 pruebas por lote
THRESHOLD = ALPHA / N_TESTS                    # corrección de Bonferroni: 0.00125


def data_drift_report(reference: pd.DataFrame, batch: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Compara cada variable del lote contra la referencia. drift=True si p-value < umbral."""
    rows = []
    for col in NUMERIC:
        stat, p = stats.ks_2samp(reference[col], batch[col])
        rows.append({"variable": col, "test": "KS", "statistic": stat, "p_value": p, "drift": p < threshold})
    for col in CATEGORICAL:
        cats = sorted(reference[col].unique())
        ref_counts = reference[col].value_counts().reindex(cats, fill_value=0).to_numpy()
        batch_counts = batch[col].value_counts().reindex(cats, fill_value=0).to_numpy()
        stat, p, _, _ = stats.chi2_contingency(np.array([ref_counts, batch_counts]))
        rows.append({"variable": col, "test": "Chi2", "statistic": stat, "p_value": p, "drift": p < threshold})
    return pd.DataFrame(rows)


def has_data_drift(report: pd.DataFrame) -> bool:
    return bool(report["drift"].any())


# ---------- Concept drift ----------
MAX_DROP = 0.05    # caída absoluta de ROC-AUC respecto al baseline
PATIENCE = 2       # lotes consecutivos por debajo del umbral


def batch_metrics(model, batch: pd.DataFrame):
    """Devuelve (ROC-AUC, probabilidad media predicha) de un lote."""
    proba = model.predict_proba(batch.drop(columns=TARGET))[:, 1]
    return roc_auc_score(batch[TARGET], proba), float(proba.mean())


def baseline_auc(model) -> float:
    """ROC-AUC del modelo sobre datos limpios que no se usaron en el entrenamiento (semilla distinta)."""
    return batch_metrics(model, generate(5000, seed=99))[0]


def run_concept_batches(model, n_batches: int = 10, change_at: int = 6, batch_size: int = 1000) -> pd.DataFrame:
    """Simula lotes sucesivos. Desde el lote `change_at` la regla del atraso cambia."""
    rows = []
    for i in range(1, n_batches + 1):
        concept = "original" if i < change_at else "changed"
        batch = generate(batch_size, seed=1000 + i, concept=concept)
        auc, mean_proba = batch_metrics(model, batch)
        rows.append({"batch": i, "concept": concept, "roc_auc": auc, "mean_proba": mean_proba})
    return pd.DataFrame(rows)


def retrain_alarm(aucs, baseline: float, max_drop: float = MAX_DROP, patience: int = PATIENCE):
    """Devuelve el número de lote donde se dispara la alarma, o None si no se dispara."""
    consecutive = 0
    for i, auc in enumerate(aucs, start=1):
        consecutive = consecutive + 1 if (baseline - auc) > max_drop else 0
        if consecutive >= patience:
            return i
    return None
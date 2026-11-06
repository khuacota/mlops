"""Demo de data drift: un lote normal (verde) y un lote con deriva (rojo)."""
import pandas as pd

from drift.monitoring import THRESHOLD, data_drift_report, has_data_drift
from src.generate_data import generate

reference = pd.read_csv("data/reference.csv")

batches = {
    "LOTE NORMAL (mismo origen)": generate(1000, seed=9),
    "LOTE CON DRIFT (temporada alta)": generate(1000, seed=8, data_drift=True),
}

print(f"Umbral de alerta: p-value < {THRESHOLD:.5f}\n")
for name, batch in batches.items():
    report = data_drift_report(reference, batch)
    status = "ROJO (drift detectado)" if has_data_drift(report) else "VERDE (sin drift)"
    print(f"=== {name} -> {status}")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print()
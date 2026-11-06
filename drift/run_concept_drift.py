"""Demo de concept drift: ROC-AUC en lotes sucesivos y gráfica temporal."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow.sklearn

from drift.monitoring import MAX_DROP, PATIENCE, baseline_auc, retrain_alarm, run_concept_batches

CHANGE_AT = 6

model = mlflow.sklearn.load_model("model/artifacts")
baseline = baseline_auc(model)
df = run_concept_batches(model, n_batches=10, change_at=CHANGE_AT)
alarm = retrain_alarm(df["roc_auc"], baseline)

clean_std = df[df["concept"] == "original"]["roc_auc"].std()
print(f"Baseline ROC-AUC: {baseline:.4f}")
print(f"Variación natural (desv. estándar en lotes limpios): {clean_std:.4f}")
print(f"Criterio: caída > {MAX_DROP} durante {PATIENCE} lotes seguidos\n")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nAlarma de reentrenamiento en el lote: {alarm if alarm else 'no se disparó'}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
ax1.plot(df["batch"], df["roc_auc"], marker="o", label="ROC-AUC por lote")
ax1.axhline(baseline, color="green", linestyle="--", label="Baseline")
ax1.axhline(baseline - MAX_DROP, color="red", linestyle="--", label=f"Umbral (baseline - {MAX_DROP})")
ax1.axvline(CHANGE_AT - 0.5, color="gray", linestyle=":", label="Cambio de concepto")
if alarm:
    ax1.axvline(alarm, color="red", alpha=0.4, linewidth=6, label="Alarma de reentrenamiento")
ax1.set_ylabel("ROC-AUC")
ax1.set_title("Concept drift: degradación del modelo por lote")
ax1.legend()

ax2.plot(df["batch"], df["mean_proba"], marker="s", color="orange")
ax2.axvline(CHANGE_AT - 0.5, color="gray", linestyle=":")
ax2.set_ylabel("Probabilidad media predicha")
ax2.set_xlabel("Lote")
ax2.set_title("Señal sin etiquetas (ver retraso de etiquetas)")

plt.tight_layout()
plt.savefig("drift/concept_drift.png", dpi=120)
print("Gráfica guardada en drift/concept_drift.png")
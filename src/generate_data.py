"""Genera datos sintéticos de pedidos de una empresa de logística.

Objetivo: late = 1 si el pedido llega tarde al cliente, 0 si llega a tiempo.

Se reutiliza en el Paso 4 para fabricar lotes con drift:
  - data_drift=True      -> temporada alta: cambia la distribución de las entradas.
  - concept="changed"    -> cambio de operador/rutas: misma entrada, distinta relación con el atraso.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
CARRIERS = ["A", "B", "C"]
SHIPPING = ["standard", "express"]

# Efecto de cada operador y del envío express sobre el logit del atraso
EFFECTS = {
    "original": {"carrier": {"A": 0.0, "B": 0.2, "C": 0.4}, "express": -1.0},
    # Concept drift: el operador A empeora (p. ej. nuevo centro de distribución)
    # y el envío express deja de ayudar tanto.
    "changed": {"carrier": {"A": 1.0, "B": 0.0, "C": -0.5}, "express": -0.2},
}


def make_target(df: pd.DataFrame, rng: np.random.Generator, concept: str = "original") -> np.ndarray:
    """Regla que decide si un pedido se atrasa (la 'verdad' del negocio)."""
    eff = EFFECTS[concept]
    logit = (
        -1.5
        + 0.003 * (df["distance_km"] - 300)           # más distancia, más atraso
        + 0.02 * (df["weight_kg"] - 5)                # paquetes pesados
        + 0.10 * (df["num_items"] - 4)                # más artículos
        + 0.5 * (df["dispatch_hour"] >= 17)           # despachos de la tarde
        + 3.0 * (df["warehouse_load"] - 0.55)         # almacén saturado
        + 3.0 * (df["weather_risk"] - 0.25)           # mal clima
        + df["carrier"].map(eff["carrier"])           # operador logístico
        + eff["express"] * (df["shipping_type"] == "express")
    )
    prob = 1 / (1 + np.exp(-logit))
    return (rng.random(len(df)) < prob).astype(int)


def generate(n: int, seed: int = SEED, data_drift: bool = False, concept: str = "original") -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    distance = rng.gamma(2, 150, n)
    weight = rng.lognormal(1.5, 0.8, n)
    num_items = rng.poisson(3, n) + 1
    hour = rng.integers(6, 23, n)
    load = rng.beta(5, 4, n)
    weather = rng.beta(2, 6, n)
    carrier_probs = [0.5, 0.3, 0.2]
    express_prob = 0.25

    if data_drift:  # temporada alta: cambia la distribución de las ENTRADAS
        distance = distance * 1.3
        weight = weight * 1.6
        num_items = num_items + 2
        load = np.clip(load + 0.2, 0, 1)
        weather = np.clip(weather + 0.2, 0, 1)
        carrier_probs = [0.2, 0.3, 0.5]
        express_prob = 0.5

    df = pd.DataFrame(
        {
            "distance_km": distance.clip(5, 3000),
            "weight_kg": weight.clip(0.1, 80),
            "num_items": num_items.astype(float),
            "dispatch_hour": hour.astype(float),
            "warehouse_load": load,
            "weather_risk": weather,
            "carrier": rng.choice(CARRIERS, n, p=carrier_probs),
            "shipping_type": rng.choice(SHIPPING, n, p=[1 - express_prob, express_prob]),
        }
    )
    df["late"] = make_target(df, rng, concept)
    return df


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    df = generate(5000)
    df.to_csv("data/reference.csv", index=False)
    print(f"Guardado data/reference.csv  filas={len(df)}  tasa de atraso={df['late'].mean():.2%}")
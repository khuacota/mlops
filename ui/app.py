"""UI simple (Streamlit) que consume la API desplegada en Kubernetes."""
import os
from collections import Counter

import pandas as pd
import requests
import streamlit as st

# NodePort del Service de Kubernetes (no el proceso local en el puerto 8000)
API_URL = os.getenv("API_URL", "http://localhost:30080")

st.set_page_config(page_title="Pedidos atrasados", page_icon="📦")
st.title("📦 Predicción de pedidos atrasados")

# ---------- Barra lateral: estado del servicio ----------
with st.sidebar:
    st.header("Servicio")
    st.code(API_URL)
    try:
        info = requests.get(f"{API_URL}/info", timeout=3).json()
        st.success("API disponible")
        st.write(f"**Modelo:** {info['model_name']}")
        st.write(f"**Versión:** {info['version']} (alias `{info['alias']}`)")
        st.write(f"**Run:** `{info['run_id'][:8]}...`")
        st.caption(f"Respondió el pod: {info['pod']}")
    except requests.RequestException:
        st.error("La API no responde. ¿Está corriendo en Kubernetes?")

# ---------- Formulario de predicción ----------
st.subheader("Datos del pedido")
with st.form("pedido"):
    col1, col2 = st.columns(2)
    with col1:
        distance_km = st.number_input("Distancia (km)", min_value=1.0, value=450.0)
        weight_kg = st.number_input("Peso (kg)", min_value=0.1, value=8.0)
        num_items = st.number_input("Número de artículos", min_value=1, value=5)
        dispatch_hour = st.slider("Hora de despacho", 0, 23, 18)
    with col2:
        warehouse_load = st.slider("Ocupación del almacén", 0.0, 1.0, 0.7)
        weather_risk = st.slider("Riesgo de clima adverso", 0.0, 1.0, 0.4)
        carrier = st.selectbox("Operador logístico", ["A", "B", "C"], index=2)
        shipping_type = st.selectbox("Tipo de envío", ["standard", "express"])
    submitted = st.form_submit_button("Predecir")

if submitted:
    payload = {
        "distance_km": distance_km,
        "weight_kg": weight_kg,
        "num_items": num_items,
        "dispatch_hour": dispatch_hour,
        "warehouse_load": warehouse_load,
        "weather_risk": weather_risk,
        "carrier": carrier,
        "shipping_type": shipping_type,
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        resp.raise_for_status()
        result = resp.json()
        if result["late"]:
            st.error("⚠️ Probable atraso")
        else:
            st.success("✅ Probablemente llegará a tiempo")
        st.metric("Probabilidad de atraso", f"{result['probability_late']:.1%}")
        st.caption(
            f"Modelo versión {result['model_version']} · run {result['run_id'][:8]}... "
            f"· atendido por el pod **{result['pod']}**"
        )
    except requests.RequestException as exc:
        st.error(f"Error al llamar a la API: {exc}")

# ---------- Demo de balanceo de carga ----------
st.divider()
st.subheader("Demo de balanceo de carga")
n = st.number_input("Número de peticiones", min_value=3, max_value=200, value=30)
if st.button("Enviar peticiones al Service"):
    pods = []
    for _ in range(int(n)):
        try:
            # requests.get abre una conexión nueva en cada llamada
            pods.append(requests.get(f"{API_URL}/health", timeout=3).json()["pod"])
        except requests.RequestException:
            pods.append("ERROR")
    counts = Counter(pods)
    st.bar_chart(pd.Series(counts, name="peticiones"))
    st.write(dict(counts))
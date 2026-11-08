# Instrucciones de ejecución

Pasos para ejecutar el proyecto de predicción de pedidos atrasados (logística) desde cero.

## 1. Crear y activar el entorno virtual

Desde la carpeta raíz del proyecto (`mlops-proyecto`):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

La terminal debe mostrar el prefijo `(.venv)`.

## 2. Instalar dependencias

```powershell
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 3. Fase 1: Entrenamiento, experimentos y registro con MLflow

### Terminal 1: Iniciar servidor de MLflow
```powershell
.venv\Scripts\python.exe -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

### Terminal 2: Generar datos, entrenar y registrar el modelo
```powershell
.venv\Scripts\python.exe src\generate_data.py
.venv\Scripts\python.exe src\train.py
.venv\Scripts\python.exe src\register_model.py
.venv\Scripts\python.exe src\export_model.py
```

Comprobación en interfaz web:
- Abre http://127.0.0.1:5000
- **Experiments (`late-delivery`)**: 6 runs registrados y ordenables por `roc_auc`.
- **Models (`late-delivery-classifier`)**: 2 versiones registradas; la versión 1 marcada con el alias `champion`.

---

## 4. Fase 2: Contenedor Docker

Con el modelo exportado en la carpeta `model/`, construye y ejecuta la imagen Docker autónoma:

```powershell
docker build -t late-delivery-api:1.0 .
docker run --rm -p 8000:8000 late-delivery-api:1.0
```

Prueba la API en el navegador:
- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/info

Ejemplo de cuerpo para `POST /predict`:

```json
{
  "distance_km": 450,
  "weight_kg": 8,
  "num_items": 5,
  "dispatch_hour": 18,
  "warehouse_load": 0.7,
  "weather_risk": 0.4,
  "carrier": "C",
  "shipping_type": "standard"
}
```

## 5. Fase 3: Despliegue en Kubernetes (3 Réplicas)

Asegúrate de tener Docker Desktop activo con Kubernetes habilitado.

```powershell
kubectl config use-context docker-desktop
kubectl get nodes
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods -w
```

Verifica que los 3 pods alcancen el estado `1/1 Running`.
- Servicio NodePort expuesto en: http://localhost:30080/info

### Verificaciones operativas en Kubernetes

1. **Balanceo de carga (30 peticiones)**:
   ```powershell
   1..30 | ForEach-Object { (curl.exe -s http://localhost:30080/health | ConvertFrom-Json).pod } | Group-Object | Select-Object Name, Count
   ```

2. **Autorreparación (*Self-healing*)**:
   ```powershell
   kubectl delete pod <nombre-de-un-pod>
   kubectl get pods -w
   ```

3. **Escalado dinámico**:
   ```powershell
   kubectl scale deployment late-delivery-api --replicas=5
   kubectl get pods
   kubectl scale deployment late-delivery-api --replicas=3
   ```

---

## 6. Fase 4: Detección de Drift y Pruebas Automáticas

Ejecuta las demostraciones de Data Drift, Concept Drift y la suite de tests desde la raíz:

```powershell
.venv\Scripts\python.exe -m drift.run_data_drift
.venv\Scripts\python.exe -m drift.run_concept_drift
.venv\Scripts\python.exe -m pytest drift -v
```

- **Data Drift**: Compara muestras numéricas (Kolmogorov-Smirnov) y categóricas ($\chi^2$) aplicando corrección de Bonferroni ($p < 0.00125$).
- **Concept Drift**: Genera el gráfico `drift/concept_drift.png` mostrando la caída del $ROC\text{-}AUC$ a partir del lote 7.

---

## 7. Fase 5: Interfaz Web (Streamlit)

Con el clúster de Kubernetes corriendo en el puerto 30080, levanta la app de Streamlit:

```powershell
.venv\Scripts\python.exe -m streamlit run ui/app.py
```

Abrir en el navegador:
- http://localhost:8501

Permite realizar predicciones interactiva y probar el balanceo de carga en vivo enviando ráfagas de peticiones al clúster.
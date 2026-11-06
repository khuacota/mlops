MLOps: predicción de pedidos atrasados (logística)

Proyecto que lleva un modelo de machine learning desde el entrenamiento hasta un despliegue reproducible y observable:

MLflow → Docker → Kubernetes (3 réplicas) → detección de drift → interfaz web

Problema: predecir si un pedido llegará tarde al cliente (late: 1 = tarde, 0 = a tiempo).
Datos: sintéticos, 5.000 pedidos y 8 variables, generados con src/generate_data.py (semilla 42).
Modelo: RandomForest de scikit-learn.
Modelo desplegado (trazabilidad)
Dato	Valor
Modelo en el registry	late-delivery-classifier
Versión desplegada	1 (alias champion)
Run de origen	a2eb441b26c245c9bee71704f12e858d
Imagen Docker	late-delivery-api:1.0
Servicio en Kubernetes	http://localhost:30080 (NodePort)


Estructura del proyecto
mlops-proyecto/
├── README.md
├── requirements.txt          dependencias de entrenamiento, monitoreo y UI
├── requirements-api.txt      dependencias de la imagen (versiones fijadas)
├── .gitignore
├── .dockerignore
├── Dockerfile
├── data/
│   └── reference.csv         baseline de entrenamiento (se genera solo)
├── src/
│   ├── generate_data.py      genera datos; reutilizado para crear drift
│   ├── train.py              6 runs con distintos hiperparámetros
│   ├── register_model.py     registra 2 versiones y marca el champion
│   └── export_model.py       descarga el champion a model/ (opción A)
├── app/
│   ├── __init__.py
│   └── main.py               API de inferencia (FastAPI)
├── model/                    modelo exportado (se genera, no se sube a git)
├── k8s/
│   ├── deployment.yaml       3 réplicas
│   └── service.yaml          NodePort 30080
├── drift/
│   ├── __init__.py
│   ├── monitoring.py         lógica de data drift y concept drift
│   ├── run_data_drift.py     demo verde / rojo
│   ├── run_concept_drift.py  demo y gráfica temporal
│   └── test_drift.py         pruebas automatizadas (pytest)
├── ui/
    └── app.py                interfaz Streamlit (extra)

Requisitos

Python 3.11.
Docker Desktop con Kubernetes habilitado
Instalación
powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe --version
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

requirements.txt:

mlflow==2.16.2
scikit-learn==1.5.2
pandas==2.2.3
numpy==1.26.4
SQLAlchemy==2.0.36
alembic==1.13.3
scipy==1.13.1
matplotlib==3.9.2
pytest==8.3.3
streamlit==1.39.0
requests==2.32.3

Para probar la API sin Docker, instala además fastapi==0.115.0, uvicorn==0.31.0 y pydantic==2.9.2.

Fase 1: modelo, experimentos y registro con MLflow

Terminal 1 (Servidor de MLflow):

powershell
.venv\Scripts\python.exe -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

Terminal 2 (misma carpeta):

powershell
.venv\Scripts\python.exe src\generate_data.py
.venv\Scripts\python.exe src\train.py
.venv\Scripts\python.exe src\register_model.py

Abre http://127.0.0.1:5000 y comprueba:

Experiments, late-delivery: 6 runs, ordenables por roc_auc.
Compare: selecciona 2 o más runs para argumentar la elección del modelo.
Models, late-delivery-classifier: 2 versiones; el alias champion marca la desplegada.

Parámetros del entrenamiento: división 80 % / 20 % estratificada, semilla 42, métricas roc_auc (principal), f1 y accuracy.

Fase 2: contenedor Docker

Con el servidor MLflow corriendo, exporta el modelo champion y construye la imagen:

powershell
.venv\Scripts\python.exe src\export_model.py
docker build -t late-delivery-api:1.0 .
docker run --rm -p 8000:8000 late-delivery-api:1.0

Abre http://localhost:8000/docs y prueba POST /predict con:

json
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

/info debe mostrar version: 1 y el run a2eb441b.... Detén el contenedor con Ctrl+C.

Decisión de arquitectura: el modelo se copia dentro de la imagen. Así la imagen es autosuficiente y los pods no dependen de MLflow. El costo: cambiar de champion exige exportar de nuevo y construir otra imagen.

Fase 3: Kubernetes con 3 réplicas
powershell
kubectl config use-context docker-desktop
kubectl get nodes
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods -w

Los 3 pods deben llegar a 1/1 Running. Comprueba http://localhost:30080/info.

Las cuatro demostraciones (guardar capturas en evidencias/)

1. Tres réplicas en Running

powershell
kubectl get pods -o wide
kubectl get deployment late-delivery-api

2. Balanceo de carga (peticiones sucesivas atendidas por pods distintos)

powershell
1..30 | ForEach-Object { (curl.exe -s http://localhost:30080/health | ConvertFrom-Json).pod } | Group-Object | Select-Object Name, Count

3. Autorreparación. Terminal A, bucle de peticiones:

powershell
kubectl get pods
kubectl delete pod <nombre-de-un-pod>
kubectl get pods -w

4. Escalado

powershell
kubectl scale deployment late-delivery-api --replicas=5
kubectl get pods
kubectl scale deployment late-delivery-api --replicas=2
kubectl get pods
kubectl scale deployment late-delivery-api --replicas=3

Fase 4: data drift y concept drift

Se ejecuta desde la raíz del proyecto, sin Docker ni Kubernetes, usando el modelo exportado en model/artifacts.

powershell
.venv\Scripts\python.exe -m drift.run_data_drift
.venv\Scripts\python.exe -m drift.run_concept_drift
.venv\Scripts\python.exe -m pytest drift -v

Data drift
Tipo de variable	Prueba	Motivo
Numérica (6 variables)	Kolmogorov-Smirnov	No asume una distribución concreta y compara la distribución completa
Categórica (carrier, shipping_type)	Chi-cuadrado	Compara las frecuencias de cada categoría
Umbral: p-value < 0.00125 = 0.01 / 8 pruebas (corrección de Bonferroni).
Verde: lote de 1.000 pedidos del mismo origen que el entrenamiento (semilla 9).
Rojo: lote de "temporada alta" (más distancia, peso, carga y clima adverso; más express y más pedidos del operador C).
Con la semilla 7, el lote normal dio rojo por casualidad (distance_km, p = 0.0011): es la falsa alarma residual de ~1 % que tiene toda prueba estadística.


Concept drift
Desde el lote 7 cambia la regla que genera el atraso: el operador A empeora, el C mejora y el envío express ayuda mucho menos. Las entradas no cambian.

Baseline: ROC-AUC 0.738 sobre 5.000 pedidos limpios con otra semilla.
Variación natural: desviación estándar 0.0085 entre lotes limpios.
Criterio de reentrenamiento: caída de más de 0.05 respecto al baseline, durante 2 lotes seguidos.
Resultado: el ROC-AUC baja de ~0.73 a ~0.66 y la alarma salta en el lote 7. La gráfica queda en drift/concept_drift.png.

Retraso de etiquetas

En logística la etiqueta late solo se conoce cuando el pedido llega. Mientras tanto el equipo: (1) vigila el data drift y la distribución de predicciones, (2) guarda las predicciones y calcula el ROC-AUC cuando llegan las etiquetas, (3) usa etiquetas parciales de los retrasos graves y (4) avisa al negocio y revisa una muestra si hay señales sospechosas. En la simulación, la probabilidad media predicha no cambia tras el cambio de concepto: solo el ROC-AUC, que necesita la etiqueta, lo delata.

Fase 5: interfaz Streamlit

Con el clúster arriba:

powershell
.venv\Scripts\python.exe -m streamlit run ui/app.py

Abre http://localhost:8501. La interfaz consume el NodePort http://localhost:30080 (variable de entorno API_URL), no un proceso local. La barra lateral muestra la versión del modelo y el run, el formulario hace la predicción y el botón de balanceo grafica cuántas peticiones atendió cada pod.

Decisiones de arquitectura
Decisión	Motivo
Datos sintéticos	Permiten generar data drift y concept drift de forma controlada
RandomForest
ROC-AUC como métrica principal	La clase "atrasado" es minoritaria (20 a 25 %)
Alias champion en el registry	Marca de forma explícita la versión desplegada, sin rutas sueltas
Modelo dentro de la imagen	Imagen autosuficiente; los pods no dependen de MLflow
Python 3.11 y dependencias fijadas	Evita incompatibilidades al cargar el modelo
3 réplicas, readiness y liveness probes	Alta disponibilidad y tráfico solo a pods listos
KS y chi-cuadrado con Bonferroni	Pruebas estándar con un umbral justificable
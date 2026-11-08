# MLOps: Predicción de Pedidos Atrasados (Logística)

![CI](https://github.com/tu-usuario/mlops-proyecto/actions/workflows/ci.yml/badge.svg)

Proyecto integrador de **MLOps**: un sistema de machine learning para predecir retrasos en entregas logísticas, llevado desde la etapa de experimentación y registro hasta un despliegue reproducible en **Kubernetes**, instrumentado con monitoreo continuo de **Data Drift y Concept Drift**, pruebas automatizadas y una interfaz web interactiva.

---

## Capacidades

- **Gobierno y Trazabilidad**: Experimentos, hiperparámetros y registro de modelos gestionados con **MLflow Model Registry** utilizando el alias `champion`.
- **API de Inferencia**: Construida sobre **FastAPI**, empaquetada en una imagen **Docker** autosuficiente con dependencias estrictamente fijadas.
- **Orquestación y Alta Disponibilidad**: Despliegue en **Kubernetes** con 3 réplicas, balanceo de carga automático y mecanismos de autorreparación habilitados mediante sondas de salud (`liveness` y `readiness` probes).
- **Observabilidad y Calidad de Datos**: Monitoreo de **Data Drift** mediante pruebas estadísticas rigurosas (Kolmogorov-Smirnov y Chi-cuadrado con ajuste de Bonferroni) y detección proactiva de **Concept Drift** sobre el rendimiento del modelo ($ROC\text{-}AUC$).
- **Suite de Pruebas Automáticas**: Pruebas unitarias e integracionales con `pytest` para la verificación del monitoreo y la estabilidad del pipeline.
- **Interfaz de Usuario (UI)**: Panel interactivo en **Streamlit** para inferencia en tiempo real y visualización gráfica de la distribución de carga del clúster.

---

## Problema del Proyecto

En operaciones de logística de comercio electrónico o distribución, predecir si un paquete llegará con retraso (`late`: 1 = tarde, 0 = a tiempo) permite actuar de forma preventiva, reasignando flotas o notificando proactivamente al usuario final.

- **Dataset**: Datos sintéticos de 5.000 pedidos con 8 variables operativas generados mediante `src/generate_data.py` (semilla fija `42` para reproducibilidad).
- **Modelo**: `RandomForestClassifier` de scikit-learn.
- **Métrica Principal**: **$ROC\text{-}AUC$**, justificada por la naturaleza desbalanceada de la variable objetivo (donde la clase "atrasado" representa entre el 20% y el 25% del total).

---

## Estructura del Proyecto

```text
mlops-proyecto/
├── README.md                 # Documentación principal del proyecto
├── requirements.txt          # Dependencias de entrenamiento, monitoreo y UI
├── requirements-api.txt      # Dependencias ligeras fijadas para la imagen Docker de la API
├── Dockerfile                # Receta de construcción de la imagen de inferencia
├── .dockerignore             # Exclusiones para el contexto de construcción de Docker
├── .gitignore                # Exclusiones de control de versiones de Git
├── data/
│   └── reference.csv         # Baseline de datos de entrenamiento (generación automática)
├── src/
│   ├── generate_data.py      # Generador de dataset base y simulación de escenarios de drift
│   ├── train.py              # Ejecución de múltiples runs con búsqueda de hiperparámetros
│   ├── register_model.py     # Registro de versiones en MLflow y asignación de alias champion
│   └── export_model.py       # Descarga y empaquetado del modelo champion hacia model/
├── app/
│   ├── __init__.py
│   └── main.py               # API REST de inferencia (FastAPI)
├── model/                    # Artefacto serializado del modelo (generado localmente, excluido de git)
├── k8s/
│   ├── deployment.yaml       # Configuración del Deployment de Kubernetes (3 réplicas)
│   └── service.yaml          # Exposición del servicio mediante NodePort (puerto 30080)
├── drift/
│   ├── __init__.py
│   ├── monitoring.py         # Algoritmos y lógica de detección de Data Drift y Concept Drift
│   ├── run_data_drift.py     # Script de demostración de escenarios normal (verde) y alterado (rojo)
│   ├── run_concept_drift.py  # Simulación de degradación funcional y generación de métricas
│   └── test_drift.py         # Pruebas automatizadas del módulo de drift (pytest)
└── ui/
    └── app.py                # Dashboard interactivo en Streamlit
```

## Carpetas principales

- `app`: API REST de inferencia (endpoints y servido del modelo).
- `src`: scripts del pipeline (generación de datos, entrenamiento, registro en MLflow y exportación).
- `model`: modelo serializado descargado del registry para la imagen Docker.
- `drift`: scripts y pruebas estadísticas de Data Drift, Concept Drift y tests automatizados con `pytest`.
- `k8s`: manifiestos YAML para el despliegue de 3 réplicas en Kubernetes con NodePort.
- `ui`: dashboard web interactivo en Streamlit.

---

## Modelo y Trazabilidad (MLflow)

El ciclo de vida del modelo se gestiona a través de **MLflow**, garantizando la trazabilidad entre el código de entrenamiento, los parámetros y la versión desplegada en producción.

### Metadata del Despliegue

| Atributo | Valor |
| :--- | :--- |
| **Modelo en Registry** | `late-delivery-classifier` |
| **Versión Desplegada** | `1` (Alias: `champion`) |
| **Run ID de Origen** | `a2eb441b26c245c9bee71704f12e858d` |
| **Imagen Docker** | `late-delivery-api:1.0` |
| **Servicio en Kubernetes** | `http://localhost:30080` (NodePort) |

### Decisión de Arquitectura: Empaquetado Autónomo

> **Nota de Diseño:** El artefacto del modelo `champion` se descarga mediante `src/export_model.py` y se copia internamente al construir la imagen Docker. Esta estrategia garantiza que los Pods de Kubernetes sean **autosuficientes y desacoplados**, permitiendo que la API funcione sin requerir conectividad permanente con el servidor de MLflow. La contrapartida de esta decisión es que la actualización del modelo exige compilar y desplegar una nueva versión de la imagen Docker.

---

## Flujo de Trabajo y Guía de Instalación

### Requisitos Previos

- **Python**: 3.11
- **Motor de Contenedores**: Docker Desktop con la opción de **Kubernetes** habilitada.

### Configuración del Entorno Virtual

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Ejecución del Pipeline por Fases

### Fase 1: Entrenamiento, Experimentos y Registro con MLflow

1. **Iniciar el servidor de MLflow** (Terminal 1):
   ```powershell
   .venv\Scripts\python.exe -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
   ```

2. **Generar datos y ejecutar experimentos** (Terminal 2):
   ```powershell
   .venv\Scripts\python.exe src\generate_data.py
   .venv\Scripts\python.exe src\train.py
   .venv\Scripts\python.exe src\register_model.py
   ```

3. **Verificación en Interfaz UI de MLflow** (`http://127.0.0.1:5000`):
   - **Experimento `late-delivery`**: Se ejecutan 6 *runs* comparables y ordenables por la métrica $ROC\text{-}AUC$.
   - **Registro de Modelos**: Se registran 2 versiones, asignando el alias `champion` a la versión seleccionada para producción.
   - **Parámetros**: Estratifación 80/20, evaluación sobre $ROC\text{-}AUC$, $F_1\text{-}score$ y $Accuracy$.

---

### Fase 2: Empaquetado y Contenedor Docker

1. **Exportar artefacto y construir imagen**:
   ```powershell
   .venv\Scripts\python.exe src\export_model.py
   docker build -t late-delivery-api:1.0 .
   ```

2. **Verificación local del contenedor**:
   ```powershell
   docker run --rm -p 8000:8000 late-delivery-api:1.0
   ```
   - **Documentación Swagger**: `http://localhost:8000/docs`
   - **Endpoint de Información**: GET `/info` (Retorna versión `1` y Run ID).
   - **Ejemplo de Payload para `POST /predict`**:
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

---

### Fase 3: Orquestación en Kubernetes (3 Réplicas)

1. **Desplegar en el clúster**:
   ```powershell
   kubectl config use-context docker-desktop
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   kubectl get pods -w
   ```
   *Asegurarse de que las 3 réplicas alcancen el estado `1/1 Running`.*

2. **Acceso al Servicio**: `http://localhost:30080/info`

---

## Demostraciones Operativas e Infraestructura

El proyecto incluye scripts de comprobación para validar los pilares operacionales en el clúster de Kubernetes (almacenar evidencias en `evidencias/`):

### 1. Alta Disponibilidad (3 Réplicas en Running)
```powershell
kubectl get pods -o wide
kubectl get deployment late-delivery-api
```

### 2. Balanceo de Carga
Comprobación de distribución aleatoria del tráfico entre los pods del clúster:
```powershell
1..30 | ForEach-Object { (curl.exe -s http://localhost:30080/health | ConvertFrom-Json).pod } | Group-Object | Select-Object Name, Count
```

### 3. Resiliencia y Autorreparación (*Self-healing*)
Eliminación selectiva de un Pod en ejecución para verificar que el Deployment recrea de forma inmediata una nueva instancia para mantener el número de réplicas objetivo:
```powershell
kubectl delete pod <nombre-de-un-pod>
kubectl get pods -w
```

### 4. Escalado Dinámico Horizonal
```powershell
# Escalar a 5 réplicas
kubectl scale deployment late-delivery-api --replicas=5
kubectl get pods

# Reducir a 2 réplicas y estabilizar en 3
kubectl scale deployment late-delivery-api --replicas=2
kubectl scale deployment late-delivery-api --replicas=3
```

---

## Observabilidad: Monitoreo de Data Drift y Concept Drift

La evaluación de degradación de datos y concepto se realiza ejecutando el módulo de monitoreo directamente sobre los datos de referencia y los datos de producción simulados:

```powershell
.venv\Scripts\python.exe -m drift.run_data_drift
.venv\Scripts\python.exe -m drift.run_concept_drift
.venv\Scripts\python.exe -m pytest drift -v
```

### Monitoreo de Data Drift

Se emplean pruebas estadísticas adaptadas al tipo de variable:

| Tipo de Variable | Prueba Estadística | Justificación Técnica |
| :--- | :--- | :--- |
| **Numéricas** (6 variables) | Kolmogorov-Smirnov (KS) | Prueba no paramétrica que compara distribuciones continuas completas. |
| **Categóricas** (`carrier`, `shipping_type`) | Chi-cuadrado ($\chi^2$) | Evalúa variaciones significativas en las frecuencias de categorías. |

- **Ajuste de Bonferroni**: Se establece un umbral de p-valor estricto $p < 0.00125$ ($\alpha = 0.01 / 8 \text{ pruebas}$) para mitigar el riesgo de falsos positivos al evaluar múltiples variables en paralelo.
- **Resultados de Simulación**:
  - **Lote Normal (Verde)**: Evalúa 1.000 pedidos generados bajo la misma distribución de origen (Semilla `9`).
  - **Lote de Temporada Alta (Rojo)**: Simula variaciones significativas en distancias, pesos, clima y distribución de operadores logísticos, gatillando alertas de drift estadístico.
  - *Análisis de falsos positivos*: La semilla 7 genera una alerta residual por aleatoriedad ($p = 0.0011$ en `distance_km`), lo cual representa el nivel de error tipo I esperado ($\sim 1\%$).

### Monitoreo de Concept Drift

El *Concept Drift* se manifiesta cuando la relación entre las variables de entrada y la variable objetivo cambia en el tiempo (por ejemplo, cambios de rendimiento en los transportistas).

- **Línea Base ($ROC\text{-}AUC$)**: `0.738` sobre 5.000 registros limpios.
- **Variabilidad Natural**: Desviación estándar $\sigma = 0.0085$ entre lotes no alterados.
- **Criterio de Reentrenamiento**: Caída del $ROC\text{-}AUC > 0.05$ respecto al baseline sostenida durante 2 lotes consecutivos.
- **Resultado del Experimento**: En la simulación (lote 7), el cambio de reglas de negocio degrada la métrica a $\sim 0.66$, disparando exitosamente la alerta de reentrenamiento. *(Artefacto generado en `drift/concept_drift.png`)*.

#### Estrategia frente al Retraso de Etiquetas (*Label Delay*)
En logística, la confirmación de retraso (`late`) solo se conoce tras la entrega efectiva del paquete. Por ello, la estrategia de observabilidad comprende:
1. Vigilancia activa del **Data Drift** y monitoreo de la distribución de probabilidades predichas (métricas sin etiqueta).
2. Cálculo retroactivo del $ROC\text{-}AUC$ a medida que se consolidan las entregas.
3. Alertas tempranas basadas en muestras de etiquetas parciales en pedidos con demoras severas.

---

## Interfaz de Usuario (Streamlit)

Con los Pods ejecutándose en Kubernetes, la interfaz web permite interactuar directamente con la infraestructura desplegada:

```powershell
.venv\Scripts\python.exe -m streamlit run ui/app.py
```

Acceso: `http://localhost:8501`. La aplicación consume el endpoint expuesto en Kubernetes (`http://localhost:30080` vía la variable de entorno `API_URL`). Permite enviar formularios de prueba, consultar metadatos del modelo desplegado y ejecutar pruebas de balanceo de carga en tiempo real.

---

## Matriz de Decisiones de Arquitectura

| Dimensión | Selección Tecnológica | Razón Técnica y Justificación |
| :--- | :--- | :--- |
| **Origen de Datos** | Datos Sintéticos | Facilita la inyección y simulación controlada de fenómenos de *Data Drift* y *Concept Drift*. |
| **Algoritmo** | `RandomForestClassifier` | Ofrece excelente equilibrio entre interpretabilidad, manejo de relaciones no lineales y estabilidad con datos tabulares. |
| **Métrica Principal** | $ROC\text{-}AUC$ | Robusta frente al desbalance de clase objetivo ($\sim 20\text{--}25\%$ de entregas atrasadas), superando limitaciones de *Accuracy*. |
| **Gestión de Registro** | Alias `champion` (MLflow) | Define de forma unívoca y auditable la versión del modelo promovida a producción sin depender de rutas relativas. |
| **Empaquetado** | Contenedor Imutable | Incluye el artefacto del modelo dentro de la imagen Docker para asegurar la total autonomía de los pods de Kubernetes. |
| **Entorno de Ejecución** | Python 3.11 con fijado estricto | Previene errores de deserialización (`pickle`/`joblib`) y conflictos de dependencias entre entornos. |
| **Orquestación** | Kubernetes (3 Réplicas + Probes) | Garantiza tolerancia a fallos, balanceo de carga continuo y enrutamiento exclusivo a Pods en estado saludable. |
| **Monitoreo Estadístico** | Kolmogorov-Smirnov / $\chi^2$ + Bonferroni | Formalismo estadístico riguroso diseñado para minimizar falsas alarmas en monitoreo multivariable. |
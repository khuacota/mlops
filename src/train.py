"""Entrena 6 modelos con distintos hiperparámetros; todo queda registrado en MLflow."""
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
TEST_SIZE = 0.2
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "late-delivery"
TARGET = "late"
CATEGORICAL = ["carrier", "shipping_type"]

# 6 combinaciones de hiperparámetros (mínimo pedido: 5)
GRID = [
    {"n_estimators": 20, "max_depth": 2, "min_samples_leaf": 10},
    {"n_estimators": 50, "max_depth": 3, "min_samples_leaf": 5},
    {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 5},
    {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 2},
    {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 2},
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
]


def build_pipeline(params: dict) -> Pipeline:
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    clf = RandomForestClassifier(**params, random_state=SEED)
    return Pipeline([("prep", prep), ("clf", clf)])


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = pd.read_csv("data/reference.csv")
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    for i, params in enumerate(GRID, start=1):
        with mlflow.start_run(run_name=f"rf-{i}"):
            pipe = build_pipeline(params)
            pipe.fit(X_train, y_train)

            pred = pipe.predict(X_test)
            proba = pipe.predict_proba(X_test)[:, 1]

            mlflow.log_params({**params, "seed": SEED, "test_size": TEST_SIZE})
            mlflow.log_metrics(
                {
                    "accuracy": accuracy_score(y_test, pred),
                    "f1": f1_score(y_test, pred),
                    "roc_auc": roc_auc_score(y_test, proba),
                }
            )
            mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                signature=infer_signature(X_train, pipe.predict(X_train)),
                input_example=X_train.head(3),
            )
            print(f"rf-{i}: {params}  roc_auc={roc_auc_score(y_test, proba):.4f}")


if __name__ == "__main__":
    main()
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators",   type=int,   default=100)
    parser.add_argument("--max-depth",      type=int,   default=5)
    parser.add_argument("--learning-rate",  type=float, default=0.1)
    parser.add_argument("--model-dir",      type=str,   default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train",          type=str,   default=os.environ.get("SM_CHANNEL_TRAIN"))
    args = parser.parse_args()

    # Cargar datos
    files = [f for f in os.listdir(args.train) if f.endswith(".csv")]
    df = pd.concat([pd.read_csv(os.path.join(args.train, f)) for f in files])
    
    # Separar features y target
    target = "price"
    drop_cols = [target, "precio_por_m2"] if "precio_por_m2" in df.columns else [target]
    X = df.drop(columns=drop_cols)
    y = df[target]

    # Encoding de categóricas
    cat_cols = X.select_dtypes(include=["object"]).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Imputar nulos restantes
    X = X.fillna(X.median(numeric_only=True))

    # Split train/test 80-20
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Entrenamiento
    model = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

    # Guardar modelo
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    print("Modelo guardado correctamente.")

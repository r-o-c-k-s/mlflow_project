import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Générer des données factices
X = np.random.rand(100, 1) * 10
y = 3 * X[:, 0] + np.random.randn(100) * 2

# Diviser en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Activer le suivi MLflow
mlflow.set_experiment("Linear Regression Demo")

with mlflow.start_run():
    # Entraîner un modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédictions et métriques
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Enregistrer la métrique et le modèle
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "linear_regression_model")

    print(f"Mean Squared Error: {mse}")

# Utiliser une image Python officielle
FROM python:3.9-slim

# Installer les dépendances nécessaires
RUN pip install --no-cache-dir mlflow pandas scikit-learn

# Créer un répertoire de travail
WORKDIR /app

# Copier les fichiers locaux dans le conteneur
RUN apt-get update && apt-get install -y git
COPY . /app

# Spécifier la commande par défaut
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]

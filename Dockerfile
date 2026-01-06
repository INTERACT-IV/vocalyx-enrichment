# vocalyx-enrichment/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .
# Créer les répertoires nécessaires
RUN mkdir -p /app/logs

# Commande de démarrage du worker Celery
CMD ["celery", "-A", "worker.celery_app", "worker", "--loglevel=info", "--concurrency=2"]


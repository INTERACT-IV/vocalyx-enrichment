# Quick Start - Vocalyx Enrichment

## Installation Rapide

### 1. Installer les dépendances

```bash
cd vocalyx-enrichment
pip3 install -r requirements.txt
```

**Note importante pour llama-cpp-python :**

Pour de meilleures performances CPU, installez avec optimisations :

```bash
# Linux (avec OpenBLAS)
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip3 install llama-cpp-python

# Ou installation standard (plus simple)
pip3 install llama-cpp-python
```

### 2. Vérifier que le modèle existe

Le modèle Phi-3-mini devrait être dans :
```
../shared/models/enrichment/Phi-3-mini-4k-instruct-q4.gguf
```

Vérifier :
```bash
ls -lh ../shared/models/enrichment/Phi-3-mini-4k-instruct-q4.gguf
```

### 3. Tester l'installation

```bash
python3 test_enrichment.py
```

## Configuration Minimale

Créer un fichier `config.ini` minimal :

```ini
[CORE]
instance_name = enrichment-worker-01

[API]
url = http://localhost:8000

[CELERY]
broker_url = redis://localhost:6379/0
result_backend = redis://localhost:6379/0

[LLM]
model = phi-3-mini
device = cpu
n_threads = 0
n_ctx = 2048

[PERFORMANCE]
max_workers = 2
max_chunk_size = 500
enable_cache = true
```

## Dépannage

### Erreur: "No module named 'psutil'"

```bash
pip3 install psutil
```

### Erreur: "llama-cpp-python is not installed"

```bash
pip3 install llama-cpp-python
```

### Erreur: "Model file not found"

Vérifier que le modèle existe :
```bash
python3 scripts/find_model.py phi-3-mini
```

Si le modèle n'est pas trouvé, spécifier le chemin complet dans `config.ini` :
```ini
[LLM]
model = /chemin/absolu/vers/Phi-3-mini-4k-instruct-q4.gguf
```

## Démarrage du Worker

```bash
celery -A worker.celery_app worker \
  --loglevel=info \
  --concurrency=2 \
  --hostname=enrichment-worker-01@%h \
  -Q enrichment
```

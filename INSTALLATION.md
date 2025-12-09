# Guide d'Installation - Vocalyx Enrichment

## 🚀 Installation pour Production CPU-Only

### Prérequis

- Python 3.8+
- 4+ GB RAM (8+ GB recommandé)
- CPU avec support AVX2 (la plupart des CPUs modernes)
- Espace disque : 2-7 GB selon le modèle choisi

### 1. Installation des Dépendances

#### Installation Standard

```bash
pip install -r requirements.txt
```

#### Installation Optimisée pour CPU (Recommandé)

Pour de meilleures performances CPU, installez `llama-cpp-python` avec optimisations :

**Linux (avec OpenBLAS) :**
```bash
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
```

**macOS (Apple Silicon M1/M2) :**
```bash
CMAKE_ARGS="-DLLAMA_METAL=ON" pip install llama-cpp-python
```

**macOS (Intel) ou Linux sans OpenBLAS :**
```bash
pip install llama-cpp-python
```

**Windows :**
```bash
pip install llama-cpp-python
```

### 2. Téléchargement des Modèles

Les modèles GGUF quantisés seront téléchargés automatiquement au premier usage, ou vous pouvez les télécharger manuellement :

#### Option A : Téléchargement Automatique (Recommandé)

Le modèle sera téléchargé automatiquement depuis Hugging Face Hub lors du premier chargement.

#### Option B : Téléchargement Manuel

```bash
# Installer huggingface_hub si pas déjà fait
pip install huggingface-hub

# Télécharger un modèle recommandé
python -c "
from infrastructure.models.model_manager import ModelManager
manager = ModelManager('./models/enrichment')
manager.download_model('phi-3-mini')
"
```

#### Modèles Recommandés

| Modèle | Taille | RAM Requise | Vitesse | Qualité |
|--------|--------|-------------|---------|---------|
| **phi-3-mini** | 2.3 GB | 4 GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **mistral-7b-instruct** | 4.1 GB | 6 GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **llama-3-8b-instruct** | 4.6 GB | 6 GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **phi-3-medium** | 7.0 GB | 8 GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **gemma-2b** | 1.4 GB | 3 GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**Recommandation pour production CPU :** `phi-3-mini` (bon équilibre vitesse/qualité)

### 3. Configuration

Copier `config.ini.example` vers `config.ini` :

```bash
cp config.ini.example config.ini
```

Éditer `config.ini` et configurer :

```ini
[LLM]
# Modèle à utiliser (nom ou chemin)
model = phi-3-mini

# Paramètres CPU
n_threads = 0  # 0 = auto-détection (recommandé)
n_ctx = 2048   # Taille du contexte
n_batch = 512  # Batch size

[PERFORMANCE]
max_workers = 2
max_chunk_size = 500
enable_cache = true
cache_max_models = 2
```

### 4. Test de l'Installation

```bash
# Tester le chargement du modèle
python -c "
from enrichment_service import EnrichmentService
from config import Config

config = Config()
service = EnrichmentService(config, 'phi-3-mini')
result = service.enrich_text('Bonjour, comment allez-vous ?')
print(f'Résultat: {result}')
"
```

### 5. Démarrage du Worker

```bash
celery -A worker.celery_app worker \
  --loglevel=info \
  --concurrency=2 \
  --hostname=enrichment-worker-01@%h \
  --without-gossip \
  --without-mingle \
  -Q enrichment
```

## 🐳 Installation Docker

### Dockerfile Optimisé CPU

Créer un `Dockerfile` :

```dockerfile
FROM python:3.11-slim

# Installer les dépendances système pour llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier les requirements
COPY requirements.txt .

# Installer llama-cpp-python avec optimisations CPU
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    pip install --no-cache-dir llama-cpp-python

# Installer les autres dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Créer le répertoire des modèles
RUN mkdir -p /app/models/enrichment

# Exposer les volumes
VOLUME ["/app/models/enrichment", "/app/logs"]

CMD ["celery", "-A", "worker.celery_app", "worker", \
     "--loglevel=info", "--concurrency=2", \
     "--hostname=enrichment-worker-01@%h", \
     "--without-gossip", "--without-mingle", "-Q", "enrichment"]
```

### Build et Run

```bash
docker build -t vocalyx-enrichment .
docker run -d \
  -v ./models/enrichment:/app/models/enrichment \
  -v ./logs:/app/logs \
  -e LLM_MODEL=phi-3-mini \
  -e MAX_WORKERS=2 \
  vocalyx-enrichment
```

## 🔧 Optimisations CPU

### Paramètres Recommandés

Pour un CPU avec N cores :

```ini
[LLM]
n_threads = N-1  # Laisser 1 core libre
n_ctx = 2048     # Contexte suffisant pour la plupart des cas
n_batch = 512    # Bon équilibre mémoire/vitesse
```

### Monitoring des Performances

```python
import psutil
import time

# Avant enrichissement
mem_before = psutil.virtual_memory()
start = time.time()

# Enrichissement
result = service.enrich_text(text)

# Après enrichissement
mem_after = psutil.virtual_memory()
elapsed = time.time() - start

print(f"Temps: {elapsed:.2f}s")
print(f"Mémoire: {mem_after.used - mem_before.used} MB")
```

## 🐛 Dépannage

### Erreur: "llama-cpp-python is not installed"

```bash
pip install llama-cpp-python
```

### Erreur: "Model file not found"

1. Vérifier que le modèle est téléchargé :
```bash
ls -lh models/enrichment/*.gguf
```

2. Télécharger manuellement :
```bash
python -c "
from infrastructure.models.model_manager import ModelManager
ModelManager('./models/enrichment').download_model('phi-3-mini')
"
```

### Performance lente

1. Vérifier le nombre de threads :
```python
import os
print(f"CPU cores: {os.cpu_count()}")
```

2. Augmenter `n_batch` (si RAM suffisante) :
```ini
n_batch = 1024
```

3. Réduire `n_ctx` (si contexte court suffit) :
```ini
n_ctx = 1024
```

### Mémoire insuffisante

1. Utiliser un modèle plus petit (gemma-2b au lieu de phi-3-mini)
2. Réduire `n_ctx` et `n_batch`
3. Réduire `cache_max_models` à 1

## 📚 Ressources

- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-python GitHub](https://github.com/abetlen/llama-cpp-python)
- [Hugging Face GGUF Models](https://huggingface.co/models?library=gguf)

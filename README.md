# Vocalyx Enrichment Worker

Worker Celery pour l'enrichissement de transcriptions avec modèles LLM, optimisé pour CPU avec architecture distribuée.

## 🚀 Fonctionnalités

- **Architecture Distribuée** : Découpage intelligent en chunks et traitement parallèle
- **Cache de Modèles LRU** : Réutilisation des modèles LLM pour économiser 10-30s par requête
- **Redis pour Agrégation** : Stockage temporaire avec compression (60-70% d'économie mémoire)
- **Backend LLM Production-Ready** : `llama-cpp-python` avec modèles GGUF quantisés (CPU-only)
- **Optimisations CPU** : Quantisation Q4_K_M, threading optimisé, batch processing
- **Scalabilité** : Workers partagés via Celery, distribution automatique

## 📋 Prérequis

- Python 3.8+
- Redis (DB 3 dédiée pour l'enrichissement)
- CPU avec support AVX2 (la plupart des CPUs modernes)
- 4+ GB RAM (8+ GB recommandé pour meilleures performances)

## 🔧 Installation

Voir le guide complet : [INSTALLATION.md](INSTALLATION.md)

### Installation Rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Installer llama-cpp-python avec optimisations CPU (Linux)
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python

# 3. Télécharger un modèle (optionnel, sera fait automatiquement)
python scripts/download_model.py phi-3-mini

# 4. Configurer
cp config.ini.example config.ini
# Éditer config.ini selon vos besoins
```

## ⚙️ Configuration

Copier `config.ini.example` vers `config.ini` et adapter :

```ini
[LLM]
model = phi-3-mini  # ou chemin vers fichier .gguf
n_threads = 0       # 0 = auto-détection
n_ctx = 2048        # Taille du contexte
n_batch = 512       # Batch size

[PERFORMANCE]
max_workers = 2
max_chunk_size = 500
enable_cache = true
cache_max_models = 2
```

## 🏃 Démarrage

### Mode développement
```bash
celery -A worker.celery_app worker --loglevel=info --concurrency=2 -Q enrichment
```

### Mode production (Docker)
Voir `docker-compose.yml` pour la configuration complète.

## 📊 Architecture

### Mode Classique (petites transcriptions)
```
API → enrich_transcription_task → EnrichmentService → Résultat
```

### Mode Distribué (grandes transcriptions)
```
API → orchestrate_distributed_enrichment_task
  ↓
Découpage en chunks intelligents
  ↓
enrich_chunk_task (×N workers en parallèle)
  ↓
Redis (stockage temporaire)
  ↓
aggregate_enrichment_chunks_task
  ↓
Résultat final
```

## 🎯 Backend LLM : llama-cpp-python

### Modèles Supportés

Le worker utilise `llama-cpp-python` avec des modèles GGUF quantisés :

- **Phi-3 Mini** (recommandé) : 2.3 GB, rapide, bonne qualité
- **Mistral 7B Instruct** : 4.1 GB, excellente qualité
- **Llama 3 8B Instruct** : 4.6 GB, excellente qualité
- **Phi-3 Medium** : 7.0 GB, meilleure qualité, plus lent
- **Gemma 2B** : 1.4 GB, très léger, très rapide

### Optimisations CPU

1. **Quantisation GGUF Q4_K_M** : Réduit la taille mémoire de 4x
2. **Threading optimisé** : Utilise tous les cores CPU disponibles
3. **Memory mapping** : Économise la RAM
4. **Batch processing** : Traite plusieurs tokens en parallèle
5. **OpenBLAS** : Accélération mathématique (optionnel)

## 📈 Performance

### Gains attendus
- **Distribution (4 workers)** : 4x plus rapide
- **Cache de modèles** : 10-30s économisées par requête
- **Quantisation GGUF** : 4x moins de mémoire, 2-3x plus rapide
- **TOTAL** : 6-10x plus rapide

### Exemple
- Transcription de 100 segments
- Mode séquentiel : 200s
- Mode distribué (4 workers) : 50s
- **Accélération : 4x**

### Benchmarks CPU (Phi-3 Mini Q4_K_M)

| CPU | Cores | Tokens/s | RAM |
|-----|-------|----------|-----|
| Intel i7-12700 | 12 | ~25-30 | 3.5 GB |
| AMD Ryzen 7 5800X | 8 | ~20-25 | 3.5 GB |
| Apple M1 | 8 | ~30-35 | 3.5 GB |
| Apple M2 | 8 | ~35-40 | 3.5 GB |

## 🔍 Monitoring

Le worker expose des métriques de santé via Celery :

```python
from celery import current_app
inspect = current_app.control.inspect()
stats = inspect.stats()
```

## 📚 Structure

```
vocalyx-enrichment/
├── worker.py                          # Worker principal avec tâches Celery
├── enrichment_service.py              # Service d'enrichissement LLM
├── config.py                          # Configuration
├── config.ini.example                 # Exemple de configuration
├── INSTALLATION.md                    # Guide d'installation détaillé
├── scripts/
│   └── download_model.py              # Script de téléchargement de modèles
├── infrastructure/
│   ├── models/
│   │   ├── llm_model_cache.py         # Cache LRU des modèles
│   │   └── model_manager.py           # Gestionnaire de modèles (téléchargement)
│   ├── redis/
│   │   └── redis_enrichment_manager.py # Gestionnaire Redis
│   └── api/
│       └── api_client.py              # Client API
└── application/
    └── services/
        └── chunk_splitter.py          # Découpage intelligent
```

## 🐛 Dépannage

### Le modèle ne se charge pas
- Vérifier que le fichier .gguf existe dans `models/enrichment/`
- Vérifier que le modèle est quantisé (GGUF format)
- Vérifier la mémoire disponible (minimum 4 GB)

### Performance lente
- Augmenter le nombre de workers
- Réduire `max_chunk_size` pour plus de parallélisation
- Vérifier que le cache est activé
- Vérifier que `n_threads` est correctement configuré

### Erreur de mémoire
- Utiliser un modèle plus petit (gemma-2b)
- Réduire `n_ctx` et `n_batch`
- Réduire `cache_max_models` à 1

## 📝 Notes

- Les modèles GGUF sont déjà quantisés (Q4_K_M par défaut)
- Le cache de modèles limite la RAM utilisée (max_models=2 par défaut)
- La compression Redis réduit la mémoire mais ajoute un léger overhead CPU
- Pour production, utiliser `phi-3-mini` ou `mistral-7b-instruct` selon les besoins

## 🔗 Ressources

- [Guide d'Installation](INSTALLATION.md)
- [Documentation llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Modèles GGUF sur Hugging Face](https://huggingface.co/models?library=gguf)

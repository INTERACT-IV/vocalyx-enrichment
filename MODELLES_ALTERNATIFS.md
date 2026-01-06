# 🚀 Modèles LLM Alternatifs - Plus Rapides et Plus Précis que Mistral 7B

Ce document liste les modèles GGUF disponibles qui sont **plus rapides** et/ou **plus précis** que Mistral 7B Instruct pour l'enrichissement sur CPU.

## 📊 Comparaison des Modèles

| Modèle | Taille | Vitesse vs Mistral | Qualité vs Mistral | Meilleur pour |
|--------|--------|-------------------|-------------------|---------------|
| **Phi-3 Mini Q3** | 1.8 GB | **3-4x plus rapide** | Légèrement inférieur | Tâches rapides, CPU limité |
| **Phi-3 Mini Q4** | 2.3 GB | **2-3x plus rapide** | Légèrement inférieur | Équilibre vitesse/qualité (défaut) |
| **Gemma 2B** | 1.4 GB | **4-5x plus rapide** | Inférieur | Tâches très simples, très rapide |
| **Mistral 7B Q4_0** | 3.8 GB | **1.2-1.5x plus rapide** | Similaire | Alternative directe à Mistral |
| **Mistral 7B Q4_K_M** | 4.1 GB | 1x (référence) | Référence | Équilibre actuel |
| **Qwen 2.5 7B** | 4.1 GB | Similaire | **Meilleur pour français** | Transcription française |
| **Llama 3 8B** | 4.8 GB | Légèrement plus lent | **Plus précis** | Qualité maximale |
| **Gemma 7B** | 4.6 GB | Similaire à plus rapide | Similaire | Alternative moderne |
| **Phi-3 Medium** | 7.8 GB | Plus lent | **Plus précis** | Qualité maximale (lourd) |

## 🎯 Recommandations par Cas d'Usage

### ⚡ Priorité Vitesse (2-4x plus rapide)
1. **`phi-3-mini-q3`** - Le plus rapide avec qualité acceptable
2. **`phi-3-mini`** - Bon compromis vitesse/qualité (défaut actuel)
3. **`gemma-2b`** - Très rapide mais qualité réduite

### 🎯 Priorité Qualité (plus précis que Mistral)
1. **`llama-3-8b-instruct`** - Meilleur équilibre qualité/vitesse
2. **`qwen2.5-7b-instruct`** - Excellent pour le français
3. **`phi-3-medium`** - Le plus précis mais plus lent

### ⚖️ Équilibre Vitesse/Qualité
1. **`mistral-7b-instruct-q4_0`** - Version plus rapide de Mistral
2. **`gemma-7b`** - Alternative moderne à Mistral

## 🔧 Comment Changer de Modèle

### Option 1: Variable d'environnement (recommandé)
```bash
# Dans docker-compose.yml ou .env
LLM_MODEL=llama-3-8b-instruct
```

### Option 2: Fichier config.ini
```ini
[LLM]
model = llama-3-8b-instruct
```

### Option 3: Télécharger un nouveau modèle
```bash
# Télécharger Llama 3 8B
python scripts/download_model.py llama-3-8b-instruct

# Télécharger Qwen 2.5 7B (excellent pour français)
python scripts/download_model.py qwen2.5-7b-instruct
```

## 📥 Téléchargement des Modèles

Tous les modèles peuvent être téléchargés automatiquement via le script :

```bash
cd vocalyx-enrichment
python scripts/download_model.py <nom_du_modele>
```

Exemples :
```bash
# Modèle rapide
python scripts/download_model.py phi-3-mini-q3

# Modèle précis
python scripts/download_model.py llama-3-8b-instruct

# Modèle optimisé pour français
python scripts/download_model.py qwen2.5-7b-instruct
```

## 🧪 Tests de Performance

Pour tester les performances d'un modèle :

```bash
python test_enrichment.py --model <nom_du_modele>
```

## 💡 Conseils d'Optimisation

### Pour CPU avec 4-8 cores et 8-16 GB RAM
- **Recommandé** : `phi-3-mini` ou `llama-3-8b-instruct`
- **Éviter** : `phi-3-medium` (trop lourd)

### Pour CPU avec 8+ cores et 16+ GB RAM
- **Recommandé** : `llama-3-8b-instruct` ou `qwen2.5-7b-instruct`
- **Alternative rapide** : `mistral-7b-instruct-q4_0`

### Pour CPU limité (2-4 cores, 4-8 GB RAM)
- **Recommandé** : `phi-3-mini-q3` ou `gemma-2b`
- **Éviter** : Modèles 7B+ (trop lourds)

## 📈 Benchmarks Approximatifs

Basés sur des tests CPU typiques (8 cores, 16 GB RAM) :

| Modèle | Temps enrichissement* | Qualité** |
|--------|----------------------|-----------|
| Phi-3 Mini Q3 | ~3-5s | 7/10 |
| Phi-3 Mini Q4 | ~4-6s | 8/10 |
| Mistral 7B Q4_K_M | ~10-15s | 9/10 |
| Mistral 7B Q4_0 | ~8-12s | 9/10 |
| Llama 3 8B | ~12-18s | 9.5/10 |
| Qwen 2.5 7B | ~10-15s | 9/10 (10/10 pour français) |

*Temps pour enrichir une transcription de ~500 mots (titre + résumé + score + bullets)
**Qualité subjective basée sur la précision et la cohérence

## 🔄 Migration depuis Mistral 7B

### Vers un modèle plus rapide
```bash
# 1. Télécharger le nouveau modèle
python scripts/download_model.py phi-3-mini-q3

# 2. Modifier la configuration
export LLM_MODEL=phi-3-mini-q3

# 3. Redémarrer le worker
docker-compose restart vocalyx-enrichment-01
```

### Vers un modèle plus précis
```bash
# 1. Télécharger le nouveau modèle
python scripts/download_model.py llama-3-8b-instruct

# 2. Modifier la configuration
export LLM_MODEL=llama-3-8b-instruct

# 3. Redémarrer le worker
docker-compose restart vocalyx-enrichment-01
```

## ⚠️ Notes Importantes

1. **Premier chargement** : Le premier chargement d'un modèle peut prendre 10-30 secondes
2. **Mémoire** : Les modèles 7B+ nécessitent au moins 8 GB RAM
3. **Cache** : Les modèles sont mis en cache pour éviter les rechargements
4. **Compatibilité** : Tous les modèles utilisent le format GGUF (compatible llama.cpp)

## 🆘 Dépannage

### Le modèle ne se charge pas
- Vérifier que le modèle est téléchargé : `python scripts/find_model.py <nom>`
- Vérifier les logs : `docker-compose logs vocalyx-enrichment-01`

### Erreur de mémoire
- Utiliser un modèle plus petit (Phi-3 Mini ou Gemma 2B)
- Réduire `n_ctx` dans la configuration (ex: 1024 au lieu de 2048)

### Modèle trop lent
- Utiliser une version Q3 au lieu de Q4
- Utiliser `phi-3-mini-q3` ou `gemma-2b`


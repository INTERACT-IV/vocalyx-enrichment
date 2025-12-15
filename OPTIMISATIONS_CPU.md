# 🚀 Pistes d'amélioration pour accélérer l'enrichissement sur CPU

Ce document liste les principales pistes d'optimisation pour améliorer les performances de l'enrichissement sur CPU.

## 📊 Analyse de l'état actuel

### Points de blocage identifiés

1. **Génération séquentielle des métadonnées** (worker.py:434-496)
   - Titre, résumé, satisfaction et bullet_points sont générés un par un
   - Temps total = somme des temps individuels
   - Impact : ~4x plus lent que nécessaire

2. **Correction de texte segment par segment** (enrichment_service.py:466-544)
   - Chaque segment est traité individuellement dans une boucle
   - Pas de traitement par batch
   - Impact : Latence élevée pour les transcriptions longues

3. **Paramètres CPU non optimisés**
   - `n_threads` : Auto-détection (CPU_COUNT - 1)
   - `n_batch` : 512 (peut être augmenté)
   - `n_ctx` : 2048 (peut être réduit selon les besoins)

4. **Pas de cache de résultats**
   - Même texte = même traitement répété
   - Pas de mise en cache des métadonnées générées

5. **Taille du contexte non optimisée**
   - Texte complet envoyé pour toutes les métadonnées
   - Certaines tâches (titre) n'ont besoin que d'un échantillon

---

## 🎯 Pistes d'amélioration (par ordre de priorité)

### 1. ⚡ Parallélisation de la génération des métadonnées

**Impact estimé : 3-4x plus rapide pour les métadonnées**

**Problème actuel :**
```python
# worker.py:434-496 - Génération séquentielle
title = enrichment_service.generate_metadata(...)  # ~2-5s
summary = enrichment_service.generate_metadata(...)  # ~3-7s
satisfaction = enrichment_service.generate_metadata(...)  # ~2-5s
bullet_points = enrichment_service.generate_metadata(...)  # ~3-7s
# Total: ~10-24s
```

**Solution :**
Utiliser `ThreadPoolExecutor` ou `concurrent.futures` pour paralléliser les 4 appels LLM.

**Implémentation :**
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_metadata_parallel(enrichment_service, text, final_prompts):
    """Génère les métadonnées en parallèle"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(enrichment_service.generate_metadata, text, "title", final_prompts, 50): "title",
            executor.submit(enrichment_service.generate_metadata, text, "summary", final_prompts, 150): "summary",
            executor.submit(enrichment_service.generate_metadata, text, "satisfaction", final_prompts, 100): "satisfaction",
            executor.submit(enrichment_service.generate_metadata, text, "bullet_points", final_prompts, 200): "bullet_points"
        }
        
        results = {}
        for future in as_completed(futures):
            task_type = futures[future]
            try:
                results[task_type] = future.result()
            except Exception as e:
                logger.warning(f"Failed to generate {task_type}: {e}")
                results[task_type] = None
        return results
```

**Gain attendu :** 3-4x plus rapide (de ~15s à ~5s)

---

### 2. 📦 Traitement par batch des segments

**Impact estimé : 2-3x plus rapide pour la correction de texte**

**Problème actuel :**
```python
# enrichment_service.py:487-536 - Traitement séquentiel
for segment in segments:
    enriched_text = self._generate_text(prompt, ...)  # Appel LLM par segment
```

**Solution :**
Grouper plusieurs segments courts en un seul batch pour réduire le nombre d'appels LLM.

**Implémentation :**
```python
def enrich_segments_batch(self, segments, batch_size=5):
    """Enrichit les segments par batch"""
    enriched_segments = []
    
    # Grouper les segments en batches
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]
        batch_text = "\n".join([f"Segment {j+1}: {seg.get('text', '')}" for j, seg in enumerate(batch)])
        
        # Un seul appel LLM pour le batch
        corrected_batch = self._generate_text(
            f"{base_instructions}\n\nCorrige ces segments:\n{batch_text}",
            max_tokens=len(batch_text.split()) * 2,
            temperature=0.05
        )
        
        # Parser et distribuer les résultats
        corrected_segments = self._parse_batch_response(corrected_batch, batch)
        enriched_segments.extend(corrected_segments)
    
    return enriched_segments
```

**Gain attendu :** 2-3x plus rapide pour les transcriptions avec beaucoup de segments courts

---

### 3. 🔧 Optimisation des paramètres CPU

**Impact estimé : 10-30% d'amélioration**

**Paramètres à ajuster :**

#### a) `n_threads` - Nombre de threads
```python
# Actuel: max(1, os.cpu_count() - 1)
# Optimisé: Utiliser tous les cores disponibles si RAM suffisante
self.n_threads = os.cpu_count() or 4  # Utiliser tous les cores
```

**Note :** Pour les modèles GGUF, utiliser tous les cores peut être bénéfique si la RAM est suffisante.

#### b) `n_batch` - Taille du batch
```python
# Actuel: 512
# Optimisé: Augmenter si RAM disponible
self.n_batch = 1024  # ou 2048 si RAM > 16GB
```

**Impact :** Réduit la latence en traitant plus de tokens à la fois.

#### c) `n_ctx` - Taille du contexte
```python
# Actuel: 2048
# Optimisé: Réduire selon les besoins
self.n_ctx = 1024  # Pour les transcriptions courtes (< 500 mots)
# ou 1536 pour un compromis
```

**Impact :** Réduit la mémoire utilisée et accélère le traitement.

**Configuration recommandée :**
```python
# Pour CPU avec 8+ cores et 16GB+ RAM
n_threads = os.cpu_count()
n_batch = 1024
n_ctx = 1536

# Pour CPU avec 4 cores et 8GB RAM
n_threads = 3  # Laisser 1 core libre
n_batch = 512
n_ctx = 1024
```

---

### 4. 🎯 Réduction intelligente du contexte

**Impact estimé : 20-40% d'amélioration pour certaines tâches**

**Problème actuel :**
- Le texte complet est envoyé pour toutes les métadonnées
- Le titre n'a besoin que d'un échantillon (déjà fait : `text[:500]`)
- Le résumé peut utiliser un échantillon intelligent

**Solution :**
```python
def get_smart_sample(text, task_type, max_chars=1000):
    """Extrait un échantillon intelligent du texte"""
    if task_type == "title":
        # Prendre le début (contexte initial)
        return text[:500]
    elif task_type == "summary":
        # Prendre début + milieu + fin (structure narrative)
        if len(text) <= max_chars:
            return text
        third = len(text) // 3
        return f"{text[:third]}...\n\n{text[third:2*third]}...\n\n{text[2*third:]}"
    elif task_type == "satisfaction":
        # Prendre le texte complet (analyse globale nécessaire)
        return text[:2000]  # Limiter quand même
    else:
        return text[:max_chars]
```

**Gain attendu :** Réduction de 20-40% du temps de traitement pour les transcriptions longues

---

### 5. 💾 Cache de résultats

**Impact estimé : 100% plus rapide pour les textes identiques**

**Solution :**
Mettre en cache les résultats d'enrichissement basés sur un hash du texte.

**Implémentation :**
```python
import hashlib
import json
from functools import lru_cache

class EnrichmentCache:
    def __init__(self, redis_client=None, ttl=3600):
        self.redis = redis_client
        self.ttl = ttl
        self.local_cache = {}  # Cache local (LRU)
    
    def _get_hash(self, text, task_type):
        """Génère un hash du texte + type de tâche"""
        content = f"{task_type}:{text[:500]}"  # Limiter pour le hash
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text, task_type):
        """Récupère un résultat depuis le cache"""
        cache_key = f"enrichment:{self._get_hash(text, task_type)}"
        
        # Vérifier le cache local
        if cache_key in self.local_cache:
            return self.local_cache[cache_key]
        
        # Vérifier Redis
        if self.redis:
            cached = self.redis.get(cache_key)
            if cached:
                result = json.loads(cached)
                self.local_cache[cache_key] = result
                return result
        
        return None
    
    def set(self, text, task_type, result):
        """Stocke un résultat dans le cache"""
        cache_key = f"enrichment:{self._get_hash(text, task_type)}"
        
        # Stocker localement
        self.local_cache[cache_key] = result
        
        # Stocker dans Redis
        if self.redis:
            self.redis.setex(
                cache_key,
                self.ttl,
                json.dumps(result)
            )
```

**Utilisation :**
```python
def generate_metadata(self, text, task_type, prompts, max_tokens=100):
    # Vérifier le cache
    cached = self.cache.get(text, task_type)
    if cached:
        logger.debug(f"Cache hit for {task_type}")
        return cached
    
    # Générer normalement
    result = self._generate_text(...)
    
    # Mettre en cache
    self.cache.set(text, task_type, result)
    return result
```

**Gain attendu :** Instantané pour les textes déjà traités

---

### 6. 🔄 Optimisation des prompts

**Impact estimé : 10-20% d'amélioration**

**Problème actuel :**
Les prompts sont assez longs et répétitifs.

**Solution :**
- Réduire la taille des prompts
- Utiliser des templates plus courts
- Éviter les répétitions

**Exemple :**
```python
# Avant (58 mots)
DEFAULT_ENRICHMENT_PROMPTS = {
    "title": "Cette transcription provient d'un appel entre un client (appelant) et un agent de support client. Génère un titre court et accrocheur (maximum 10 mots) pour cette transcription d'appel client. IMPORTANT: Réponds UNIQUEMENT en français.",
}

# Après (25 mots - 57% plus court)
DEFAULT_ENRICHMENT_PROMPTS = {
    "title": "Appel client-agent. Génère un titre court (max 10 mots) en français:",
}
```

**Gain attendu :** 10-20% de réduction du temps de génération

---

### 7. 🧵 Utilisation de modèles plus légers/quantifiés

**Impact estimé : 2-3x plus rapide**

**Solution :**
Utiliser des modèles plus quantifiés (Q3, Q4) au lieu de Q4_K_M.

**Modèles recommandés :**
- `phi-3-mini-4k-instruct-q3_K_M.gguf` (plus rapide que Q4)
- `phi-3-mini-4k-instruct-q4_0.gguf` (plus rapide que Q4_K_M)

**Trade-off :** Légère baisse de qualité pour gain de vitesse significatif.

**Gain attendu :** 2-3x plus rapide avec perte de qualité minime (< 5%)

---

### 8. 📊 Pré-filtrage des segments vides

**Impact estimé : 5-10% d'amélioration**

**Solution :**
Filtrer les segments vides avant le traitement.

```python
def enrich_segments(self, segments, ...):
    # Filtrer les segments vides
    valid_segments = [s for s in segments if s.get('text', '').strip()]
    empty_segments = [s for s in segments if not s.get('text', '').strip()]
    
    # Traiter uniquement les segments valides
    enriched_valid = self._process_segments(valid_segments, ...)
    
    # Réassembler avec les segments vides
    return self._merge_segments(enriched_valid, empty_segments)
```

**Gain attendu :** Évite les appels LLM inutiles

---

### 9. 🔀 Parallélisation au niveau Celery

**Impact estimé : Déjà implémenté (mode distribué)**

**État actuel :**
Le mode distribué existe déjà et fonctionne bien. Les chunks sont traités en parallèle par différents workers.

**Amélioration possible :**
- Réduire le seuil de distribution (actuellement 10 segments)
- Optimiser la taille des chunks pour un meilleur équilibre

---

### 10. 🎛️ Optimisation de la température

**Impact estimé : 5-10% d'amélioration**

**Solution :**
Réduire la température pour les tâches déterministes (correction, satisfaction).

```python
# Actuel
temperature = 0.7  # Pour toutes les tâches

# Optimisé
temperature_map = {
    "title": 0.5,  # Plus déterministe
    "summary": 0.6,
    "satisfaction": 0.3,  # Très déterministe (score)
    "bullet_points": 0.5,
    "correction": 0.05  # Déjà fait
}
```

**Gain attendu :** Génération plus rapide et plus cohérente

---

## 📈 Synthèse des gains attendus

| Optimisation | Gain estimé | Priorité | Complexité |
|-------------|-------------|----------|------------|
| 1. Parallélisation métadonnées | 3-4x | 🔴 Haute | Moyenne |
| 2. Batch processing segments | 2-3x | 🟠 Moyenne | Moyenne |
| 3. Optimisation paramètres CPU | 10-30% | 🟠 Moyenne | Faible |
| 4. Réduction contexte | 20-40% | 🟡 Faible | Faible |
| 5. Cache de résultats | 100% (cache hit) | 🟡 Faible | Moyenne |
| 6. Optimisation prompts | 10-20% | 🟡 Faible | Faible |
| 7. Modèles plus légers | 2-3x | 🟠 Moyenne | Faible |
| 8. Pré-filtrage | 5-10% | 🟢 Très faible | Très faible |
| 9. Parallélisation Celery | Déjà fait | - | - |
| 10. Optimisation température | 5-10% | 🟢 Très faible | Très faible |

**Gain total potentiel :** 5-10x plus rapide avec les optimisations prioritaires (1, 2, 3, 7)

---

## 🚀 Plan d'implémentation recommandé

### Phase 1 - Quick wins (1-2 jours)
1. ✅ Optimisation paramètres CPU (#3)
2. ✅ Réduction contexte (#4)
3. ✅ Optimisation prompts (#6)
4. ✅ Pré-filtrage segments (#8)
5. ✅ Optimisation température (#10)

**Gain attendu :** 30-50% d'amélioration

### Phase 2 - Optimisations majeures (3-5 jours)
1. ✅ Parallélisation métadonnées (#1)
2. ✅ Batch processing segments (#2)
3. ✅ Modèles plus légers (#7)

**Gain attendu :** 5-8x plus rapide au total

### Phase 3 - Optimisations avancées (2-3 jours)
1. ✅ Cache de résultats (#5)

**Gain attendu :** Amélioration supplémentaire pour les cas répétitifs

---

## 📝 Notes importantes

1. **Compatibilité CPU :** Toutes ces optimisations sont compatibles avec CPU uniquement
2. **Mémoire :** Augmenter `n_batch` nécessite plus de RAM
3. **Qualité :** Réduire la température et utiliser des modèles plus légers peut légèrement affecter la qualité
4. **Tests :** Tester chaque optimisation individuellement pour mesurer l'impact réel

---

## 🔍 Monitoring recommandé

Ajouter des métriques pour mesurer :
- Temps de génération par type de métadonnée
- Taux de cache hit
- Utilisation CPU/RAM
- Temps total d'enrichissement

Cela permettra de valider les gains réels et d'ajuster les paramètres.

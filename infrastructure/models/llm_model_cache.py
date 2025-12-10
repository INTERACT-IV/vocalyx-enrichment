"""
Cache de modèles LLM avec stratégie LRU
Optimisé pour CPU avec quantisation int8/int4
"""

import logging
import time
import threading
from typing import Optional, Dict

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger("vocalyx")


class LLMModelCache:
    """Cache LRU pour les modèles LLM (Mistral, Phi-3, etc.)"""
    
    def __init__(self, max_models: int = 2, enrichment_service_class=None):
        """
        Initialise le cache de modèles LLM.
        
        Args:
            max_models: Nombre maximum de modèles à garder en cache (défaut: 2 pour économiser la RAM)
            enrichment_service_class: Classe EnrichmentService (passée depuis worker.py pour éviter les problèmes d'import)
        """
        self.max_models = max_models
        self._cache: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._enrichment_service_class = enrichment_service_class
    
    def get(self, model_name: str, config) -> Optional[object]:
        """
        Récupère un modèle depuis le cache ou le charge si nécessaire.
        
        Args:
            model_name: Nom du modèle (normalisé)
            config: Configuration du worker
            
        Returns:
            Service LLM avec le modèle chargé
        """
        normalized_name = self._normalize_model_name(model_name)
        
        with self._lock:
            if normalized_name in self._cache:
                logger.info(f"✅ Using cached LLM model: {normalized_name}")
                self._cache[normalized_name]['last_used'] = time.time()
                return self._cache[normalized_name]['service']
            
            # Vérifier la mémoire disponible avant de charger un nouveau modèle (si psutil disponible)
            if PSUTIL_AVAILABLE:
                mem_percent = psutil.virtual_memory().percent
                if mem_percent > 85:
                    logger.warning(f"⚠️ High memory usage ({mem_percent:.1f}%), evicting LRU model before loading new one")
                    if len(self._cache) >= 1:  # Éviction agressive si mémoire faible
                        self._evict_lru()
            
            # Si le cache est plein, supprimer le moins récemment utilisé
            if len(self._cache) >= self.max_models:
                self._evict_lru()
            
            # Charger le nouveau modèle
            logger.info(f"🚀 Loading LLM model into cache: {normalized_name} (cache: {len(self._cache)}/{self.max_models})")
            try:
                # Utiliser la classe passée en paramètre si disponible, sinon importer
                if self._enrichment_service_class is not None:
                    EnrichmentService = self._enrichment_service_class
                else:
                    # Import dynamique pour éviter les dépendances circulaires
                    # Essayer plusieurs chemins d'import possibles
                    try:
                        from enrichment_service import EnrichmentService
                    except ImportError:
                        # Si l'import direct échoue, essayer avec le chemin absolu
                        import sys
                        from pathlib import Path
                        # Ajouter le répertoire parent au path si nécessaire
                        current_file = Path(__file__).resolve()
                        parent_dir = current_file.parent.parent.parent
                        if str(parent_dir) not in sys.path:
                            sys.path.insert(0, str(parent_dir))
                        try:
                            from enrichment_service import EnrichmentService
                        except ImportError:
                            # Dernier recours : import depuis le répertoire de travail
                            import os
                            cwd = os.getcwd()
                            if cwd not in sys.path:
                                sys.path.insert(0, cwd)
                            from enrichment_service import EnrichmentService
                
                service = EnrichmentService(config, model_name=normalized_name)
                self._cache[normalized_name] = {
                    'service': service,
                    'last_used': time.time()
                }
                logger.info(f"✅ Model {normalized_name} loaded and cached successfully")
                return service
            except Exception as e:
                logger.error(f"❌ Failed to load model {normalized_name}: {e}", exc_info=True)
                raise
    
    def _normalize_model_name(self, model_name: str) -> str:
        """Normalise le nom du modèle"""
        if not model_name:
            return 'phi-3-mini'  # Modèle par défaut
        
        model_name = model_name.lower()
        
        # Mapper les anciens noms vers les nouveaux noms
        if 'mistral-7b' in model_name and 'instruct' not in model_name:
            return 'mistral-7b-instruct'
        
        # Si c'est un chemin, extraire le nom du modèle
        if model_name.startswith('./') or model_name.startswith('/'):
            parts = model_name.replace('\\', '/').split('/')
            # Chercher le nom du modèle dans le chemin
            for part in reversed(parts):
                if 'mistral-7b' in part and 'instruct' not in part:
                    return 'mistral-7b-instruct'
                elif any(m in part for m in ['phi-3', 'mistral', 'llama', 'gemma']):
                    # Normaliser mistral-7b vers mistral-7b-instruct
                    if 'mistral-7b' in part and 'instruct' not in part:
                        return 'mistral-7b-instruct'
                    return part
            return parts[-1] if parts else 'phi-3-mini'
        
        # Noms de modèles connus
        known_models = ['phi-3-mini', 'mistral-7b-instruct']
        for known in known_models:
            if known in model_name:
                return known
        
        # Si on trouve mistral-7b sans instruct, le mapper vers mistral-7b-instruct
        if 'mistral-7b' in model_name and 'instruct' not in model_name:
            return 'mistral-7b-instruct'
        
        return model_name
    
    def _evict_lru(self):
        """Supprime le modèle le moins récemment utilisé"""
        if not self._cache:
            return
        
        oldest_model = min(self._cache.keys(), key=lambda k: self._cache[k]['last_used'])
        logger.info(f"🗑️ Removing least recently used LLM model from cache: {oldest_model}")
        
        # Nettoyer les ressources du modèle si nécessaire
        try:
            service = self._cache[oldest_model]['service']
            if hasattr(service, 'cleanup'):
                service.cleanup()
        except Exception as e:
            logger.warning(f"⚠️ Error during model cleanup: {e}")
        
        del self._cache[oldest_model]
    
    def clear(self):
        """Vide complètement le cache"""
        with self._lock:
            for model_name in list(self._cache.keys()):
                try:
                    service = self._cache[model_name]['service']
                    if hasattr(service, 'cleanup'):
                        service.cleanup()
                except Exception as e:
                    logger.warning(f"⚠️ Error during cleanup of {model_name}: {e}")
            self._cache.clear()
            logger.info("🧹 LLM model cache cleared")

"""
Service d'enrichissement de transcription avec modèles LLM
Optimisé pour CPU avec quantisation GGUF via llama-cpp-python
Production-ready pour environnement CPU-only
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from infrastructure.models.model_manager import ModelManager

logger = logging.getLogger("vocalyx")

if not PSUTIL_AVAILABLE:
    logger.warning("⚠️ psutil not available, memory monitoring disabled")


class EnrichmentService:
    """Service d'enrichissement utilisant des modèles LLM via llama-cpp-python"""
    
    def __init__(self, config, model_name: str = None):
        """
        Initialise le service d'enrichissement.
        
        Args:
            config: Configuration du worker
            model_name: Nom du modèle LLM à utiliser (chemin vers fichier .gguf)
        """
        self.config = config
        self.model_name = model_name or getattr(config, 'llm_model', 'phi-3-mini')
        self.model = None
        self.tokenizer = None
        self.device = getattr(config, 'llm_device', 'cpu')
        self.compute_type = getattr(config, 'llm_compute_type', 'int8')
        self.max_tokens = getattr(config, 'llm_max_tokens', 256)
        self.temperature = getattr(config, 'llm_temperature', 0.7)
        self.top_p = getattr(config, 'llm_top_p', 0.9)
        self.top_k = getattr(config, 'llm_top_k', 40)
        
        # Paramètres CPU
        n_threads_config = getattr(config, 'llm_n_threads', None)
        if n_threads_config == 0:
            n_threads_config = None
        self.n_threads = n_threads_config  # Auto-détecté si None
        self.n_ctx = getattr(config, 'llm_n_ctx', 2048)  # Contexte maximum
        self.n_batch = getattr(config, 'llm_n_batch', 512)  # Batch size pour CPU
        
        # Détecter le nombre de threads CPU si non spécifié
        if self.n_threads is None:
            cpu_count = os.cpu_count() or 4
            # Utiliser tous les cores sauf 1 pour laisser de la marge
            self.n_threads = max(1, cpu_count - 1)
        
        # Gestionnaire de modèles
        # Par défaut, utiliser /app/shared/models/enrichment (Docker) comme transcription
        models_dir = getattr(config, 'llm_models_dir', '/app/shared/models/enrichment')
        self.model_manager = ModelManager(models_dir=models_dir)
        
        logger.info(
            f"🎯 EnrichmentService initialized | "
            f"Model: {self.model_name} | "
            f"Device: {self.device} | "
            f"Compute: {self.compute_type} | "
            f"Threads: {self.n_threads} | "
            f"Context: {self.n_ctx}"
        )
    
    def _load_model(self):
        """Charge le modèle LLM GGUF via llama-cpp-python (lazy loading)"""
        if self.model is not None:
            return
        
        try:
            logger.info(f"🚀 Loading LLM model: {self.model_name}...")
            
            # Obtenir le chemin du modèle via le gestionnaire
            model_path = self.model_manager.get_model_path(self.model_name)
            
            # Vérifier si le modèle existe
            if not model_path.exists():
                logger.warning(
                    f"⚠️ Model not found at: {model_path}\n"
                    f"   Searching in shared directories..."
                )
                
                # Le gestionnaire a déjà cherché dans shared/, mais on peut essayer de télécharger
                # seulement si c'est un modèle recommandé et qu'on a vraiment besoin
                if self.model_name in self.model_manager.RECOMMENDED_MODELS:
                    logger.info(f"📥 Model not found locally, attempting download...")
                    try:
                        model_path = self.model_manager.download_model(self.model_name)
                    except Exception as download_error:
                        logger.error(
                            f"❌ Failed to download model {self.model_name}: {download_error}\n"
                            f"   Please ensure the model file exists in:\n"
                            f"   - {self.model_manager.models_dir}\n"
                            f"   - /app/shared/models/enrichment/ (Docker)\n"
                            f"   - ./shared/models/enrichment/ (local)\n"
                            f"   Or provide the full path to the .gguf file."
                        )
                        raise
                else:
                    raise FileNotFoundError(
                        f"Model file not found: {model_path}\n"
                        f"Please ensure the model exists or provide the correct path."
                    )
            
            # Vérifier la santé du modèle
            if not self.model_manager.check_model_health(model_path):
                raise ValueError(f"Model health check failed: {model_path}")
            
            model_path_str = str(model_path.absolute())
            
            # Importer llama-cpp-python
            try:
                from llama_cpp import Llama
            except ImportError as e:
                import sys
                raise ImportError(
                    f"llama-cpp-python is not installed or not accessible.\n"
                    f"Error: {e}\n"
                    f"Python: {sys.executable}\n"
                    f"Python version: {sys.version}\n"
                    f"Install it with: pip3 install llama-cpp-python\n"
                    f"Or verify installation with: python3 -c 'import llama_cpp'"
                )
            
            # Déterminer le nombre de threads GPU (0 pour CPU-only)
            n_gpu_layers = 0  # CPU-only
            
            # Charger le modèle avec optimisations CPU
            logger.info(
                f"📦 Loading GGUF model | "
                f"Path: {model_path_str} | "
                f"Threads: {self.n_threads} | "
                f"Context: {self.n_ctx} | "
                f"Batch: {self.n_batch}"
            )
            
            self.model = Llama(
                model_path=model_path_str,
                n_ctx=self.n_ctx,  # Taille du contexte
                n_threads=self.n_threads,  # Nombre de threads CPU
                n_batch=self.n_batch,  # Taille du batch
                n_gpu_layers=n_gpu_layers,  # 0 = CPU-only
                verbose=False,  # Désactiver les logs verbeux de llama.cpp
                use_mmap=True,  # Memory mapping pour économiser la RAM
                use_mlock=False,  # Ne pas verrouiller en mémoire (permet swap si nécessaire)
            )
            
            # Vérifier la mémoire utilisée (si psutil disponible)
            if PSUTIL_AVAILABLE:
                mem_info = psutil.virtual_memory()
                logger.info(
                    f"✅ LLM model loaded successfully | "
                    f"Memory used: {mem_info.used / 1024**3:.2f} GB / {mem_info.total / 1024**3:.2f} GB "
                    f"({mem_info.percent:.1f}%)"
                )
            else:
                logger.info("✅ LLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load LLM model {self.model_name}: {e}", exc_info=True)
            raise
    
    def enrich_text(self, text: str, context: Optional[str] = None) -> str:
        """
        Enrichit un texte avec le modèle LLM.
        
        Args:
            text: Texte à enrichir
            context: Contexte optionnel (texte précédent)
            
        Returns:
            Texte enrichi
        """
        if not text or not text.strip():
            return text
        
        try:
            self._load_model()
            
            # Construire le prompt
            prompt = self._build_prompt(text, context)
            
            # Générer avec le modèle
            # Utiliser les paramètres optimisés pour CPU
            response = self.model(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repeat_penalty=1.1,  # Éviter la répétition
                stop=["</s>", "\n\n\n"],  # Tokens d'arrêt
                echo=False,  # Ne pas retourner le prompt
            )
            
            # Extraire le texte généré
            if isinstance(response, dict):
                enriched_text = response.get('choices', [{}])[0].get('text', '').strip()
            else:
                # Fallback si le format est différent
                enriched_text = str(response).strip()
            
            # Nettoyer le texte (supprimer les tokens d'arrêt qui auraient pu passer)
            enriched_text = enriched_text.replace('</s>', '').strip()
            
            # Si le modèle n'a rien généré ou a généré quelque chose de suspect,
            # retourner le texte original
            if not enriched_text or len(enriched_text) < len(text) * 0.5:
                logger.warning(
                    f"⚠️ Model generated suspicious output (too short or empty), "
                    f"returning original text"
                )
                return text
            
            logger.debug(
                f"✅ Text enriched | "
                f"Input: {len(text)} chars | "
                f"Output: {len(enriched_text)} chars"
            )
            return enriched_text
            
        except Exception as e:
            logger.error(f"❌ Error enriching text: {e}", exc_info=True)
            # En cas d'erreur, retourner le texte original
            return text
    
    def enrich_segments(self, segments: List[Dict], context: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Enrichit une liste de segments de transcription.
        
        Args:
            segments: Liste de segments à enrichir
            context: Segments précédents pour le contexte
            
        Returns:
            Liste de segments enrichis
        """
        if not segments:
            return []
        
        try:
            self._load_model()
            
            enriched_segments = []
            previous_text = None
            
            for i, segment in enumerate(segments):
                text = segment.get('text', '').strip()
                if not text:
                    enriched_segments.append(segment)
                    continue
                
                # Utiliser le texte précédent comme contexte
                context_text = previous_text if context is None else None
                
                # Enrichir le segment
                enriched_text = self.enrich_text(text, context_text)
                
                # Créer le segment enrichi
                enriched_segment = segment.copy()
                enriched_segment['enriched_text'] = enriched_text
                enriched_segment['original_text'] = text
                
                enriched_segments.append(enriched_segment)
                previous_text = text
            
            logger.info(f"✅ Enriched {len(enriched_segments)} segments")
            return enriched_segments
            
        except Exception as e:
            logger.error(f"❌ Error enriching segments: {e}", exc_info=True)
            # En cas d'erreur, retourner les segments originaux
            return segments
    
    def _build_prompt(self, text: str, context: Optional[str] = None) -> str:
        """
        Construit le prompt pour le modèle LLM.
        Adapté pour les modèles instruct (Phi-3, Mistral, Llama, etc.)
        
        Args:
            text: Texte à enrichir
            context: Contexte optionnel
            
        Returns:
            Prompt formaté selon le format du modèle
        """
        # Détecter le type de modèle depuis le nom
        model_lower = self.model_name.lower()
        
        # Format pour Phi-3
        if 'phi-3' in model_lower or 'phi3' in model_lower:
            system_prompt = "Tu es un assistant qui améliore et enrichit des transcriptions audio en français. Tu corriges les erreurs, améliores la ponctuation et la structure, tout en conservant le sens original."
            if context:
                user_prompt = f"Contexte précédent: {context}\n\nTexte à enrichir: {text}\n\nEnrichis ce texte:"
            else:
                user_prompt = f"Enrichis et améliore ce texte de transcription: {text}"
            
            prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>\n"
        
        # Format pour Mistral
        elif 'mistral' in model_lower:
            system_prompt = "Tu es un assistant qui améliore et enrichit des transcriptions audio en français."
            if context:
                user_prompt = f"Contexte: {context}\n\nTexte: {text}\n\nEnrichis ce texte:"
            else:
                user_prompt = f"Enrichis et améliore ce texte de transcription: {text}"
            
            prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        
        # Format pour Llama 3
        elif 'llama-3' in model_lower or 'llama3' in model_lower:
            system_prompt = "Tu es un assistant qui améliore et enrichit des transcriptions audio en français."
            if context:
                user_prompt = f"Contexte: {context}\n\nTexte: {text}\n\nEnrichis ce texte:"
            else:
                user_prompt = f"Enrichis et améliore ce texte de transcription: {text}"
            
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Format générique (ChatML)
        else:
            system_prompt = "Tu es un assistant qui améliore et enrichit des transcriptions audio en français. Tu corriges les erreurs, améliores la ponctuation et la structure, tout en conservant le sens original."
            if context:
                user_prompt = f"Contexte précédent: {context}\n\nTexte à enrichir: {text}\n\nEnrichis ce texte:"
            else:
                user_prompt = f"Enrichis et améliore ce texte de transcription: {text}"
            
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        return prompt
    
    def cleanup(self):
        """Nettoie les ressources du modèle"""
        if self.model is not None:
            try:
                # llama-cpp-python libère automatiquement les ressources
                # mais on peut forcer la libération
                del self.model
                self.model = None
                
                # Forcer le garbage collection
                import gc
                gc.collect()
                
                logger.info("🧹 EnrichmentService cleaned up")
            except Exception as e:
                logger.warning(f"⚠️ Error during cleanup: {e}")
                self.model = None

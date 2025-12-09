"""
Service d'enrichissement de transcription avec modèles LLM
Optimisé pour CPU avec quantisation GGUF via llama-cpp-python
Production-ready pour environnement CPU-only
"""

import logging
import os
import re
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
    
    def enrich_text(self, text: str, context: Optional[str] = None, custom_prompts: Optional[Dict] = None) -> str:
        """
        Enrichit un texte avec le modèle LLM.
        
        Args:
            text: Texte à enrichir
            context: Contexte optionnel (texte précédent)
            custom_prompts: Prompts personnalisés depuis l'API
            
        Returns:
            Texte enrichi
        """
        if not text or not text.strip():
            return text
        
        try:
            self._load_model()
            
            # Construire le prompt
            prompt = self._build_prompt(text, context, custom_prompts)
            
            # Déterminer les tokens d'arrêt selon le modèle
            model_lower = self.model_name.lower()
            if 'tinyllama' in model_lower or 'tiny-llama' in model_lower:
                stop_tokens = ["</s>", "<|user|>", "<|system|>", "<|assistant|>", "\n\n\n"]
            elif 'phi-3' in model_lower or 'phi3' in model_lower:
                stop_tokens = ["<|end|>", "<|user|>", "<|system|>", "<|assistant|>", "\n\n\n"]
            elif 'mistral' in model_lower:
                stop_tokens = ["</s>", "[INST]", "[/INST]", "\n\n\n"]
            elif 'llama-3' in model_lower or 'llama3' in model_lower:
                stop_tokens = ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>", "\n\n\n"]
            else:
                stop_tokens = ["<|im_end|>", "<|im_start|>", "</s>", "\n\n\n"]
            
            # Générer avec le modèle
            # Utiliser les paramètres optimisés pour CPU
            # Temperature plus basse pour être plus déterministe et moins créatif
            response = self.model(
                prompt,
                max_tokens=min(self.max_tokens, len(text.split()) * 2),  # Limiter selon la longueur du texte
                temperature=0.3,  # Plus bas pour être plus déterministe (correction plutôt que création)
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,  # Éviter la répétition
                stop=stop_tokens,  # Tokens d'arrêt adaptés au modèle
                echo=False,  # Ne pas retourner le prompt
            )
            
            # Extraire le texte généré (llama-cpp-python retourne un dict avec 'choices')
            if isinstance(response, dict):
                enriched_text = response.get('choices', [{}])[0].get('text', '').strip()
            elif hasattr(response, 'choices') and len(response.choices) > 0:
                enriched_text = response.choices[0].text.strip()
            else:
                # Fallback si le format est différent
                enriched_text = str(response).strip()
            
            # Nettoyer le texte (supprimer tous les tokens spéciaux qui auraient pu passer)
            tokens_to_remove = [
                '</s>', '<|end|>', '<|user|>', '<|system|>', '<|assistant|>',
                '<|im_start|>', '<|im_end|>', '<|eot_id|>', 
                '<|start_header_id|>', '<|end_header_id|>',
                '[INST]', '[/INST]', '<s>', '</s>'
            ]
            for token in tokens_to_remove:
                enriched_text = enriched_text.replace(token, '')
            
            # Nettoyer les espaces multiples et sauts de ligne excessifs
            import re
            enriched_text = re.sub(r'\n{3,}', '\n\n', enriched_text)  # Max 2 sauts de ligne
            enriched_text = re.sub(r' {2,}', ' ', enriched_text)  # Max 1 espace
            enriched_text = enriched_text.strip()
            
            # Si le modèle n'a rien généré ou a généré quelque chose de suspect,
            # retourner le texte original
            if not enriched_text:
                logger.warning(
                    f"⚠️ Model generated empty output, returning original text"
                )
                return text
            
            # Si le texte généré est beaucoup plus long que l'original (hallucination),
            # ou beaucoup plus court, retourner l'original
            if len(enriched_text) > len(text) * 3 or len(enriched_text) < len(text) * 0.3:
                logger.warning(
                    f"⚠️ Model generated suspicious output (length mismatch: "
                    f"input={len(text)} chars, output={len(enriched_text)} chars), "
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
    
    def enrich_segments(self, segments: List[Dict], context: Optional[List[Dict]] = None, custom_prompts: Optional[Dict] = None) -> List[Dict]:
        """
        Enrichit une liste de segments de transcription.
        
        Args:
            segments: Liste de segments à enrichir
            context: Segments précédents pour le contexte
            custom_prompts: Prompts personnalisés depuis l'API
            
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
                    # Segment vide, le garder tel quel
                    enriched_segments.append(segment)
                    continue
                
                # Utiliser le texte précédent comme contexte (limité à 200 caractères pour éviter les prompts trop longs)
                context_text = None
                if context is None and previous_text:
                    context_text = previous_text[-200:] if len(previous_text) > 200 else previous_text
                elif context:
                    # Construire le contexte depuis les segments précédents
                    context_segments = context[:i] if i < len(context) else context
                    context_text = " ".join(seg.get('text', '') for seg in context_segments[-3:])  # Derniers 3 segments
                    context_text = context_text[-200:] if len(context_text) > 200 else context_text
                
                # Enrichir le segment
                enriched_text = self.enrich_text(text, context_text, custom_prompts)
                
                # Créer le segment enrichi
                enriched_segment = segment.copy()
                enriched_segment['enriched_text'] = enriched_text
                enriched_segment['original_text'] = text  # Garder l'original pour comparaison
                
                enriched_segments.append(enriched_segment)
                previous_text = text
            
            logger.info(f"✅ Enriched {len(enriched_segments)} segments")
            return enriched_segments
            
        except Exception as e:
            logger.error(f"❌ Error enriching segments: {e}", exc_info=True)
            # En cas d'erreur, retourner les segments originaux
            return segments
    
    def _build_prompt(self, text: str, context: Optional[str] = None, custom_prompts: Optional[Dict] = None) -> str:
        """
        Construit le prompt pour le modèle LLM.
        Adapté pour les modèles instruct (TinyLlama, Phi-3, Mistral, Llama, etc.)
        
        Args:
            text: Texte à enrichir
            context: Contexte optionnel
            custom_prompts: Prompts personnalisés depuis l'API (dict avec title, summary, etc.)
            
        Returns:
            Prompt formaté selon le format du modèle
        """
        # Détecter le type de modèle depuis le nom
        model_lower = self.model_name.lower()
        
        # Instructions précises pour l'enrichissement
        # Le modèle doit CORRIGER et AMÉLIORER, pas inventer
        base_instructions = (
            "Tu es un assistant qui CORRIGE et AMÉLIORE des transcriptions audio en français. "
            "Ta tâche est de :\n"
            "1. Corriger les erreurs d'orthographe et de grammaire\n"
            "2. Améliorer la ponctuation (points, virgules, majuscules)\n"
            "3. Améliorer la structure (majuscules en début de phrase, paragraphes si nécessaire)\n"
            "4. CONSERVER EXACTEMENT le sens original - ne rien ajouter, ne rien inventer\n"
            "5. Retourner UNIQUEMENT le texte corrigé, sans explications ni commentaires"
        )
        
        # Si des prompts personnalisés sont fournis, les utiliser
        if custom_prompts:
            # Pour l'instant, on utilise le prompt de base mais on pourrait adapter selon le type
            # (title, summary, satisfaction, bullet_points)
            task_instruction = custom_prompts.get('summary', 
                "Corrige et améliore ce texte de transcription en conservant le sens original:")
        else:
            task_instruction = "Corrige et améliore ce texte de transcription en conservant le sens original:"
        
        # Construire le prompt utilisateur
        if context:
            user_prompt = f"{task_instruction}\n\nContexte précédent: {context}\n\nTexte à corriger:\n{text}"
        else:
            user_prompt = f"{task_instruction}\n\n{text}"
        
        # Format pour TinyLlama (utilise <|system|>, </s>, <|user|>, <|assistant|>)
        if 'tinyllama' in model_lower or 'tiny-llama' in model_lower:
            prompt = f"<|system|>\n{base_instructions}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>\n"
        
        # Format pour Phi-3
        elif 'phi-3' in model_lower or 'phi3' in model_lower:
            prompt = f"<|system|>\n{base_instructions}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>\n"
        
        # Format pour Mistral
        elif 'mistral' in model_lower:
            prompt = f"<s>[INST] {base_instructions}\n\n{user_prompt} [/INST]"
        
        # Format pour Llama 3
        elif 'llama-3' in model_lower or 'llama3' in model_lower:
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{base_instructions}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Format générique (ChatML) - pour Gemma et autres
        else:
            prompt = f"<|im_start|>system\n{base_instructions}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
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

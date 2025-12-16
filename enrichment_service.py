"""
EnrichmentService - Service pour l'enrichissement des transcriptions avec LLM (llama-cpp-python)
"""

import logging
import json
import re
import os
from typing import Dict, Optional, List
from pathlib import Path
from llama_cpp import Llama

from infrastructure.models.model_manager import ModelManager
from application.prompts import (
    DEFAULT_ENRICHMENT_PROMPTS,
    SEGMENT_CORRECTION_BASE_PROMPT,
    SEGMENT_CORRECTION_TASK_PROMPT,
)

logger = logging.getLogger("vocalyx.enrichment")


class EnrichmentService:
    """Service pour enrichir les transcriptions avec un LLM local (GGUF)"""
    
    def __init__(self, config=None, model_name: str = None, models_dir: Path = None, device: str = "cpu"):
        """
        Initialise le service d'enrichissement avec un modèle LLM local.
        
        Args:
            config: Configuration du worker (optionnel)
            model_name: Nom du modèle LLM (ex: "phi-3-mini", "mistral-7b-instruct") ou chemin complet
            models_dir: Répertoire contenant les modèles (défaut: depuis config ou /app/shared/models/enrichment)
            device: Device à utiliser ("cpu" uniquement pour GGUF)
        """
        self.config = config
        self.model_name = model_name or (getattr(config, 'llm_model', 'qwen2.5-7b-instruct') if config else 'qwen2.5-7b-instruct')
        self.device = device or (getattr(config, 'llm_device', 'cpu') if config else 'cpu')
        
        # Gestionnaire de modèles
        if models_dir:
            self.models_dir = Path(models_dir)
        elif config:
            self.models_dir = Path(getattr(config, 'llm_models_dir', '/app/shared/models/enrichment'))
        else:
            self.models_dir = Path("/app/shared/models/enrichment")
        
        self.model_manager = ModelManager(models_dir=str(self.models_dir))
        
        # Paramètres CPU depuis config
        n_threads_config = getattr(config, 'llm_n_threads', None) if config else None
        if n_threads_config == 0:
            n_threads_config = None
        self.n_threads = n_threads_config or max(1, (os.cpu_count() or 4) - 1)
        self.n_ctx = getattr(config, 'llm_n_ctx', 2048) if config else 2048
        self.n_batch = getattr(config, 'llm_n_batch', 512) if config else 512
        
        self.model_path = None
        self.llm = None
        # Cache de contexte / prompt cache (pour réutiliser les préfixes invariants)
        self.enable_prompt_cache = getattr(config, "llm_enable_prompt_cache", True) if config else True
        
        logger.info(
            f"🎯 EnrichmentService initialized | "
            f"Model: {self.model_name} | "
            f"Device: {self.device} | "
            f"Threads: {self.n_threads} | "
            f"Context: {self.n_ctx}"
        )
    
    def _load_model(self):
        """Charge le modèle LLM GGUF via llama-cpp-python (lazy loading)"""
        if self.llm is not None:
            return
        
        try:
            logger.info(f"🚀 Loading LLM model: {self.model_name}...")
            
            # Obtenir le chemin du modèle via le gestionnaire
            model_path = self.model_manager.get_model_path(self.model_name)
            
            if not model_path.exists():
                logger.warning(f"⚠️ Model not found at: {model_path}")
                
                # Lister les emplacements possibles pour aider au diagnostic
                possible_locations = [
                    "/app/shared/models/enrichment",
                    "/app/models/enrichment",
                    str(self.models_dir),
                ]
                logger.info(f"🔍 Searching for model '{self.model_name}' in possible locations:")
                for loc in possible_locations:
                    loc_path = Path(loc)
                    if loc_path.exists():
                        files = list(loc_path.glob("*.gguf"))
                        logger.info(f"   {loc}: {len(files)} GGUF file(s) found")
                        if files:
                            logger.info(f"      Files: {[f.name for f in files[:5]]}")
                    else:
                        logger.info(f"   {loc}: directory does not exist")
                
                # Essayer de télécharger si c'est un modèle recommandé
                if self.model_name in self.model_manager.RECOMMENDED_MODELS:
                    logger.info(f"📥 Attempting to download model {self.model_name}...")
                    try:
                        model_path = self.model_manager.download_model(self.model_name)
                    except Exception as download_error:
                        logger.error(f"❌ Failed to download model: {download_error}")
                        raise FileNotFoundError(
                            f"Model file not found: {model_path}\n"
                            f"Expected locations: {', '.join(possible_locations)}\n"
                            f"Model name: {self.model_name}"
                        )
                else:
                    raise FileNotFoundError(
                        f"Model file not found: {model_path}\n"
                        f"Expected locations: {', '.join(possible_locations)}\n"
                        f"Model name: {self.model_name}"
                    )
            
            # Vérifier la santé du modèle (existence, taille, etc.)
            if not self.model_manager.check_model_health(model_path):
                # Obtenir plus de détails sur le fichier
                if model_path.exists():
                    size_mb = model_path.stat().st_size / (1024 * 1024)
                    logger.error(
                        f"❌ Model health check failed: {model_path}\n"
                        f"   File exists: True\n"
                        f"   File size: {size_mb:.1f} MB\n"
                        f"   Expected size: ~{self.model_manager.RECOMMENDED_MODELS.get(self.model_name, {}).get('size_gb', '?')} GB"
                    )
                raise ValueError(f"Model health check failed: {model_path}")
            
            self.model_path = model_path
            model_path_str = str(model_path.absolute())
            
            # Vérifier que le fichier est lisible
            try:
                with open(model_path, 'rb') as f:
                    # Lire les premiers bytes pour vérifier le format GGUF
                    header = f.read(4)
                    if header != b'GGUF':
                        logger.warning(f"⚠️ File does not start with GGUF magic bytes: {header}")
            except Exception as e:
                logger.error(f"❌ Cannot read model file: {e}")
                raise ValueError(f"Cannot read model file: {model_path}")
            
            logger.info(
                f"📦 Loading GGUF model | "
                f"Path: {model_path_str} | "
                f"Size: {model_path.stat().st_size / (1024**3):.2f} GB | "
                f"Threads: {self.n_threads} | "
                f"Context: {self.n_ctx} | "
                f"Batch: {self.n_batch}"
            )
            
            # Vérifier la version de llama-cpp-python
            try:
                import llama_cpp
                llama_version = getattr(llama_cpp, '__version__', 'unknown')
                logger.info(f"📦 llama-cpp-python version: {llama_version}")
            except Exception:
                logger.warning("⚠️ Cannot determine llama-cpp-python version")
            
            # Charger le modèle GGUF avec llama-cpp-python
            try:
                llama_kwargs = {
                    "model_path": model_path_str,
                    "n_ctx": self.n_ctx,
                    "n_threads": self.n_threads,
                    "n_batch": self.n_batch,
                    "n_gpu_layers": 0,  # CPU only
                    "verbose": False,
                    "use_mmap": True,
                    "use_mlock": False,
                }

                # Activer le prompt cache si disponible dans cette version de llama-cpp-python.
                # Le cache permet de réutiliser les préfixes communs des prompts (règles, contexte),
                # ce qui réduit le temps de génération pour des prompts très similaires.
                if self.enable_prompt_cache:
                    # Certains environnements utilisent 'cache' / 'cache_type' (RAM).
                    llama_kwargs["cache"] = True
                    llama_kwargs["cache_type"] = "ram"

                self.llm = Llama(**llama_kwargs)
            except (ValueError, RuntimeError, OSError) as e:
                # Améliorer le message d'erreur pour les problèmes de chargement
                import llama_cpp
                llama_version = getattr(llama_cpp, '__version__', 'unknown')
                
                # Vérifier la version minimale recommandée
                version_ok = True
                try:
                    from packaging import version
                    if version.parse(llama_version) < version.parse("0.2.20"):
                        version_ok = False
                except:
                    pass
                
                error_msg = (
                    f"❌ Failed to load GGUF model with llama-cpp-python:\n"
                    f"   Path: {model_path_str}\n"
                    f"   File exists: {model_path.exists()}\n"
                    f"   File size: {model_path.stat().st_size / (1024**3):.2f} GB\n"
                    f"   llama-cpp-python version: {llama_version}\n"
                    f"   Error: {e}\n"
                    f"   Possible causes:\n"
                    f"   - llama-cpp-python version too old (need >= 0.2.20 for Qwen 2.5)\n"
                    f"   - File is corrupted or incomplete\n"
                    f"   - File is not a valid GGUF format\n"
                    f"   - Insufficient memory\n"
                )
                
                if not version_ok:
                    error_msg += (
                        f"\n   💡 Solution: Upgrade llama-cpp-python:\n"
                        f"      pip install --upgrade llama-cpp-python\n"
                        f"      or with CPU optimizations:\n"
                        f"      CMAKE_ARGS=\"-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS\" pip install --upgrade llama-cpp-python"
                    )
                
                logger.error(error_msg)
                raise ValueError(f"Failed to load model: {e}") from e
            
            logger.info(f"✅ Model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model {self.model_name}: {e}", exc_info=True)
            raise
    
    def _generate_text(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7, stop_tokens: List[str] = None) -> str:
        """
        Génère du texte avec le modèle LLM.
        
        Args:
            prompt: Prompt à envoyer au modèle
            max_tokens: Nombre maximum de tokens à générer
            temperature: Température pour la génération (0.0-1.0)
            stop_tokens: Liste de tokens d'arrêt (défaut: selon le modèle)
            
        Returns:
            str: Texte généré
        """
        try:
            self._load_model()
            
            # Déterminer les tokens d'arrêt selon le modèle
            if stop_tokens is None:
                model_lower = self.model_name.lower()
                if 'phi-3' in model_lower or 'phi3' in model_lower:
                    stop_tokens = ["<|end|>", "<|user|>", "<|system|>", "<|assistant|>", "\n\n\n"]
                elif 'mistral' in model_lower:
                    stop_tokens = ["</s>", "[INST]", "[/INST]", "\n\n\n"]
                elif 'llama-3' in model_lower or 'llama3' in model_lower:
                    stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "\n\n\n"]
                elif 'qwen' in model_lower:
                    stop_tokens = ["<|im_end|>", "<|endoftext|>", "\n\n\n"]
                elif 'gemma' in model_lower:
                    stop_tokens = ["<end_of_text>", "<eos>", "\n\n\n"]
                else:
                    stop_tokens = ["</s>", "<|im_end|>", "<|im_start|>", "\n\n\n"]
            
            # Formater le prompt selon le modèle
            model_lower = self.model_name.lower()
            if 'phi-3' in model_lower or 'phi3' in model_lower:
                formatted_prompt = f"<|system|>\n{prompt}<|end|>\n<|assistant|>\n"
            elif 'mistral' in model_lower:
                # Format Mistral : [INST] prompt [/INST]
                # Note: llama-cpp-python ajoute automatiquement <s> au début, ne pas l'inclure
                formatted_prompt = f"[INST] {prompt} [/INST]"
            elif 'llama-3' in model_lower or 'llama3' in model_lower:
                # Format Llama 3 : <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
                # Version simplifiée pour llama-cpp-python
                formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            elif 'qwen' in model_lower:
                # Format Qwen : <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
                formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            elif 'gemma' in model_lower:
                # Format Gemma : <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
                formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            else:
                formatted_prompt = prompt
            
            # Générer la réponse
            response = self.llm(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_tokens,
                echo=False
            )
            
            # Extraire le texte généré
            if isinstance(response, dict):
                generated_text = response.get('choices', [{}])[0].get('text', '').strip()
            elif hasattr(response, 'choices') and len(response.choices) > 0:
                generated_text = response.choices[0].text.strip()
            else:
                generated_text = str(response).strip()
            
            # Nettoyer les tokens spéciaux
            tokens_to_remove = [
                '</s>', '<|end|>', '<|user|>', '<|system|>', '<|assistant|>',
                '<|im_start|>', '<|im_end|>', '[INST]', '[/INST]', '<s>', '</s>',
                '<|eot_id|>', '<|end_of_text|>', '<end_of_text>', '<eos>'
            ]
            for token in tokens_to_remove:
                generated_text = generated_text.replace(token, '')
            
            # Nettoyer les espaces
            generated_text = re.sub(r'\n{2,}', '\n', generated_text)
            generated_text = re.sub(r' {2,}', ' ', generated_text)
            generated_text = generated_text.strip()
            
            return generated_text
        except Exception as e:
            logger.error(f"Error generating text: {e}", exc_info=True)
            raise
    
    def generate_title(self, transcription_text: str, custom_prompt: Optional[str] = None) -> str:
        """
        Génère un titre pour la transcription.
        
        Args:
            transcription_text: Texte de la transcription
            custom_prompt: Prompt personnalisé (optionnel)
            
        Returns:
            str: Titre généré
        """
        if not transcription_text or not transcription_text.strip():
            return ""
        
        prompt = custom_prompt or DEFAULT_ENRICHMENT_PROMPTS.get("title", "Appel client-agent. Génère un titre court (max 10 mots) en français:")
        # Limiter le texte à 500 caractères pour éviter les prompts trop longs
        text_sample = transcription_text[:500] if len(transcription_text) > 500 else transcription_text
        full_prompt = f"{prompt}\n\n{text_sample}"
        
        try:
            title = self._generate_text(full_prompt, max_tokens=30, temperature=0.7)
            if not title or not title.strip():
                return ""
            
            # Nettoyer le titre : supprimer les préfixes, guillemets, prendre la première phrase, max 10 mots
            title = title.strip()
            
            # Supprimer les préfixes indésirables (spécifique à Qwen et autres modèles)
            # Exemples: "Titre :", "Titre:", "Title:", "Titre: Rappel sinistre" -> "Rappel sinistre"
            # Pattern amélioré pour capturer avec espaces avant/après les deux-points
            title = re.sub(r'^Titre\s*:\s*', '', title, flags=re.IGNORECASE)
            title = re.sub(r'^Titre\s+', '', title, flags=re.IGNORECASE)  # "Titre " sans deux-points
            title = re.sub(r'^Title\s*:\s*', '', title, flags=re.IGNORECASE)
            title = title.strip()
            
            # Supprimer les guillemets au début et à la fin
            if title.startswith('"') and title.endswith('"'):
                title = title[1:-1]
            elif title.startswith("'") and title.endswith("'"):
                title = title[1:-1]
            
            # Prendre seulement la première phrase (jusqu'au premier point, exclamation, ou question)
            first_sentence_match = re.search(r'^([^.!?]+[.!?]?)', title)
            if first_sentence_match:
                title = first_sentence_match.group(1).strip()
            
            # Supprimer les explications communes du modèle
            title = re.sub(r'\s*(This title|Ce titre|Le titre|Title:).*$', '', title, flags=re.IGNORECASE)
            title = re.sub(r'\s*(captures|reflète|décrit|represents).*$', '', title, flags=re.IGNORECASE)
            
            # Prendre maximum 10 mots
            words = title.split()[:10]
            result = " ".join(words).strip()
            
            # Nettoyer les ponctuations finales si ce n'est pas une phrase complète
            if result and result[-1] in ['.', '!', '?'] and len(words) < 10:
                result = result[:-1].strip()
            
            return result if result else ""
        except Exception as e:
            logger.error(f"Error generating title: {e}", exc_info=True)
            return ""
    
    def generate_summary(self, transcription_text: str, custom_prompt: Optional[str] = None) -> str:
        """
        Génère un résumé de moins de 100 mots.
        
        Args:
            transcription_text: Texte de la transcription
            custom_prompt: Prompt personnalisé (optionnel)
            
        Returns:
            str: Résumé généré
        """
        if not transcription_text or not transcription_text.strip():
            return ""
        
        prompt = custom_prompt or DEFAULT_ENRICHMENT_PROMPTS.get("summary", "Appel client-agent. Génère un résumé concis (max 50 mots) en français:")
        full_prompt = f"{prompt}\n\n{transcription_text}"
        
        try:
            summary = self._generate_text(full_prompt, max_tokens=150, temperature=0.7)
            if not summary or not summary.strip():
                return ""
            
            # Supprimer les préfixes indésirables (spécifique à Qwen et autres modèles)
            # Exemples: "Résumé (50 mots) :", "Résumé (50 mots):", "Summary:", "Résumé:", etc.
            # Pattern amélioré pour capturer avec espaces avant/après les deux-points
            summary = re.sub(r'^Résumé\s*\([^)]*\)\s*:\s*', '', summary, flags=re.IGNORECASE)  # "Résumé (50 mots) :"
            summary = re.sub(r'^Résumé\s*\([^)]*\)\s+', '', summary, flags=re.IGNORECASE)  # "Résumé (50 mots) " sans deux-points
            summary = re.sub(r'^Résumé\s*:\s*', '', summary, flags=re.IGNORECASE)  # "Résumé :"
            summary = re.sub(r'^Résumé\s+', '', summary, flags=re.IGNORECASE)  # "Résumé " sans deux-points
            summary = re.sub(r'^Summary\s*\([^)]*\)\s*:\s*', '', summary, flags=re.IGNORECASE)
            summary = re.sub(r'^Summary\s*:\s*', '', summary, flags=re.IGNORECASE)
            summary = summary.strip()
            
            # Limiter à 100 mots
            words = summary.split()[:100]
            result = " ".join(words).strip()
            return result if result else ""
        except Exception as e:
            logger.error(f"Error generating summary: {e}", exc_info=True)
            return ""
    
    def generate_satisfaction_score(self, transcription_text: str, custom_prompt: Optional[str] = None) -> Dict:
        """
        Génère un score de satisfaction de 1 à 10.
        
        Args:
            transcription_text: Texte de la transcription
            custom_prompt: Prompt personnalisé (optionnel)
            
        Returns:
            dict: {"score": int}
        """
        if not transcription_text or not transcription_text.strip():
            return {"score": 5}
        
        prompt = custom_prompt or DEFAULT_ENRICHMENT_PROMPTS.get("satisfaction", "Appel client-agent. Score satisfaction du point de vue client de 1-10. JSON: {\"score\": nombre}.")
        full_prompt = f"{prompt}\n\n{transcription_text}"
        
        try:
            response = self._generate_text(full_prompt, max_tokens=100, temperature=0.5)
            
            if not response or not response.strip():
                return {"score": 5}
            
            # Essayer d'extraire le JSON de la réponse (chercher le JSON le plus externe)
            try:
                # Chercher tous les JSON imbriqués et prendre le plus externe
                start = response.find('{')
                if start >= 0:
                    # Trouver la fin du JSON en comptant les accolades
                    brace_count = 0
                    end = start
                    for i in range(start, len(response)):
                        if response[i] == '{':
                            brace_count += 1
                        elif response[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end = i + 1
                                break
                    
                    if end > start:
                        json_str = response[start:end]
                        data = json.loads(json_str)
                        score = int(data.get("score", 5))
                        
                        return { "score": max(1, min(10, score)) }
            except Exception as json_error:
                logger.debug(f"Failed to parse JSON from satisfaction response: {json_error}, response: {response[:100]}")
            
            # Fallback: extraire un score simple
            score_match = re.search(r'\b([1-9]|10)\b', response)
            score = int(score_match.group(1)) if score_match else 5
            # Extraire une justification simple (première phrase)
            justification_match = re.search(r'[Jj]ustification[:\s]+(.+?)(?:\.|$|\n)', response)
            if not justification_match:
                justification_match = re.search(r'[Pp]arce que[:\s]+(.+?)(?:\.|$|\n)', response)
            justification = justification_match.group(1).strip()[:200] if justification_match else response.strip()[:200]
            
            return {
                "score": max(1, min(10, score)),
                "justification": justification
            }
        except Exception as e:
            logger.error(f"Error generating satisfaction score: {e}", exc_info=True)
            return {"score": 5, "justification": f"Erreur: {str(e)[:100]}"}
    
    def generate_bullet_points(self, transcription_text: str, custom_prompt: Optional[str] = None) -> list:
        """
        Génère des bullet points pour la transcription.
        
        Args:
            transcription_text: Texte de la transcription
            custom_prompt: Prompt personnalisé (optionnel)
            
        Returns:
            list: Liste de bullet points
        """
        if not transcription_text or not transcription_text.strip():
            return []
        
        prompt = custom_prompt or DEFAULT_ENRICHMENT_PROMPTS.get("bullet_points", "Appel client-agent. Points clés en puces. JSON: {\"points\": [...]}. Réponds en français.")
        full_prompt = f"{prompt}\n\n{transcription_text}"
        
        try:
            response = self._generate_text(full_prompt, max_tokens=200, temperature=0.7)
            
            if not response or not response.strip():
                return []
            
            # Essayer d'extraire le JSON de la réponse
            try:
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    data = json.loads(json_str)
                    points = data.get("points", [])
                    # Filtrer les points vides
                    points = [p.strip() for p in points if p and p.strip()]
                    return points[:4] if points else []
            except Exception as json_error:
                logger.debug(f"Failed to parse JSON from bullet points response: {json_error}")
            
            # Fallback: extraire les points avec regex
            points = re.findall(r'[-•*]\s*(.+?)(?=\n|$)', response)
            if not points:
                # Essayer d'extraire des lignes numérotées
                points = re.findall(r'\d+[\.\)]\s*(.+?)(?=\n|$)', response)
            # Filtrer et limiter
            points = [p.strip() for p in points if p and p.strip()]
            return points[:4] if points else []
        except Exception as e:
            logger.error(f"Error generating bullet points: {e}", exc_info=True)
            return []
    
    def enrich_transcription(
        self,
        transcription_text: str,
        prompts: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Enrichit une transcription complète en générant titre, résumé, score et bullet points.
        
        Args:
            transcription_text: Texte de la transcription
            prompts: Dictionnaire avec les prompts personnalisés (optionnel)
            
        Returns:
            dict: Données d'enrichissement avec temps individuels
        """
        import time
        
        logger.info("Starting enrichment...")
        enrichment_start_time = time.time()
        
        # D'abord, tenter un enrichissement "full" en un seul appel LLM
        try:
            logger.info("Generating full metadata (single LLM call)...")
            full_start = time.time()
            metadata = self.generate_full_metadata(transcription_text, prompts or {})
            full_time = round(time.time() - full_start, 2)

            if metadata:
                total_enrichment_time = round(time.time() - enrichment_start_time, 2)
                logger.info(
                    f"✅ Enrichment (single call) completed in {total_enrichment_time}s "
                    f"(llm_call_time: {full_time}s)"
                )

                # Normaliser la structure de retour
                title = metadata.get("title") or ""
                summary = metadata.get("summary") or ""
                satisfaction_score = metadata.get("satisfaction", {}).get("score", 5)
                bullet_points = metadata.get("bullet_points") or []

                enrichment_data = {
                    "title": title,
                    "summary": summary,
                    "satisfaction_score": satisfaction_score,
                    "bullet_points": bullet_points[:4],
                    "timing": {
                        "llm_call_time": full_time,
                        "total_time": total_enrichment_time,
                    },
                }
                return enrichment_data
        except Exception as e:
            logger.error(
                f"❌ Error during single-call metadata enrichment, falling back to multi-call mode: {e}",
                exc_info=True,
            )

        # Fallback : comportement historique (4 appels LLM séparés)
        # Utiliser les prompts personnalisés ou les defaults
        title_prompt = prompts.get("title") if prompts and isinstance(prompts, dict) else None
        summary_prompt = prompts.get("summary") if prompts and isinstance(prompts, dict) else None
        satisfaction_prompt = prompts.get("satisfaction") if prompts and isinstance(prompts, dict) else None
        bullet_points_prompt = prompts.get("bullet_points") if prompts and isinstance(prompts, dict) else None
        
        # Générer tous les éléments avec mesure du temps
        logger.info("Generating title (fallback)...")
        title_start = time.time()
        title = self.generate_title(transcription_text, title_prompt)
        title_time = round(time.time() - title_start, 2)
        
        logger.info("Generating summary (fallback)...")
        summary_start = time.time()
        summary = self.generate_summary(transcription_text, summary_prompt)
        summary_time = round(time.time() - summary_start, 2)
        
        logger.info("Generating satisfaction score (fallback)...")
        satisfaction_start = time.time()
        satisfaction = self.generate_satisfaction_score(transcription_text, satisfaction_prompt)
        satisfaction_time = round(time.time() - satisfaction_start, 2)
        
        logger.info("Generating bullet points (fallback)...")
        bullet_points_start = time.time()
        bullet_points = self.generate_bullet_points(transcription_text, bullet_points_prompt)
        bullet_points_time = round(time.time() - bullet_points_start, 2)
        
        total_enrichment_time = round(time.time() - enrichment_start_time, 2)
        
        enrichment_data = {
            "title": title,
            "summary": summary,
            "satisfaction_score": satisfaction["score"],
            "bullet_points": bullet_points[:4],  # Limiter à 4 points maximum
            "timing": {
                "title_time": title_time,
                "summary_time": summary_time,
                "satisfaction_time": satisfaction_time,
                "bullet_points_time": bullet_points_time,
                "total_time": total_enrichment_time
            }
        }
        
        logger.info(f"✅ Enrichment completed in {total_enrichment_time}s (title: {title_time}s, summary: {summary_time}s, score: {satisfaction_time}s, bullets: {bullet_points_time}s)")
        return enrichment_data

    def enrich_segments(self, segments: List[Dict], context: Optional[List[Dict]] = None, custom_prompts: Optional[Dict] = None) -> List[Dict]:
        """
        Enrichit une liste de segments de transcription (correction du texte).
        
        Args:
            segments: Liste de segments à enrichir
            context: Segments précédents pour le contexte (non utilisé pour la correction)
            custom_prompts: Prompts personnalisés (non utilisé pour la correction)
            
        Returns:
            Liste de segments enrichis avec 'enriched_text'
        """
        if not segments:
            return []
        
        # Appel du mode batch par défaut pour optimiser les performances.
        # Cette méthode est conservée pour la compatibilité avec l'existant.
        try:
            return self.enrich_segments_batch(segments)
        except Exception as e:
            logger.error(f"❌ Error enriching segments in batch mode, falling back to per-segment mode: {e}", exc_info=True)
        
        # Fallback : ancien comportement (traitement segment par segment) en cas de problème avec le batch.
        try:
            self._load_model()
            
            enriched_segments = []
            
            for segment in segments:
                text = segment.get('text', '').strip()
                if not text:
                    enriched_segments.append(segment)
                    continue
                
                prompt = (
                    f"{SEGMENT_CORRECTION_BASE_PROMPT}\n\n"
                    f"{SEGMENT_CORRECTION_TASK_PROMPT}\n\n"
                    f"Texte:\n{text}"
                )
                
                estimated_tokens = len(text.split())
                max_tokens_for_text = min(256, max(50, int(estimated_tokens * 1.2)))
                
                enriched_text = self._generate_text(
                    prompt,
                    max_tokens=max_tokens_for_text,
                    temperature=0.05,
                    stop_tokens=["</s>", "<|end|>", "<|user|>", "<|system|>", "<|assistant|>", "\n\n\n"]
                )
                
                if not enriched_text:
                    enriched_text = text
                else:
                    length_ratio = len(enriched_text) / len(text) if len(text) > 0 else 1.0
                    if length_ratio > 1.5 or length_ratio < 0.5:
                        logger.warning(
                            "⚠️ Model generated suspicious output (length mismatch: "
                            f"input={len(text)} chars, output={len(enriched_text)} chars), "
                            "returning original text"
                        )
                        enriched_text = text
                
                enriched_segment = segment.copy()
                enriched_segment["enriched_text"] = enriched_text
                enriched_segment["original_text"] = text
                enriched_segments.append(enriched_segment)
            
            logger.info(f"✅ Enriched {len(enriched_segments)} segments (fallback per-segment mode)")
            return enriched_segments
        except Exception as e:
            logger.error(f"❌ Error enriching segments in fallback per-segment mode: {e}", exc_info=True)
            return segments

    def enrich_segments_batch(self, segments: List[Dict], batch_size: int = 5) -> List[Dict]:
        """
        Enrichit une liste de segments de transcription en mode batch.

        Objectif : réduire le nombre d'appels LLM en traitant plusieurs segments
        dans un même prompt, avec un format de réponse JSON structuré.

        Args:
            segments: Liste de segments à enrichir (chacun contient au moins 'text')
            batch_size: Nombre de segments par batch dans un même appel LLM

        Returns:
            Liste de segments enrichis, avec les clés 'enriched_text' et 'original_text'
        """
        if not segments:
            return []

        self._load_model()

        enriched_segments: List[Dict] = []

        # Nous allons traiter les segments par paquets pour optimiser le coût du prompt.
        for batch_start in range(0, len(segments), batch_size):
            batch = segments[batch_start : batch_start + batch_size]

            # Préparer la liste des entrées à corriger (en gardant les index pour faire le mapping)
            batch_items = []
            for idx_in_batch, segment in enumerate(batch):
                text = segment.get("text", "").strip()
                if not text:
                    # Segment vide : on le recopiera tel quel plus bas
                    continue
                batch_items.append(
                    {
                        "batch_index": idx_in_batch,
                        "text": text,
                    }
                )

            # Si tous les segments du batch sont vides, on passe au suivant
            if not batch_items:
                for segment in batch:
                    enriched_segments.append(segment)
                continue

            # Construire un prompt compact pour tout le batch
            # On demande une réponse JSON structurée pour faciliter le parsing.
            instructions = (
                f"{SEGMENT_CORRECTION_BASE_PROMPT}\n\n"
                "Tu vas corriger plusieurs segments NUMÉROTÉS.\n"
                "Pour chaque segment, retourne UNIQUEMENT le texte corrigé dans un objet JSON.\n"
                "Format de réponse STRICT :\n"
                '{ "segments": [ { "id": <index_segment>, "text": "texte corrigé" }, ... ] }\n'
                "Ne retourne rien d'autre que ce JSON."
            )

            segments_block_lines = []
            for item in batch_items:
                segments_block_lines.append(f"[{item['batch_index']}] {item['text']}")
            segments_block = "\n".join(segments_block_lines)

            prompt = (
                f"{instructions}\n\n"
                f"{SEGMENT_CORRECTION_TASK_PROMPT}\n\n"
                f"SEGMENTS À CORRIGER :\n{segments_block}\n"
            )

            # Estimation simple de la taille max de sortie (somme des tokens des textes)
            total_words = sum(len(item["text"].split()) for item in batch_items)
            max_tokens_for_batch = min(512, max(100, int(total_words * 1.2)))

            try:
                response = self._generate_text(
                    prompt,
                    max_tokens=max_tokens_for_batch,
                    temperature=0.05,
                    stop_tokens=["</s>", "<|end|>", "<|user|>", "<|system|>", "<|assistant|>", "\n\n\n"],
                )
            except Exception as e:
                logger.error(f"❌ Error generating batch enrichment: {e}", exc_info=True)
                # En cas d'erreur, on retombe sur le mode unitaire pour ce batch
                for segment in batch:
                    text = segment.get("text", "").strip()
                    if not text:
                        enriched_segments.append(segment)
                        continue

                    single_prompt = (
                        f"{SEGMENT_CORRECTION_BASE_PROMPT}\n\n"
                        f"{SEGMENT_CORRECTION_TASK_PROMPT}\n\n"
                        f"Texte:\n{text}"
                    )
                    estimated_tokens = len(text.split())
                    max_tokens_for_text = min(256, max(50, int(estimated_tokens * 1.2)))

                    enriched_text = self._generate_text(
                        single_prompt,
                        max_tokens=max_tokens_for_text,
                        temperature=0.05,
                        stop_tokens=["</s>", "<|end|>", "<|user|>", "<|system|>", "<|assistant|>", "\n\n\n"],
                    )
                    if not enriched_text:
                        enriched_text = text
                    else:
                        length_ratio = len(enriched_text) / len(text) if len(text) > 0 else 1.0
                        if length_ratio > 1.5 or length_ratio < 0.5:
                            logger.warning(
                                "⚠️ Model generated suspicious output in batch fallback (length mismatch: "
                                f"input={len(text)} chars, output={len(enriched_text)} chars), "
                                "returning original text"
                            )
                            enriched_text = text

                    enriched_segment = segment.copy()
                    enriched_segment["enriched_text"] = enriched_text
                    enriched_segment["original_text"] = text
                    enriched_segments.append(enriched_segment)

                continue

            # Tenter de parser la réponse JSON
            corrected_by_id: Dict[int, str] = {}
            if response:
                try:
                    start = response.find("{")
                    end = response.rfind("}") + 1
                    if start >= 0 and end > start:
                        json_str = response[start:end]
                        data = json.loads(json_str)
                        for item in data.get("segments", []):
                            try:
                                seg_id = int(item.get("id"))
                                seg_text = (item.get("text") or "").strip()
                                if seg_text:
                                    corrected_by_id[seg_id] = seg_text
                            except Exception:
                                continue
                except Exception as json_error:
                    logger.warning(f"⚠️ Failed to parse batch enrichment JSON: {json_error} | response snippet: {response[:200]}")

            # Appliquer les corrections (en gardant les garde-fous sur la longueur)
            for idx_in_batch, segment in enumerate(batch):
                text = segment.get("text", "").strip()
                if not text:
                    enriched_segments.append(segment)
                    continue

                enriched_text = corrected_by_id.get(idx_in_batch, text)

                if enriched_text != text:
                    length_ratio = len(enriched_text) / len(text) if len(text) > 0 else 1.0
                    if length_ratio > 1.5 or length_ratio < 0.5:
                        logger.warning(
                            "⚠️ Model generated suspicious output in batch mode (length mismatch: "
                            f"input={len(text)} chars, output={len(enriched_text)} chars), "
                            "returning original text"
                        )
                        enriched_text = text

                enriched_segment = segment.copy()
                enriched_segment["enriched_text"] = enriched_text
                enriched_segment["original_text"] = text
                enriched_segments.append(enriched_segment)

        logger.info(f"✅ Enriched {len(enriched_segments)} segments in batch mode")
        return enriched_segments
    
    def generate_metadata(self, text: str, task_type: str, prompts: Dict[str, str], max_tokens: int = 100) -> str:
        """
        Génère des métadonnées (titre, résumé, score, bullet points) à partir du texte.
        
        Args:
            text: Texte de la transcription
            task_type: Type de métadonnée ("title", "summary", "satisfaction", "bullet_points")
            prompts: Dict avec les prompts (par défaut + personnalisés)
            max_tokens: Nombre maximum de tokens à générer
            
        Returns:
            Texte généré (métadonnée)
        """
        if not text or not text.strip():
            return ""
    
    def generate_full_metadata(self, transcription_text: str, prompts: Dict[str, str]) -> Dict:
        """
        Génère toutes les métadonnées en un seul appel LLM.

        Structure de réponse attendue (JSON strict) :
        {
          "title": "Titre court",
          "summary": "Résumé concis",
          "satisfaction": { "score": 7 },
          "bullet_points": ["point 1", "point 2", ...]
        }

        Args:
            transcription_text: Texte complet de la transcription
            prompts: Prompts personnalisés optionnels pour chaque champ

        Returns:
            dict: Métadonnées structurées (peut être vide en cas d'erreur)
        """
        if not transcription_text or not transcription_text.strip():
            return {}

        self._load_model()

        # Construire des prompts courts (ou utiliser les prompts custom)
        title_prompt = prompts.get("title") if prompts else DEFAULT_ENRICHMENT_PROMPTS.get("title", "")
        summary_prompt = prompts.get("summary") if prompts else DEFAULT_ENRICHMENT_PROMPTS.get("summary", "")
        satisfaction_prompt = prompts.get("satisfaction") if prompts else DEFAULT_ENRICHMENT_PROMPTS.get("satisfaction", "")
        bullet_points_prompt = prompts.get("bullet_points") if prompts else DEFAULT_ENRICHMENT_PROMPTS.get("bullet_points", "")

        system_instructions = (
            "Tu es un assistant qui analyse des appels entre un client et un agent en français.\n"
            "À partir de la transcription fournie, tu dois retourner un objet JSON STRICT avec le format suivant :\n"
            '{\n'
            '  "title": "titre très court (max 10 mots)",\n'
            '  "summary": "résumé concis (max 50 mots)",\n'
            '  "satisfaction": { "score": nombre_entre_1_et_10 },\n'
            '  "bullet_points": ["point 1", "point 2", ...]\n'
            '}\n'
            "Ne retourne STRICTEMENT RIEN d'autre que ce JSON.\n"
        )

        # On rappelle brièvement la consigne spécifique de chaque champ pour guider le modèle
        meta_instructions = (
            f"Titre : {title_prompt}\n"
            f"Résumé : {summary_prompt}\n"
            f"Satisfaction : {satisfaction_prompt}\n"
            f"Points clés : {bullet_points_prompt}\n"
        )

        full_prompt = (
            f"{system_instructions}\n"
            f"{meta_instructions}\n"
            "TRANSCRIPTION COMPLÈTE :\n"
            f"{transcription_text}\n"
        )

        # Estimer un max_tokens raisonnable pour générer tous les champs.
        # Comparaison avec les méthodes individuelles :
        # - title: 30 tokens
        # - summary: 150 tokens
        # - satisfaction: 100 tokens
        # - bullet_points: 200 tokens
        # Total: ~480 tokens. On prend 600 pour avoir de la marge et éviter les troncatures.
        max_tokens = 600

        response = self._generate_text(
            full_prompt,
            max_tokens=max_tokens,
            temperature=0.3,  # Plus bas pour un comportement déterministe et structuré
        )

        if not response:
            return {}

        # Essayer d'extraire et de parser le JSON
        start = response.find("{")
        if start < 0:
            logger.warning(f"⚠️ Full metadata response does not contain JSON: {response[:200]}")
            return {}

        # Trouver la fin du JSON en comptant les accolades (plus robuste que rfind)
        # Cela permet de gérer les cas où il y a des objets imbriqués
        brace_count = 0
        end = start
        for i in range(start, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        
        # Si on n'a pas trouvé de fermeture complète, le JSON est tronqué
        # On utilisera directement les regex de fallback
        if brace_count != 0:
            logger.warning(
                f"⚠️ JSON appears to be truncated (unclosed braces: {brace_count}). "
                f"Response length: {len(response)} chars. Will use regex fallback extraction."
            )
            # Passer directement au fallback regex (on ne fait pas de json.loads)
            json_str = None
        else:
            json_str = response[start:end]

        data: Dict = {}
        try:
            if json_str is None:
                # JSON tronqué, on force l'exception pour utiliser le fallback
                raise ValueError("JSON truncated - using regex fallback")
            data = json.loads(json_str)
        except Exception as e:
            # Certains modèles (comme Qwen) peuvent générer un JSON presque valide mais
            # avec une virgule manquante ou une ligne tronquée. On tente alors une
            # extraction plus robuste champ par champ via des regex.
            json_str_info = f"Extracted JSON string length: {len(json_str)} chars | Extracted JSON string: {json_str}" if json_str else "JSON was truncated, using regex fallback"
            logger.error(
                f"❌ Error parsing full metadata JSON: {e} | "
                f"Full response length: {len(response)} chars | "
                f"Full response: {response} | "
                f"{json_str_info}",
                exc_info=True,
            )

            # Extraction du titre
            title_match = re.search(r'"title"\s*:\s*"([^"]*)"', response)
            title = title_match.group(1).strip() if title_match else ""

            # Extraction du résumé
            summary_match = re.search(r'"summary"\s*:\s*"([^"]*)"', response)
            summary = summary_match.group(1).strip() if summary_match else ""

            # Extraction du score de satisfaction
            score_int = 5
            score_match = re.search(r'"satisfaction"\s*:\s*\{\s*"score"\s*:\s*([0-9]+)', response)
            if score_match:
                try:
                    score_int = int(score_match.group(1))
                    score_int = max(1, min(10, score_int))
                except Exception:
                    score_int = 5

            # Extraction des bullet points (liste simple de chaînes)
            # Gère aussi les cas où la liste est tronquée (pas de ] final)
            bullet_points: List[str] = []
            # Chercher "bullet_points": [ suivi de contenu (même si pas fermé)
            bullets_start_match = re.search(r'"bullet_points"\s*:\s*\[', response)
            if bullets_start_match:
                start_pos = bullets_start_match.end()
                # Chercher la fin de la liste (] ou fin de la réponse si tronquée)
                end_pos = response.find(']', start_pos)
                if end_pos < 0:
                    # Liste tronquée, prendre jusqu'à la fin de la réponse
                    end_pos = len(response)
                
                raw_block = response[start_pos:end_pos]
                # Capturer toutes les chaînes entre guillemets dans le bloc
                for m in re.finditer(r'"([^"]+)"', raw_block):
                    pt = m.group(1).strip()
                    if pt:
                        bullet_points.append(pt)
            
            # Limiter à 4 bullet points (cohérent avec generate_bullet_points)
            bullet_points = bullet_points[:4]

            # Même si le JSON complet est invalide, on renvoie ce qu'on a pu extraire
            return {
                "title": title,
                "summary": summary,
                "satisfaction": {"score": score_int},
                "bullet_points": bullet_points,
            }

        # Normalisation / garde-fous (chemin JSON valide)
        title = str(data.get("title", "") or "").strip()
        summary = str(data.get("summary", "") or "").strip()
        satisfaction = data.get("satisfaction") or {}
        if isinstance(satisfaction, dict):
            score = satisfaction.get("score", 5)
        else:
            # Si le modèle renvoie un score brut (ex: 7)
            try:
                score = int(satisfaction)
            except Exception:
                score = 5
        try:
            score_int = int(score)
        except Exception:
            score_int = 5
        score_int = max(1, min(10, score_int))

        bullet_points = data.get("bullet_points") or []
        if not isinstance(bullet_points, list):
            bullet_points = []
        # Nettoyer les points et limiter à 4 (cohérent avec generate_bullet_points)
        bullet_points = [str(p).strip() for p in bullet_points if p and str(p).strip()]
        bullet_points = bullet_points[:4]

        return {
            "title": title,
            "summary": summary,
            "satisfaction": {"score": score_int},
            "bullet_points": bullet_points,
        }
        
        try:
            self._load_model()
            
            # Obtenir le prompt pour cette tâche
            prompt_text = prompts.get(task_type, DEFAULT_ENRICHMENT_PROMPTS.get(task_type, ""))
            
            # Construire le prompt complet
            if task_type == "title":
                full_prompt = f"{prompt_text}\n\n{text[:500]}"
                result = self.generate_title(text, prompt_text)
                return result
            elif task_type == "summary":
                full_prompt = f"{prompt_text}\n\n{text}"
                result = self.generate_summary(text, prompt_text)
                return result
            elif task_type == "satisfaction":
                full_prompt = f"{prompt_text}\n\n{text}"
                result = self.generate_satisfaction_score(text, prompt_text)
                # Retourner en format JSON string
                return json.dumps(result, ensure_ascii=False)
            elif task_type == "bullet_points":
                full_prompt = f"{prompt_text}\n\n{text}"
                result = self.generate_bullet_points(text, prompt_text)
                # Retourner en format JSON string
                return json.dumps({"points": result}, ensure_ascii=False)
            else:
                # Fallback: utiliser _generate_text directement
                full_prompt = f"{prompt_text}\n\n{text}"
                return self._generate_text(full_prompt, max_tokens=max_tokens, temperature=0.7)
            
        except Exception as e:
            logger.error(f"❌ Error generating metadata '{task_type}': {e}", exc_info=True)
            return ""
    
    def cleanup(self):
        """Nettoie les ressources du modèle"""
        if self.llm is not None:
            try:
                del self.llm
                self.llm = None
                import gc
                gc.collect()
                logger.info("🧹 EnrichmentService cleaned up")
            except Exception as e:
                logger.warning(f"⚠️ Error during cleanup: {e}")
                self.llm = None
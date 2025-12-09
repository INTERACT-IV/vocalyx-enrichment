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

logger = logging.getLogger("vocalyx.enrichment")

# Prompts par défaut pour l'enrichissement
DEFAULT_ENRICHMENT_PROMPTS = {
    "title": "Génère un titre court et accrocheur (maximum 10 mots) pour cette transcription.",
    "summary": "Génère un résumé concis de moins de 100 mots pour cette transcription.",
    "satisfaction": "Analyse cette transcription et attribue un score de satisfaction client de 1 à 10. Justifie brièvement ton score. Format JSON: {\"score\": nombre, \"justification\": \"texte\"}",
    "bullet_points": "Extrais les points clés de cette transcription sous forme de puces. Format JSON: {\"points\": [\"point 1\", \"point 2\", ...]}"
}


class EnrichmentService:
    """Service pour enrichir les transcriptions avec un LLM local (GGUF)"""
    
    def __init__(self, config=None, model_name: str = None, models_dir: Path = None, device: str = "cpu"):
        """
        Initialise le service d'enrichissement avec un modèle LLM local.
        
        Args:
            config: Configuration du worker (optionnel)
            model_name: Nom du modèle LLM (ex: "tinyllama", "phi-3-mini") ou chemin complet
            models_dir: Répertoire contenant les modèles (défaut: depuis config ou /app/shared/models/enrichment)
            device: Device à utiliser ("cpu" uniquement pour GGUF)
        """
        self.config = config
        self.model_name = model_name or (getattr(config, 'llm_model', 'tinyllama') if config else 'tinyllama')
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
                # Essayer de télécharger si c'est un modèle recommandé
                if self.model_name in self.model_manager.RECOMMENDED_MODELS:
                    logger.info(f"📥 Attempting to download model {self.model_name}...")
                    try:
                        model_path = self.model_manager.download_model(self.model_name)
                    except Exception as download_error:
                        logger.error(f"❌ Failed to download model: {download_error}")
                        raise FileNotFoundError(f"Model file not found: {model_path}")
                else:
                    raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Vérifier la santé du modèle
            if not self.model_manager.check_model_health(model_path):
                raise ValueError(f"Model health check failed: {model_path}")
            
            self.model_path = model_path
            model_path_str = str(model_path.absolute())
            
            logger.info(
                f"📦 Loading GGUF model | "
                f"Path: {model_path_str} | "
                f"Threads: {self.n_threads} | "
                f"Context: {self.n_ctx} | "
                f"Batch: {self.n_batch}"
            )
            
            # Charger le modèle GGUF avec llama-cpp-python
            self.llm = Llama(
                model_path=model_path_str,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_batch=self.n_batch,
                n_gpu_layers=0,  # CPU only
                verbose=False,
                use_mmap=True,
                use_mlock=False
            )
            
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
                if 'tinyllama' in model_lower:
                    stop_tokens = ["</s>", "<|user|>", "<|system|>", "<|assistant|>", "\n\n\n"]
                elif 'phi-3' in model_lower or 'phi3' in model_lower:
                    stop_tokens = ["<|end|>", "<|user|>", "<|system|>", "<|assistant|>", "\n\n\n"]
                elif 'mistral' in model_lower:
                    stop_tokens = ["</s>", "[INST]", "[/INST]", "\n\n\n"]
                else:
                    stop_tokens = ["</s>", "<|im_end|>", "<|im_start|>", "\n\n\n"]
            
            # Formater le prompt selon le modèle
            model_lower = self.model_name.lower()
            if 'tinyllama' in model_lower:
                formatted_prompt = f"<|system|>\n{prompt}</s>\n<|assistant|>\n"
            elif 'phi-3' in model_lower or 'phi3' in model_lower:
                formatted_prompt = f"<|system|>\n{prompt}<|end|>\n<|assistant|>\n"
            elif 'mistral' in model_lower:
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
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
                '<|im_start|>', '<|im_end|>', '[INST]', '[/INST]', '<s>', '</s>'
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
        prompt = custom_prompt or "Génère un titre court et accrocheur (maximum 10 mots) pour cette transcription:"
        full_prompt = f"{prompt}\n\n{transcription_text[:500]}"
        
        try:
            title = self._generate_text(full_prompt, max_tokens=30, temperature=0.7)
            # Nettoyer le titre (prendre la première phrase, max 10 mots)
            words = title.split()[:10]
            return " ".join(words)
        except Exception as e:
            logger.error(f"Error generating title: {e}")
            return "Titre généré automatiquement"
    
    def generate_summary(self, transcription_text: str, custom_prompt: Optional[str] = None) -> str:
        """
        Génère un résumé de moins de 100 mots.
        
        Args:
            transcription_text: Texte de la transcription
            custom_prompt: Prompt personnalisé (optionnel)
            
        Returns:
            str: Résumé généré
        """
        prompt = custom_prompt or "Génère un résumé concis de moins de 100 mots pour cette transcription:"
        full_prompt = f"{prompt}\n\n{transcription_text}"
        
        try:
            summary = self._generate_text(full_prompt, max_tokens=150, temperature=0.7)
            # Limiter à 100 mots
            words = summary.split()[:100]
            return " ".join(words)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Résumé généré automatiquement"
    
    def generate_satisfaction_score(self, transcription_text: str, custom_prompt: Optional[str] = None) -> Dict:
        """
        Génère un score de satisfaction de 1 à 10 avec justification.
        
        Args:
            transcription_text: Texte de la transcription
            custom_prompt: Prompt personnalisé (optionnel)
            
        Returns:
            dict: {"score": int, "justification": str}
        """
        prompt = custom_prompt or "Analyse cette transcription et attribue un score de satisfaction client de 1 à 10. Réponds en JSON: {\"score\": nombre}"
        full_prompt = f"{prompt}\n\n{transcription_text}"
        
        try:
            response = self._generate_text(full_prompt, max_tokens=100, temperature=0.5)
            
            # Essayer d'extraire le JSON de la réponse
            try:
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    data = json.loads(json_str)
                    return {
                        "score": int(data.get("score", 5))
                    }
            except:
                pass
            
            # Fallback: extraire un score simple
            score_match = re.search(r'\b([1-9]|10)\b', response)
            score = int(score_match.group(1)) if score_match else 5
            return {
                "score": score
            }
        except Exception as e:
            logger.error(f"Error generating satisfaction score: {e}")
            return {"score": 5}
    
    def generate_bullet_points(self, transcription_text: str, custom_prompt: Optional[str] = None) -> list:
        """
        Génère des bullet points pour la transcription.
        
        Args:
            transcription_text: Texte de la transcription
            custom_prompt: Prompt personnalisé (optionnel)
            
        Returns:
            list: Liste de bullet points
        """
        prompt = custom_prompt or "Extrais les points clés de cette transcription sous forme de puces. Réponds en JSON: {\"points\": [\"point 1\", \"point 2\", ...]}"
        full_prompt = f"{prompt}\n\n{transcription_text}"
        
        try:
            response = self._generate_text(full_prompt, max_tokens=200, temperature=0.7)
            
            # Essayer d'extraire le JSON de la réponse
            try:
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    data = json.loads(json_str)
                    return data.get("points", [])
            except:
                pass
            
            # Fallback: extraire les points avec regex
            points = re.findall(r'[-•*]\s*(.+?)(?=\n|$)', response)
            if not points:
                # Essayer d'extraire des lignes numérotées
                points = re.findall(r'\d+[\.\)]\s*(.+?)(?=\n|$)', response)
            return points[:4] if points else ["Point clé généré automatiquement"]
        except Exception as e:
            logger.error(f"Error generating bullet points: {e}")
            return ["Point clé généré automatiquement"]
    
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
        
        # Utiliser les prompts personnalisés ou les defaults
        title_prompt = prompts.get("title") if prompts and isinstance(prompts, dict) else None
        summary_prompt = prompts.get("summary") if prompts and isinstance(prompts, dict) else None
        satisfaction_prompt = prompts.get("satisfaction") if prompts and isinstance(prompts, dict) else None
        bullet_points_prompt = prompts.get("bullet_points") if prompts and isinstance(prompts, dict) else None
        
        # Générer tous les éléments avec mesure du temps
        logger.info("Generating title...")
        title_start = time.time()
        title = self.generate_title(transcription_text, title_prompt)
        title_time = round(time.time() - title_start, 2)
        
        logger.info("Generating summary...")
        summary_start = time.time()
        summary = self.generate_summary(transcription_text, summary_prompt)
        summary_time = round(time.time() - summary_start, 2)
        
        logger.info("Generating satisfaction score...")
        satisfaction_start = time.time()
        satisfaction = self.generate_satisfaction_score(transcription_text, satisfaction_prompt)
        satisfaction_time = round(time.time() - satisfaction_start, 2)
        
        logger.info("Generating bullet points...")
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
        
        try:
            self._load_model()
            
            enriched_segments = []
            previous_text = None
            
            for i, segment in enumerate(segments):
                text = segment.get('text', '').strip()
                if not text:
                    enriched_segments.append(segment)
                    continue
                
                # Construire le prompt pour la correction
                base_instructions = (
                    "Tu es un assistant qui CORRIGE et AMÉLIORE des transcriptions audio en français. "
                    "RÈGLES STRICTES :\n"
                    "1. Corriger UNIQUEMENT les erreurs d'orthographe et de grammaire\n"
                    "2. Améliorer UNIQUEMENT la ponctuation (points, virgules, majuscules)\n"
                    "3. Améliorer UNIQUEMENT la structure (majuscules en début de phrase)\n"
                    "4. CONSERVER EXACTEMENT le sens original - ne rien ajouter, ne rien inventer\n"
                    "5. Retourner UNIQUEMENT le texte corrigé, sans explications\n"
                    "6. La longueur du texte corrigé doit être SIMILAIRE à l'original"
                )
                task_instruction = "Corrige et améliore ce texte de transcription en conservant le sens original. Retourne UNIQUEMENT le texte corrigé:"
                prompt = f"{base_instructions}\n\n{task_instruction}\n\nTexte:\n{text}"
                
                # Générer avec température très basse pour être déterministe
                estimated_tokens = len(text.split())
                max_tokens_for_text = min(256, max(50, int(estimated_tokens * 1.2)))
                
                enriched_text = self._generate_text(
                    prompt,
                    max_tokens=max_tokens_for_text,
                    temperature=0.05,  # Très bas pour correction
                    stop_tokens=["</s>", "<|end|>", "<|user|>", "<|system|>", "<|assistant|>", "\n\n\n"]
                )
                
                # Vérifier la longueur (détection d'hallucination)
                if not enriched_text:
                    enriched_text = text
                else:
                    length_ratio = len(enriched_text) / len(text) if len(text) > 0 else 1.0
                    if length_ratio > 1.5 or length_ratio < 0.5:
                        logger.warning(
                            f"⚠️ Model generated suspicious output (length mismatch: "
                            f"input={len(text)} chars, output={len(enriched_text)} chars), "
                            f"returning original text"
                        )
                        enriched_text = text
                
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
            return segments
    
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
"""
EnrichmentService - Service pour l'enrichissement des transcriptions avec LLM (llama-cpp-python)
"""

import logging
import json
import re
from typing import Dict, Optional
from pathlib import Path
from llama_cpp import Llama

logger = logging.getLogger("vocalyx.enrichment")


class EnrichmentService:
    """Service pour enrichir les transcriptions avec un LLM local (GGUF)"""
    
    def __init__(self, model_name: str = "Phi-3-mini-4k-instruct-q4.gguf", models_dir: Path = None, device: str = "cpu"):
        """
        Initialise le service d'enrichissement avec un modèle LLM local.
        
        Args:
            model_name: Nom du fichier modèle GGUF (ex: "Phi-3-mini-4k-instruct-q4.gguf")
            models_dir: Répertoire contenant les modèles (défaut: ./models/enrichment)
            device: Device à utiliser ("cpu" uniquement pour GGUF)
        """
        self.model_name = model_name
        self.device = device
        self.models_dir = models_dir or Path("/app/models/enrichment")
        self.model_path = self.models_dir / model_name
        self.llm = None
        
        logger.info(f"Initializing EnrichmentService with model: {self.model_path} (device: {device})")
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle LLM GGUF local"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading model from {self.model_path}...")
            
            # Charger le modèle GGUF avec llama-cpp-python
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=2048,  # Contexte de 2048 tokens
                n_threads=4,  # 4 threads pour CPU
                n_gpu_layers=0,  # CPU only
                verbose=False
            )
            
            logger.info(f"✅ Model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model {self.model_path}: {e}", exc_info=True)
            raise
    
    def _generate_text(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """
        Génère du texte avec le modèle LLM.
        
        Args:
            prompt: Prompt à envoyer au modèle
            max_tokens: Nombre maximum de tokens à générer
            temperature: Température pour la génération (0.0-1.0)
            
        Returns:
            str: Texte généré
        """
        try:
            # Formater le prompt pour les modèles instruct
            if "instruct" in self.model_name.lower():
                # Format pour modèles instruct (Mistral, Phi-3)
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            else:
                formatted_prompt = prompt
            
            # Générer la réponse
            response = self.llm(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "[INST]", "[/INST]"],
                echo=False
            )
            
            generated_text = response["choices"][0]["text"].strip()
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

"""
EnrichmentWorkerService - Service pour la gestion des tâches d'enrichissement
"""

import logging
import json
from typing import Dict, Optional
from infrastructure.api.api_client import VocalyxAPIClient

logger = logging.getLogger("vocalyx.enrichment")


class EnrichmentWorkerService:
    """Service pour gérer les tâches d'enrichissement du worker Celery"""
    
    def __init__(self, api_client: VocalyxAPIClient):
        self.api_client = api_client
    
    def get_transcription(self, transcription_id: str) -> Optional[Dict]:
        """Récupère une transcription depuis l'API"""
        try:
            return self.api_client.get_transcription(transcription_id)
        except Exception as e:
            logger.error(f"[{transcription_id}] Error getting transcription: {e}")
            return None
    
    def mark_enrichment_processing(self, transcription_id: str, worker_id: str) -> bool:
        """Marque l'enrichissement comme en cours de traitement"""
        try:
            self.api_client.update_transcription(transcription_id, {
                "enrichment_status": "processing",
                "enrichment_worker_id": worker_id
            })
            logger.info(f"[{transcription_id}] Enrichment status updated to 'processing'")
            return True
        except Exception as e:
            logger.error(f"[{transcription_id}] Error updating enrichment status to processing: {e}")
            return False
    
    def mark_enrichment_done(
        self,
        transcription_id: str,
        enrichment_data: Dict,
        processing_time: float
    ) -> bool:
        """Marque l'enrichissement comme terminé avec ses résultats"""
        try:
            # Convertir les données en JSON
            enrichment_json = json.dumps(enrichment_data, ensure_ascii=False)
            
            # Mettre à jour le statut de transcription à "done" maintenant que l'enrichissement est terminé
            self.api_client.update_transcription(transcription_id, {
                "status": "done",  # Transcription complètement terminée (transcription + enrichissement)
                "enrichment_status": "done",
                "enrichment_data": enrichment_json,
                "enrichment_processing_time": processing_time
            })
            logger.info(f"[{transcription_id}] Enrichment results saved to API")
            return True
        except Exception as e:
            logger.error(f"[{transcription_id}] Error saving enrichment results: {e}")
            return False
    
    def mark_enrichment_error(self, transcription_id: str, error_message: str) -> bool:
        """Marque l'enrichissement comme échoué"""
        try:
            self.api_client.update_transcription(transcription_id, {
                "enrichment_status": "error",
                "enrichment_error": str(error_message)
            })
            logger.error(f"[{transcription_id}] Marked enrichment as error: {error_message}")
            return True
        except Exception as e:
            logger.error(f"[{transcription_id}] Error marking enrichment as error: {e}")
            return False


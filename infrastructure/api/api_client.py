"""
VocalyxAPIClient - Client HTTP refactorisé pour communiquer avec vocalyx-api
"""

import logging
from typing import Dict, Optional
import httpx
import json

logger = logging.getLogger("vocalyx.enrichment")


class VocalyxAPIClient:
    """
    Client HTTP pour communiquer avec vocalyx-api.
    Le worker utilise ce client pour récupérer et mettre à jour les transcriptions.
    """
    
    def __init__(self, config):
        # Utiliser api_url (cohérent avec transcription) avec fallback
        self.base_url = getattr(config, 'api_url', getattr(config, 'vocalyx_api_url', 'http://localhost:8000')).rstrip('/')
        self.internal_key = getattr(config, 'internal_api_key', '')
        api_timeout = getattr(config, 'api_timeout', 60)
        self.timeout = httpx.Timeout(float(api_timeout), connect=5.0)
        
        # Client synchrone (suffisant pour le worker)
        self.client = httpx.Client(timeout=self.timeout)
        
        logger.info(f"API Client initialized: {self.base_url}")
        
        # Vérifier la connexion à l'API au démarrage (non bloquant)
        try:
            self._verify_connection()
        except Exception as e:
            logger.warning(f"⚠️ Could not verify API connection at startup: {e}")
    
    def _verify_connection(self):
        """Vérifie la connexion à l'API au démarrage"""
        try:
            response = self.client.get(
                f"{self.base_url}/health",
                timeout=httpx.Timeout(5.0)
            )
            response.raise_for_status()
            health = response.json()
            
            if health.get("status") == "healthy":
                logger.info("✅ API connection verified")
            else:
                logger.warning(f"⚠️ API health check returned: {health}")
        except Exception as e:
            logger.error(f"❌ API connection failed: {e}")
            logger.error("⚠️ Worker will start but may fail to process tasks")
    
    def _get_headers(self) -> Dict[str, str]:
        """Génère les headers d'authentification interne"""
        # Utiliser X-Internal-Key pour cohérence avec transcription
        return {
            "X-Internal-Key": self.internal_key
        }
    
    def get_transcription(self, transcription_id: str) -> Optional[Dict]:
        """
        Récupère une transcription par son ID.
        
        Args:
            transcription_id: ID de la transcription
            
        Returns:
            dict: Données de la transcription ou None si non trouvée
        """
        try:
            response = self.client.get(
                f"{self.base_url}/api/transcriptions/{transcription_id}",
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.error(f"Transcription {transcription_id} not found")
                return None
            # Logger les détails pour les erreurs 422 (validation)
            if e.response.status_code == 422:
                try:
                    error_detail = e.response.json()
                    logger.error(f"HTTP 422 error getting transcription {transcription_id}: {error_detail}")
                except:
                    logger.error(f"HTTP 422 error getting transcription {transcription_id}: {e.response.text}")
            else:
                logger.error(f"HTTP error getting transcription: {e}")
            raise
        except httpx.HTTPError as e:
            logger.error(f"Error getting transcription: {e}")
            raise
    
    def update_transcription(self, transcription_id: str, data: Dict) -> Dict:
        """
        Met à jour une transcription.
        
        Args:
            transcription_id: ID de la transcription
            data: Données à mettre à jour
            
        Returns:
            dict: Transcription mise à jour
        """
        try:
            # Logger les données envoyées (sans les segments complets pour éviter le spam)
            log_data = {k: v if k != 'enriched_segments' else f'<{len(json.loads(v) if isinstance(v, str) else v)} segments>' for k, v in data.items()}
            logger.debug(f"📤 PATCH /api/transcriptions/{transcription_id} | Data: {log_data}")
            
            response = self.client.patch(
                f"{self.base_url}/api/transcriptions/{transcription_id}",
                json=data,
                headers=self._get_headers()
            )
            response.raise_for_status()
            result = response.json()
            logger.debug(f"✅ PATCH /api/transcriptions/{transcription_id} | Response status: {response.status_code} | Updated fields: {list(data.keys())}")
            return result
        except httpx.HTTPStatusError as e:
            # Logger les détails pour les erreurs 422 (validation)
            if e.response.status_code == 422:
                try:
                    error_detail = e.response.json()
                    logger.error(f"HTTP 422 error updating transcription {transcription_id}: {error_detail}")
                except:
                    logger.error(f"HTTP 422 error updating transcription {transcription_id}: {e.response.text}")
            else:
                logger.error(f"HTTP error updating transcription: {e}")
            raise
        except httpx.HTTPError as e:
            logger.error(f"Error updating transcription: {e}")
            raise
    
    def close(self):
        """Ferme le client HTTP"""
        self.client.close()


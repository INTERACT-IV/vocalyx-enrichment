"""
Client API pour communiquer avec vocalyx-api
"""

import logging
import requests
from typing import Optional, Dict, Any

logger = logging.getLogger("vocalyx")


class VocalyxAPIClient:
    """Client pour communiquer avec l'API Vocalyx"""
    
    def __init__(self, config):
        """
        Initialise le client API.
        
        Args:
            config: Configuration du worker
        """
        self.api_url = config.vocalyx_api_url
        self.api_key = config.internal_api_key
        self.timeout = getattr(config, 'api_timeout', 60)
        self.headers = {
            'Content-Type': 'application/json',
            'X-Internal-API-Key': self.api_key
        }
    
    def get_transcription(self, transcription_id: str) -> Optional[Dict]:
        """
        Récupère une transcription depuis l'API.
        
        Args:
            transcription_id: ID de la transcription
            
        Returns:
            Dictionnaire avec les données de transcription ou None
        """
        try:
            url = f"{self.api_url}/api/transcriptions/{transcription_id}"
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Error getting transcription {transcription_id}: {e}")
            return None
    
    def update_transcription(self, transcription_id: str, data: Dict[str, Any]) -> bool:
        """
        Met à jour une transcription dans l'API.
        
        Args:
            transcription_id: ID de la transcription
            data: Données à mettre à jour
            
        Returns:
            True si succès, False sinon
        """
        try:
            url = f"{self.api_url}/api/transcriptions/{transcription_id}"
            response = requests.patch(url, json=data, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"❌ Error updating transcription {transcription_id}: {e}")
            return False

"""
vocalyx-enrichment/config.py
Configuration du worker d'enrichissement (adapté pour l'architecture microservices)
"""

import os
import logging
import configparser
from pathlib import Path

logger = logging.getLogger("vocalyx")

class Config:
    """Charge et gère la configuration depuis config.ini"""
    
    def __init__(self, config_file: str = "config.ini"):
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        
        if not os.path.exists(config_file):
            self._create_default_config()
        
        self.config.read(config_file)
        self._load_settings()
        
    def _create_default_config(self):
        """Crée un fichier de configuration par défaut"""
        config = configparser.ConfigParser()
        
        config['CORE'] = {
            'instance_name': 'enrichment-worker-01'
        }
        
        config['API'] = {
            'url': 'http://localhost:8000',
            'timeout': '60'
        }
        
        config['CELERY'] = {
            'broker_url': 'redis://localhost:6379/0',
            'result_backend': 'redis://localhost:6379/0'
        }
        
        config['LLM'] = {
            'device': 'cpu',
            'max_length': '512',
            'temperature': '0.7',
            'model_cache_dir': './models/llm'
        }
        
        config['PROMPTS'] = {
            'title_prompt': 'Génère un titre court et accrocheur (maximum 10 mots) pour cette transcription:',
            'summary_prompt': 'Génère un résumé concis de moins de 100 mots pour cette transcription:',
            'satisfaction_prompt': 'Analyse cette transcription et attribue un score de satisfaction client de 1 à 10. Justifie brièvement ton score. Format JSON: {"score": nombre, "justification": "texte"}',
            'bullet_points_prompt': 'Extrais les points clés de cette transcription sous forme de puces. Format JSON: {"points": ["point 1", "point 2", ...]}'
        }
        
        config['LOGGING'] = {
            'level': 'INFO',
            'file_enabled': 'true',
            'file_path': 'logs/vocalyx-enrichment.log',
            'colored': 'true'
        }
        
        config['SECURITY'] = {
            'internal_api_key': 'CHANGE_ME_SECRET_INTERNAL_KEY_12345'
        }
        
        with open(self.config_file, 'w') as f:
            config.write(f)
        
        logging.info(f"✅ Created default config file: {self.config_file}")
    
    def _load_settings(self):
        """Charge les paramètres dans des attributs"""
        
        # CORE
        self.instance_name = os.environ.get(
            'INSTANCE_NAME', 
            self.config.get('CORE', 'instance_name', fallback=f"enrichment-worker-{os.getpid()}")
        )
        
        # API
        self.api_url = os.environ.get(
            'VOCALYX_API_URL', 
            self.config.get('API', 'url')
        )
        self.api_timeout = self.config.getint('API', 'timeout', fallback=60)
        
        # CELERY
        self.celery_broker_url = os.environ.get(
            'CELERY_BROKER_URL', 
            self.config.get('CELERY', 'broker_url')
        )
        self.celery_result_backend = os.environ.get(
            'CELERY_RESULT_BACKEND', 
            self.config.get('CELERY', 'result_backend')
        )
        
        # LLM
        self.device = os.environ.get(
            'LLM_DEVICE', 
            self.config.get('LLM', 'device', fallback='cpu')
        )
        self.max_length = self.config.getint('LLM', 'max_length', fallback=512)
        self.temperature = self.config.getfloat('LLM', 'temperature', fallback=0.7)
        self.model_cache_dir = Path(self.config.get('LLM', 'model_cache_dir', fallback='./models/llm'))
        
        # PROMPTS
        self.title_prompt = self.config.get('PROMPTS', 'title_prompt', fallback='Génère un titre pour cette transcription:')
        self.summary_prompt = self.config.get('PROMPTS', 'summary_prompt', fallback='Génère un résumé de moins de 100 mots pour cette transcription:')
        self.satisfaction_prompt = self.config.get('PROMPTS', 'satisfaction_prompt', fallback='Score de satisfaction 1-10:')
        self.bullet_points_prompt = self.config.get('PROMPTS', 'bullet_points_prompt', fallback='Extrais les points clés:')
        
        # SECURITY
        self.internal_api_key = os.environ.get(
            'INTERNAL_API_KEY', 
            self.config.get('SECURITY', 'internal_api_key')
        )
        
        if self.internal_api_key == 'CHANGE_ME_SECRET_INTERNAL_KEY_12345':
            logging.warning("⚠️ SECURITY: Internal API key is using default value. Please change it!")
        
        # LOGGING
        self.log_level = os.environ.get(
            'LOG_LEVEL', 
            self.config.get('LOGGING', 'level', fallback='INFO')
        )
        self.log_file_enabled = self.config.getboolean('LOGGING', 'file_enabled', fallback=True)
        self.log_file_path = os.environ.get(
            'LOG_FILE_PATH',
            self.config.get('LOGGING', 'file_path', fallback='logs/vocalyx-enrichment.log')
        )
        
        log_colored_str = os.environ.get(
            'LOG_COLORED', 
            self.config.get('LOGGING', 'colored', fallback='true')
        )
        self.log_colored = log_colored_str.lower() in ['true', '1', 't']
    
    def reload(self):
        """Recharge la configuration depuis le fichier"""
        self.config.read(self.config_file)
        self._load_settings()
        logging.info("🔄 Configuration reloaded")


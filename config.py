"""
Configuration pour le worker d'enrichissement
"""

import os
import configparser
from pathlib import Path


class Config:
    """Configuration du worker d'enrichissement"""
    
    def __init__(self, config_file: str = None):
        """
        Initialise la configuration depuis les variables d'environnement ou un fichier.
        
        Args:
            config_file: Chemin vers le fichier de configuration (optionnel)
        """
        # Valeurs par défaut
        self.instance_name = os.getenv('INSTANCE_NAME', 'enrichment-worker-01')
        # API URL - utiliser api_url pour cohérence avec transcription
        self.api_url = os.getenv('VOCALYX_API_URL', 'http://localhost:8000')
        # Alias pour compatibilité
        self.vocalyx_api_url = self.api_url
        self.celery_broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
        self.celery_result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
        self.redis_enrichment_url = os.getenv('REDIS_ENRICHMENT_URL', 'redis://localhost:6379/3')
        self.internal_api_key = os.getenv('INTERNAL_API_KEY', '')
        
        # LLM
        # Par défaut, chercher dans /app/shared/models/enrichment (Docker) ou ./shared/models/enrichment (local)
        default_model = os.getenv('LLM_MODEL', 'phi-3-mini')
        self.llm_model = default_model
        # Utiliser /app/shared/models/enrichment comme dans transcription (Docker)
        self.llm_models_dir = os.getenv('LLM_MODELS_DIR', '/app/shared/models/enrichment')
        self.llm_device = os.getenv('LLM_DEVICE', 'cpu')
        self.llm_compute_type = os.getenv('LLM_COMPUTE_TYPE', 'int8')
        self.llm_max_tokens = int(os.getenv('LLM_MAX_TOKENS', '256'))
        self.llm_temperature = float(os.getenv('LLM_TEMPERATURE', '0.7'))
        self.llm_top_p = float(os.getenv('LLM_TOP_P', '0.9'))
        self.llm_top_k = int(os.getenv('LLM_TOP_K', '40'))
        
        # Paramètres CPU pour llama-cpp-python
        self.llm_n_threads = int(os.getenv('LLM_N_THREADS', '0')) if os.getenv('LLM_N_THREADS') else None  # Auto-détecté si 0 ou None
        self.llm_n_ctx = int(os.getenv('LLM_N_CTX', '2048'))  # Taille du contexte
        self.llm_n_batch = int(os.getenv('LLM_N_BATCH', '512'))  # Batch size pour CPU
        
        # Performance
        self.max_workers = int(os.getenv('MAX_WORKERS', '2'))
        self.max_chunk_size = int(os.getenv('MAX_CHUNK_SIZE', '500'))
        self.enable_cache = os.getenv('ENABLE_CACHE', 'true').lower() == 'true'
        self.cache_max_models = int(os.getenv('CACHE_MAX_MODELS', '2'))
        
        # Redis
        self.redis_enrichment_compress = os.getenv('REDIS_ENRICHMENT_COMPRESS', 'true').lower() == 'true'
        self.redis_enrichment_ttl = int(os.getenv('REDIS_ENRICHMENT_TTL', '3600'))
        
        # Logging
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_colored = os.getenv('LOG_COLORED', 'false').lower() == 'true'
        self.log_file_enabled = os.getenv('LOG_FILE_ENABLED', 'true').lower() == 'true'
        self.log_file_path = os.getenv('LOG_FILE_PATH', '/app/logs/vocalyx-enrichment.log')
        
        # API
        self.api_timeout = int(os.getenv('API_TIMEOUT', '60'))
        
        # Charger depuis fichier si fourni
        if config_file and Path(config_file).exists():
            self._load_from_file(config_file)
    
    def _load_from_file(self, config_file: str):
        """Charge la configuration depuis un fichier INI"""
        config = configparser.ConfigParser()
        config.read(config_file)
        
        # CORE
        if config.has_section('CORE'):
            self.instance_name = config.get('CORE', 'instance_name', fallback=self.instance_name)
        
        # API
        if config.has_section('API'):
            self.api_url = config.get('API', 'url', fallback=self.api_url)
            # Alias pour compatibilité
            self.vocalyx_api_url = self.api_url
            self.api_timeout = config.getint('API', 'timeout', fallback=self.api_timeout)
        
        # CELERY
        if config.has_section('CELERY'):
            self.celery_broker_url = config.get('CELERY', 'broker_url', fallback=self.celery_broker_url)
            self.celery_result_backend = config.get('CELERY', 'result_backend', fallback=self.celery_result_backend)
        
        # LLM
        if config.has_section('LLM'):
            self.llm_model = config.get('LLM', 'model', fallback=self.llm_model)
            self.llm_device = config.get('LLM', 'device', fallback=self.llm_device)
            self.llm_compute_type = config.get('LLM', 'compute_type', fallback=self.llm_compute_type)
            self.llm_max_tokens = config.getint('LLM', 'max_tokens', fallback=self.llm_max_tokens)
            self.llm_temperature = config.getfloat('LLM', 'temperature', fallback=self.llm_temperature)
        
        # PERFORMANCE
        if config.has_section('PERFORMANCE'):
            self.max_workers = config.getint('PERFORMANCE', 'max_workers', fallback=self.max_workers)
            self.max_chunk_size = config.getint('PERFORMANCE', 'max_chunk_size', fallback=self.max_chunk_size)
            self.enable_cache = config.getboolean('PERFORMANCE', 'enable_cache', fallback=self.enable_cache)
            self.cache_max_models = config.getint('PERFORMANCE', 'cache_max_models', fallback=self.cache_max_models)
        
        # REDIS_ENRICHMENT
        if config.has_section('REDIS_ENRICHMENT'):
            self.redis_enrichment_url = config.get('REDIS_ENRICHMENT', 'url', fallback=self.redis_enrichment_url)
            self.redis_enrichment_compress = config.getboolean('REDIS_ENRICHMENT', 'compress_data', fallback=self.redis_enrichment_compress)
            self.redis_enrichment_ttl = config.getint('REDIS_ENRICHMENT', 'default_ttl', fallback=self.redis_enrichment_ttl)
        
        # LOGGING
        if config.has_section('LOGGING'):
            self.log_level = config.get('LOGGING', 'level', fallback=self.log_level)
            self.log_file_enabled = config.getboolean('LOGGING', 'file_enabled', fallback=self.log_file_enabled)
            self.log_file_path = config.get('LOGGING', 'file_path', fallback=self.log_file_path)
            self.log_colored = config.getboolean('LOGGING', 'colored', fallback=self.log_colored)

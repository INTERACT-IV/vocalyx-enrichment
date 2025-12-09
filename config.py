"""
Configuration pour le worker d'enrichissement
Adapté pour fonctionner comme vocalyx-transcribe (charge depuis config.ini par défaut)
"""

import os
import logging
import configparser
from pathlib import Path

logger = logging.getLogger("vocalyx")

class Config:
    """Charge et gère la configuration depuis config.ini (comme transcription)"""
    
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
        
        config['REDIS_ENRICHMENT'] = {
            # DB Redis dédiée pour les opérations d'enrichissement (isolation des données)
            'url': 'redis://localhost:6379/3',
            # Compression des données JSON (réduit la mémoire et le réseau)
            'compress_data': 'true',
            # TTL par défaut pour les chunks (en secondes)
            'default_ttl': '3600'
        }
        
        config['LLM'] = {
            'model': 'tinyllama',
            'models_dir': '/app/shared/models/enrichment',
            'device': 'cpu',
            'compute_type': 'int8',
            'max_tokens': '256',
            'temperature': '0.7',
            'top_p': '0.9',
            'top_k': '40',
            'n_threads': '0',
            'n_ctx': '2048',
            'n_batch': '512'
        }
        
        config['PERFORMANCE'] = {
            'max_workers': '2',
            'max_chunk_size': '500',
            'enable_cache': 'true',
            'cache_max_models': '2',
            'enrichment_distribution_threshold': '10'
        }
        
        config['LOGGING'] = {
            'level': 'INFO',
            'file_enabled': 'true',
            'file_path': '/app/logs/vocalyx-enrichment.log',
            'colored': 'true'
        }
        
        config['SECURITY'] = {
            'internal_api_key': 'CHANGE_ME_SECRET_INTERNAL_KEY_12345'
        }
        
        with open(self.config_file, 'w') as f:
            config.write(f)
        
        logger.info(f"✅ Created default config file: {self.config_file}")
    
    def _load_settings(self):
        """Charge les paramètres dans des attributs (comme transcription)"""
        
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
        # Alias pour compatibilité
        self.vocalyx_api_url = self.api_url
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
        
        # REDIS ENRICHMENT (DB dédiée pour les chunks)
        # Utilise DB 3 par défaut pour isoler des opérations Celery (DB 0) et Transcription (DB 2)
        # PRIORITÉ: Variable d'environnement > config.ini > fallback depuis CELERY_BROKER_URL
        redis_enrichment_url = os.environ.get('REDIS_ENRICHMENT_URL', None)
        source = "environment variable"
        
        if not redis_enrichment_url:
            # Essayer depuis config.ini seulement si la section existe
            try:
                redis_enrichment_url = self.config.get('REDIS_ENRICHMENT', 'url')
                source = "config.ini"
            except (configparser.NoSectionError, configparser.NoOptionError):
                redis_enrichment_url = None
        
        if redis_enrichment_url:
            self.redis_enrichment_url = redis_enrichment_url
            logger.info(f"✅ Redis enrichment URL ({source}): {redis_enrichment_url}")
        else:
            # Fallback : utiliser DB 3 de la même instance Redis (depuis CELERY_BROKER_URL)
            base_redis_url = self.celery_broker_url.rsplit('/', 1)[0]  # Enlever /0
            self.redis_enrichment_url = f"{base_redis_url}/3"
            logger.info(f"✅ Redis enrichment URL (fallback from CELERY_BROKER_URL): {self.redis_enrichment_url}")
        
        self.redis_enrichment_compress = self.config.getboolean(
            'REDIS_ENRICHMENT', 'compress_data', fallback=True
        )
        self.redis_enrichment_ttl = self.config.getint(
            'REDIS_ENRICHMENT', 'default_ttl', fallback=3600
        )
        
        # LLM
        self.llm_model = os.environ.get(
            'LLM_MODEL', 
            self.config.get('LLM', 'model', fallback='tinyllama')
        )
        self.llm_models_dir = os.environ.get(
            'LLM_MODELS_DIR',
            self.config.get('LLM', 'models_dir', fallback='/app/shared/models/enrichment')
        )
        self.llm_device = os.environ.get(
            'LLM_DEVICE', 
            self.config.get('LLM', 'device', fallback='cpu')
        )
        self.llm_compute_type = os.environ.get(
            'LLM_COMPUTE_TYPE', 
            self.config.get('LLM', 'compute_type', fallback='int8')
        )
        self.llm_max_tokens = int(os.environ.get(
            'LLM_MAX_TOKENS',
            self.config.get('LLM', 'max_tokens', fallback='256')
        ))
        self.llm_temperature = float(os.environ.get(
            'LLM_TEMPERATURE',
            self.config.get('LLM', 'temperature', fallback='0.7')
        ))
        self.llm_top_p = float(os.environ.get(
            'LLM_TOP_P',
            self.config.get('LLM', 'top_p', fallback='0.9')
        ))
        self.llm_top_k = int(os.environ.get(
            'LLM_TOP_K',
            self.config.get('LLM', 'top_k', fallback='40')
        ))
        
        # Paramètres CPU pour llama-cpp-python
        n_threads_str = os.environ.get(
            'LLM_N_THREADS',
            self.config.get('LLM', 'n_threads', fallback='0')
        )
        self.llm_n_threads = int(n_threads_str) if n_threads_str and n_threads_str != '0' else None  # Auto-détecté si 0 ou None
        self.llm_n_ctx = int(os.environ.get(
            'LLM_N_CTX',
            self.config.get('LLM', 'n_ctx', fallback='2048')
        ))
        self.llm_n_batch = int(os.environ.get(
            'LLM_N_BATCH',
            self.config.get('LLM', 'n_batch', fallback='512')
        ))
        
        # PERFORMANCE
        self.max_workers = int(os.environ.get(
            'MAX_WORKERS', 
            self.config.getint('PERFORMANCE', 'max_workers', fallback=2)
        ))
        self.max_chunk_size = int(os.environ.get(
            'MAX_CHUNK_SIZE',
            self.config.getint('PERFORMANCE', 'max_chunk_size', fallback=500)
        ))
        
        enable_cache_str = os.environ.get(
            'ENABLE_CACHE',
            self.config.get('PERFORMANCE', 'enable_cache', fallback='true')
        )
        self.enable_cache = enable_cache_str.lower() in ['true', '1', 't']
        
        self.cache_max_models = int(os.environ.get(
            'CACHE_MAX_MODELS',
            self.config.getint('PERFORMANCE', 'cache_max_models', fallback=2)
        ))
        
        # Seuil pour activer le mode distribué (nombre de segments)
        self.enrichment_distribution_threshold = int(os.environ.get(
            'ENRICHMENT_DISTRIBUTION_THRESHOLD',
            self.config.getint('PERFORMANCE', 'enrichment_distribution_threshold', fallback=10)
        ))
        
        # SECURITY
        self.internal_api_key = os.environ.get(
            'INTERNAL_API_KEY', 
            self.config.get('SECURITY', 'internal_api_key', fallback='')
        )
        
        if self.internal_api_key == 'CHANGE_ME_SECRET_INTERNAL_KEY_12345':
            logger.warning("⚠️ SECURITY: Internal API key is using default value. Please change it!")
        
        # LOGGING
        self.log_level = os.environ.get(
            'LOG_LEVEL', 
            self.config.get('LOGGING', 'level', fallback='INFO')
        )
        self.log_file_enabled = self.config.getboolean('LOGGING', 'file_enabled', fallback=True)
        self.log_file_path = os.environ.get(
            'LOG_FILE_PATH',
            self.config.get('LOGGING', 'file_path', fallback='/app/logs/vocalyx-enrichment.log')
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
        logger.info("🔄 Configuration reloaded")

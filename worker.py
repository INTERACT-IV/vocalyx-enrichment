"""
vocalyx-enrichment/worker.py
Worker Celery pour l'enrichissement des transcriptions avec LLM
"""

import logging
import time
import os
import json
import psutil
import threading
from datetime import datetime
from celery.signals import worker_init
from celery.worker.control import Panel

from celery import Celery
from config import Config
from infrastructure.api.api_client import VocalyxAPIClient
from application.services.enrichment_worker_service import EnrichmentWorkerService
from enrichment_service import EnrichmentService

config = Config()

from logging_config import setup_logging, setup_colored_logging

if config.log_colored:
    logger = setup_colored_logging(
        log_level=config.log_level,
        log_file=config.log_file_path if config.log_file_enabled else None
    )
else:
    logger = setup_logging(
        log_level=config.log_level,
        log_file=config.log_file_path if config.log_file_enabled else None
    )

# Variables globales pour les services
_api_client = None
_enrichment_service = None
_enrichment_service_lock = threading.Lock()

# Variables globales pour psutil
WORKER_PROCESS = None
WORKER_START_TIME = None

@worker_init.connect
def on_worker_init(**kwargs):
    """Initialise psutil quand le worker démarre."""
    global WORKER_PROCESS, WORKER_START_TIME
    try:
        WORKER_PROCESS = psutil.Process(os.getpid())
        WORKER_START_TIME = datetime.now()
        WORKER_PROCESS.cpu_percent(interval=None)  # Initialiser la mesure
        logger.info(f"Worker {WORKER_PROCESS.pid} initialisé pour monitoring psutil.")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de psutil: {e}")


def get_api_client():
    """Charge le client API (une fois par worker)"""
    global _api_client
    if _api_client is None:
        logger.info(f"Initialisation du client API pour ce worker ({config.instance_name})...")
        _api_client = VocalyxAPIClient(config)
    return _api_client


def get_worker_service():
    """Charge le service worker (une fois par worker)"""
    api_client = get_api_client()
    return EnrichmentWorkerService(api_client)


def get_enrichment_service(model_name: str = "Phi-3-mini-4k-instruct-q4.gguf"):
    """
    Charge le service d'enrichissement avec cache du modèle.
    
    Args:
        model_name: Nom du fichier modèle GGUF à utiliser (ex: "Phi-3-mini-4k-instruct-q4.gguf")
        
    Returns:
        EnrichmentService: Service d'enrichissement avec le modèle demandé
    """
    global _enrichment_service, _enrichment_service_lock
    from pathlib import Path
    
    # Si c'est un ancien modèle DialoGPT, le remplacer par un modèle GGUF local
    if model_name and ("DialoGPT" in model_name or "microsoft/" in model_name):
        logger.warning(f"⚠️ Ancien modèle DialoGPT détecté ({model_name}), remplacement par Phi-3-mini-4k-instruct-q4.gguf")
        model_name = "Phi-3-mini-4k-instruct-q4.gguf"
    
    # Si le modèle n'est pas spécifié, utiliser le défaut
    if not model_name:
        model_name = "Phi-3-mini-4k-instruct-q4.gguf"
    
    with _enrichment_service_lock:
        if _enrichment_service is None or _enrichment_service.model_name != model_name:
            logger.info(f"🚀 Loading LLM model: {model_name}")
            try:
                # Les modèles sont dans /app/models/enrichment (monté depuis ./shared/models/enrichment)
                models_dir = Path("/app/models/enrichment")
                
                # Vérifier que le fichier existe
                model_path = models_dir / model_name
                if not model_path.exists():
                    # Essayer de lister les fichiers disponibles
                    available_models = list(models_dir.glob("*.gguf")) if models_dir.exists() else []
                    if available_models:
                        logger.warning(f"⚠️ Modèle {model_name} non trouvé, utilisation du premier modèle disponible: {available_models[0].name}")
                        model_name = available_models[0].name
                    else:
                        raise FileNotFoundError(
                            f"Modèle {model_name} non trouvé dans {models_dir}. "
                            f"Aucun modèle GGUF disponible. Vérifiez que les modèles sont dans ./shared/models/enrichment/"
                        )
                
                _enrichment_service = EnrichmentService(
                    model_name=model_name, 
                    models_dir=models_dir,
                    device=config.device
                )
                logger.info(f"✅ Model {model_name} loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load model {model_name}: {e}", exc_info=True)
                raise
        return _enrichment_service


# Créer l'application Celery (connexion au même broker que vocalyx-api)
celery_app = Celery(
    'vocalyx-enrichment',
    broker=config.celery_broker_url,
    backend=config.celery_result_backend
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=10,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
    worker_send_task_events=True,
    task_send_sent_event=True,
    worker_log_format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    worker_task_log_format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    worker_log_datefmt='%Y-%m-%d %H:%M:%S',
    worker_disable_rate_limits=False,
    worker_hijack_root_logger=False,
)


@Panel.register(
    name='get_worker_health',
    alias='health'
)
def get_worker_health_handler(state, **kwargs):
    """Handler pour la commande de contrôle 'get_worker_health'."""
    if WORKER_PROCESS is None:
        logger.warning("get_worker_health_handler appelé avant initialisation de psutil.")
        return {'error': 'Worker not initialized'}
    
    try:
        mem_info = WORKER_PROCESS.memory_info()
        uptime_seconds = (datetime.now() - WORKER_START_TIME).total_seconds()
        
        health_data = {
            'pid': WORKER_PROCESS.pid,
            'cpu_percent': WORKER_PROCESS.cpu_percent(interval=None),
            'memory_rss_bytes': mem_info.rss,
            'memory_percent': WORKER_PROCESS.memory_percent(),
            'uptime_seconds': uptime_seconds
        }
        
        return health_data
    except Exception as e:
        logger.error(f"Erreur dans get_worker_health_handler: {e}", exc_info=True)
        return {'error': str(e)}


@celery_app.task(
    bind=True,
    name='enrich_transcription',
    max_retries=3,
    default_retry_delay=60,
    soft_time_limit=1800,
    time_limit=2100,
    acks_late=True,
    reject_on_worker_lost=True,
    queue='enrichment'  # Queue spécifique pour l'enrichissement
)
def enrich_transcription_task(self, transcription_id: str):
    """
    Tâche d'enrichissement exécutée par le worker.
    """
    
    logger.info(f"[{transcription_id}] 🎯 Enrichment task started by worker {config.instance_name}")
    start_time = time.time()
    
    try:
        # 1. Récupérer les informations de la transcription depuis l'API
        logger.info(f"[{transcription_id}] 📡 Fetching transcription data from API...")
        worker_service = get_worker_service()
        transcription = worker_service.get_transcription(transcription_id)
        
        if not transcription:
            raise ValueError(f"Transcription {transcription_id} not found")
        
        text = transcription.get('text')
        if not text:
            raise ValueError(f"Transcription {transcription_id} has no text (transcription not completed)")
        
        # Vérifier si l'enrichissement est demandé
        enrichment_requested = transcription.get('enrichment_requested', False)
        if not enrichment_requested:
            logger.info(f"[{transcription_id}] Enrichment not requested, skipping...")
            return {
                "status": "skipped",
                "transcription_id": transcription_id,
                "reason": "enrichment not requested"
            }
        
        # Récupérer le modèle LLM à utiliser (depuis la DB ou config par défaut)
        llm_model = transcription.get('llm_model') or "Phi-3-mini-4k-instruct-q4.gguf"
        
        # Si c'est un ancien modèle DialoGPT, le remplacer par un modèle GGUF local
        if llm_model and ("DialoGPT" in llm_model or "microsoft/" in llm_model):
            logger.warning(f"[{transcription_id}] ⚠️ Ancien modèle DialoGPT détecté ({llm_model}), remplacement par Phi-3-mini-4k-instruct-q4.gguf")
            llm_model = "Phi-3-mini-4k-instruct-q4.gguf"
        
        # Récupérer les prompts personnalisés (s'il y en a)
        enrichment_prompts_str = transcription.get('enrichment_prompts')
        enrichment_prompts = None
        if enrichment_prompts_str:
            try:
                if isinstance(enrichment_prompts_str, str):
                    enrichment_prompts = json.loads(enrichment_prompts_str)
                elif isinstance(enrichment_prompts_str, dict):
                    enrichment_prompts = enrichment_prompts_str
            except json.JSONDecodeError as e:
                logger.warning(f"[{transcription_id}] Invalid JSON for enrichment_prompts: {e}")
                enrichment_prompts = None
        
        logger.info(f"[{transcription_id}] 📝 Text length: {len(text)} chars | LLM Model: {llm_model}")
        
        # 2. Mettre à jour le statut d'enrichissement à "processing"
        worker_service.mark_enrichment_processing(transcription_id, config.instance_name)
        logger.info(f"[{transcription_id}] ⚙️ Enrichment status updated to 'processing'")
        
        # 3. Obtenir le service d'enrichissement
        logger.info(f"[{transcription_id}] 🤖 Getting enrichment service with model: {llm_model}")
        enrichment_service = get_enrichment_service(model_name=llm_model)
        
        # 4. Exécuter l'enrichissement
        logger.info(f"[{transcription_id}] 🤖 Starting enrichment with LLM...")
        
        enrichment_data = enrichment_service.enrich_transcription(
            transcription_text=text,
            prompts=enrichment_prompts
        )
        
        logger.info(f"[{transcription_id}] ✅ Enrichment service completed")
        
        # Utiliser le temps total du timing si disponible (plus précis), sinon calculer le temps total
        if enrichment_data.get('timing') and enrichment_data['timing'].get('total_time'):
            processing_time = enrichment_data['timing']['total_time']
        else:
            processing_time = round(time.time() - start_time, 2)
        
        logger.info(
            f"[{transcription_id}] ✅ Enrichment completed | "
            f"Processing: {processing_time}s"
        )
        
        # 5. Mettre à jour avec les résultats
        logger.info(f"[{transcription_id}] 💾 Saving enrichment results to API...")
        worker_service.mark_enrichment_done(transcription_id, enrichment_data, processing_time)
        
        logger.info(f"[{transcription_id}] 💾 Enrichment results saved to API")
        
        return {
            "status": "success",
            "transcription_id": transcription_id,
            "processing_time": processing_time,
            "enrichment_data": enrichment_data
        }
        
    except Exception as e:
        logger.error(f"[{transcription_id}] ❌ Error: {e}", exc_info=True)
        
        # Mettre à jour le statut d'enrichissement à "error"
        try:
            worker_service_on_error = get_worker_service()
            worker_service_on_error.mark_enrichment_error(transcription_id, str(e))
        except Exception as update_error:
            logger.error(f"[{transcription_id}] Failed to update error status: {update_error}")
        
        # Retry si possible
        if self.request.retries < self.max_retries:
            logger.warning(f"[{transcription_id}] ⏳ Retrying in {self.default_retry_delay}s...")
            raise self.retry(exc=e)
        
        # Si toutes les tentatives échouent
        logger.error(f"[{transcription_id}] ⛔ All retries exhausted")
        return {
            "status": "error",
            "transcription_id": transcription_id,
            "error": str(e)
        }


if __name__ == "__main__":
    logger.info(f"🚀 Starting Celery enrichment worker: {config.instance_name}")
    celery_app.worker_main([
        'worker',
        f'--loglevel={config.log_level.lower()}',
        f'--concurrency=2',
        f'--hostname={config.instance_name}@%h'
    ])


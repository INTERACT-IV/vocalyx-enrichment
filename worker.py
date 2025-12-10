"""
vocalyx-enrichment/worker.py
Worker Celery pour l'enrichissement de transcriptions avec LLM
Architecture distribuée avec workers partagés et agrégation
"""

import logging
import time
import os
import psutil
import json
import redis
from pathlib import Path
from datetime import datetime
from celery.signals import worker_init
from celery.worker.control import Panel
from celery import Celery

from config import Config
from infrastructure.api.api_client import VocalyxAPIClient
from infrastructure.redis.redis_enrichment_manager import RedisCompressionManager, RedisEnrichmentManager
from infrastructure.models.llm_model_cache import LLMModelCache
from application.services.chunk_splitter import ChunkSplitter
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

# Variables globales pour les services (singletons par worker)
_api_client = None
_redis_client = None
_redis_manager = None
_model_cache = None

# Variables globales pour psutil
WORKER_PROCESS = None
WORKER_START_TIME = None


@worker_init.connect
def on_worker_init(**kwargs):
    """Initialise psutil et pré-charge le modèle au démarrage"""
    global WORKER_PROCESS, WORKER_START_TIME, _model_cache
    try:
        WORKER_PROCESS = psutil.Process(os.getpid())
        WORKER_START_TIME = datetime.now()
        WORKER_PROCESS.cpu_percent(interval=None)
        logger.info(f"Worker {WORKER_PROCESS.pid} initialisé pour monitoring psutil.")
        
        # Initialiser le cache avec la classe EnrichmentService (évite les problèmes d'import)
        # Le cache sera initialisé dans get_llm_service si nécessaire
        # Pré-charger le modèle par défaut (warm-up)
        if config.enable_cache:
            logger.info("🔥 Warming up LLM model cache...")
            try:
                get_llm_service(config.llm_model)
                logger.info("✅ LLM model cache warmed up")
            except Exception as e:
                logger.warning(f"⚠️ Failed to warm up model cache: {e}")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du worker: {e}")


def get_redis_client():
    """Obtient un client Redis pour stocker les résultats des chunks"""
    global _redis_client
    if _redis_client is None:
        redis_url = getattr(config, 'redis_enrichment_url', None)
        if not redis_url:
            # Par défaut, utiliser DB 3 pour l'enrichissement
            base_url = config.celery_broker_url.rsplit('/', 1)[0]
            redis_url = f"{base_url}/3"
        
        logger.info(f"🔌 Initializing Redis enrichment client: {redis_url}")
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        
        try:
            _redis_client.ping()
            logger.info(f"✅ Redis enrichment client connected successfully: {redis_url}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis enrichment: {redis_url} - {e}")
            raise
    
    return _redis_client


def get_redis_manager() -> RedisEnrichmentManager:
    """Obtient le gestionnaire Redis pour les opérations d'enrichissement"""
    global _redis_manager
    if _redis_manager is None:
        redis_client = get_redis_client()
        compression = RedisCompressionManager(
            enabled=getattr(config, 'redis_enrichment_compress', True)
        )
        _redis_manager = RedisEnrichmentManager(redis_client, compression)
    return _redis_manager


def get_api_client():
    """Charge le client API (une fois par worker)"""
    global _api_client
    if _api_client is None:
        logger.info(f"Initialisation du client API pour ce worker ({config.instance_name})...")
        _api_client = VocalyxAPIClient(config)
    return _api_client


def get_llm_service(model_name: str = None):
    """
    Charge le service LLM avec cache par modèle.
    
    Args:
        model_name: Nom du modèle LLM ou chemin (défaut: config.llm_model)
        
    Returns:
        EnrichmentService: Service d'enrichissement avec le modèle demandé
    """
    global _model_cache
    if _model_cache is None:
        max_models = getattr(config, 'cache_max_models', 2)
        # Passer EnrichmentService au cache pour éviter les problèmes d'import
        _model_cache = LLMModelCache(max_models=max_models, enrichment_service_class=EnrichmentService)
    
    if model_name is None:
        model_name = config.llm_model
    
    return _model_cache.get(model_name, config)


# Créer l'application Celery
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


@Panel.register(name='get_worker_health', alias='health')
def get_worker_health_handler(state, **kwargs):
    """Handler pour la commande de contrôle 'get_worker_health'"""
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
    reject_on_worker_lost=True
)
def enrich_transcription_task(self, transcription_id: str, use_distributed: bool = None):
    """
    Tâche d'enrichissement exécutée par le worker.
    
    Si use_distributed=True ou si la transcription est longue, cette tâche va
    déléguer à orchestrate_distributed_enrichment au lieu de traiter directement.
    """
    api_client = get_api_client()
    
    # 1. Récupérer les informations de la transcription depuis l'API
    logger.info(f"[{transcription_id}] 📡 Fetching transcription data from API...")
    transcription = api_client.get_transcription(transcription_id)
    
    if not transcription:
        raise ValueError(f"Transcription {transcription_id} not found")
    
    # Loguer les prompts AU DÉBUT de la tâche
    logger.info(f"[{transcription_id}] 📝 PROMPTS | ========== DÉBUT TÂCHE D'ENRICHISSEMENT ==========")
    
    # Récupérer les prompts personnalisés si fournis
    enrichment_prompts = None
    enrichment_prompts_str = transcription.get('enrichment_prompts')
    if enrichment_prompts_str:
        try:
            if isinstance(enrichment_prompts_str, str):
                enrichment_prompts = json.loads(enrichment_prompts_str)
            else:
                enrichment_prompts = enrichment_prompts_str
            logger.info(f"[{transcription_id}] 📝 PROMPTS | Reçus depuis l'interface: {list(enrichment_prompts.keys())}")
            for key, value in enrichment_prompts.items():
                logger.info(f"[{transcription_id}] 📝 PROMPTS | '{key}': {value[:100]}..." if len(value) > 100 else f"[{transcription_id}] 📝 PROMPTS | '{key}': {value}")
        except Exception as e:
            logger.warning(f"[{transcription_id}] ⚠️ Failed to parse enrichment_prompts: {e}, using default")
            enrichment_prompts = None
    
    # Loguer les prompts par défaut qui seront utilisés
    from enrichment_service import DEFAULT_ENRICHMENT_PROMPTS
    if enrichment_prompts:
        # Fusionner pour voir ce qui sera utilisé
        final_prompts = DEFAULT_ENRICHMENT_PROMPTS.copy()
        final_prompts.update(enrichment_prompts)
        logger.info(f"[{transcription_id}] 📝 PROMPTS | Prompts finaux (défaut + interface): {list(final_prompts.keys())}")
        for key in final_prompts.keys():
            source = "surchargé par interface" if key in enrichment_prompts else "par défaut"
            prompt_text = enrichment_prompts.get(key, DEFAULT_ENRICHMENT_PROMPTS.get(key, ""))
            logger.info(f"[{transcription_id}] 📝 PROMPTS | '{key}' ({source}): {prompt_text[:100]}..." if len(prompt_text) > 100 else f"[{transcription_id}] 📝 PROMPTS | '{key}' ({source}): {prompt_text}")
    else:
        logger.info(f"[{transcription_id}] 📝 PROMPTS | Utilisation des prompts par défaut uniquement: {list(DEFAULT_ENRICHMENT_PROMPTS.keys())}")
        for key, value in DEFAULT_ENRICHMENT_PROMPTS.items():
            logger.info(f"[{transcription_id}] 📝 PROMPTS | '{key}' (par défaut): {value[:100]}..." if len(value) > 100 else f"[{transcription_id}] 📝 PROMPTS | '{key}' (par défaut): {value}")
    
    # Vérifier si la correction du texte est demandée
    text_correction = transcription.get('text_correction', False)
    logger.info(f"[{transcription_id}] 📝 PROMPTS | Correction du texte (text_correction): {text_correction}")
    logger.info(f"[{transcription_id}] 📝 PROMPTS | ========== FIN LOGS PROMPTS ==========")
    
    # Récupérer les segments
    segments_json = transcription.get('segments')
    if not segments_json:
        logger.warning(f"[{transcription_id}] ⚠️ No segments found, skipping enrichment")
        return {
            "status": "skipped",
            "transcription_id": transcription_id,
            "reason": "no_segments"
        }
    
    # Parser les segments
    if isinstance(segments_json, str):
        segments = json.loads(segments_json)
    else:
        segments = segments_json
    
    if not segments:
        logger.warning(f"[{transcription_id}] ⚠️ Empty segments, skipping enrichment")
        return {
            "status": "skipped",
            "transcription_id": transcription_id,
            "reason": "empty_segments"
        }
    
    # Vérifier si on doit utiliser le mode distribué
    if use_distributed is None:
        # Décider automatiquement : distribué si plus de X segments (configurable)
        # Par défaut, utiliser le mode distribué si plus de 10 segments (plus agressif)
        distribution_threshold = getattr(config, 'enrichment_distribution_threshold', 10)
        use_distributed = len(segments) > distribution_threshold
        logger.info(
            f"[{transcription_id}] 📊 DISTRIBUTION DECISION (worker) | "
            f"Segments: {len(segments)} | "
            f"Threshold: {distribution_threshold} | "
            f"Mode: {'DISTRIBUTED' if use_distributed else 'CLASSIC'} | "
            f"Reason: {'Segments > threshold' if use_distributed else 'Segments <= threshold'}"
        )
    
    # Si mode distribué, déléguer à orchestrate_distributed_enrichment
    if use_distributed:
        logger.info(
            f"[{transcription_id}] 🚀 DISTRIBUTED MODE | "
            f"Delegating to orchestrate_distributed_enrichment | "
            f"Worker: {config.instance_name}"
        )
        
        from celery import current_app as celery_current_app
        orchestrate_task = celery_current_app.send_task(
            'orchestrate_distributed_enrichment',
            args=[transcription_id],
            queue='enrichment',
            countdown=1
        )
        
        logger.info(
            f"[{transcription_id}] ✅ DISTRIBUTED MODE | "
            f"Orchestration task enqueued: {orchestrate_task.id}"
        )
        
        return {
            "transcription_id": transcription_id,
            "task_id": self.request.id,
            "orchestration_task_id": orchestrate_task.id,
            "status": "queued_distributed",
            "mode": "distributed"
        }
    
    # MODE CLASSIQUE : Traitement direct
    logger.info(
        f"[{transcription_id}] 🎯 CLASSIC MODE STARTED | "
        f"Worker: {config.instance_name} | "
        f"Task ID: {self.request.id} | "
        f"Segments: {len(segments)}"
    )
    start_time = time.time()
    
    try:
        # Mettre à jour le statut
        api_client.update_transcription(transcription_id, {
            "enrichment_status": "processing",
            "enrichment_worker_id": config.instance_name
        })
        logger.info(f"[{transcription_id}] ⚙️ Status updated to 'processing'")
        
        # Obtenir le service d'enrichissement avec cache
        llm_model = transcription.get('llm_model', config.llm_model)
        logger.info(f"[{transcription_id}] 🎤 Getting enrichment service with model: {llm_model} (cached)")
        enrichment_service = get_llm_service(model_name=llm_model)
        
        # Vérifier si la correction du texte est demandée
        text_correction = transcription.get('text_correction', False)
        original_text = " ".join(seg.get('text', '') for seg in segments)
        
        # Corriger le texte UNIQUEMENT si text_correction=true
        if text_correction:
            logger.info(f"[{transcription_id}] ✏️ Text correction enabled - Starting text correction with LLM...")
            corrected_segments = enrichment_service.enrich_segments(segments, custom_prompts=None)  # Correction du texte
            corrected_text = " ".join(seg.get('enriched_text', seg.get('text', '')) for seg in corrected_segments)
        else:
            logger.info(f"[{transcription_id}] ℹ️ Text correction disabled - Using original text")
            corrected_segments = segments  # Pas de correction, garder les segments originaux
            corrected_text = original_text
        
        # Construire le texte pour les métadonnées (utiliser le texte corrigé si disponible, sinon l'original)
        text_for_metadata = corrected_text if text_correction else original_text
        
        # Générer les métadonnées (titre, résumé, score, bullet points) - C'EST L'ENRICHISSEMENT DE BASE
        # Les métadonnées sont TOUJOURS générées si enrichment_requested=true (ce qui est le cas ici)
        enrichment_requested = transcription.get('enrichment_requested', False)
        if not enrichment_requested:
            logger.warning(f"[{transcription_id}] ⚠️ Enrichment not requested, skipping metadata generation")
            metadata = {}
            processing_time = round(time.time() - start_time, 2)
        else:
            logger.info(f"[{transcription_id}] 📊 Generating metadata (title, summary, satisfaction, bullet_points) - ENRICHISSEMENT DE BASE...")
            metadata_start_time = time.time()
            # Obtenir les prompts finaux
            from enrichment_service import DEFAULT_ENRICHMENT_PROMPTS
            final_prompts = DEFAULT_ENRICHMENT_PROMPTS.copy()
            if enrichment_prompts:
                final_prompts.update(enrichment_prompts)
            
            metadata = {}
            
            # Générer le titre avec mesure du temps
            title_time = 0.0
            try:
                logger.info(f"[{transcription_id}] 📊 Generating title...")
                title_start = time.time()
                title_response = enrichment_service.generate_metadata(text_for_metadata, "title", final_prompts, max_tokens=50)
                title_time = round(time.time() - title_start, 2)
                metadata['title'] = title_response.strip() if title_response else None
                if metadata['title']:
                    logger.info(f"[{transcription_id}] ✅ Title generated: {metadata['title'][:50]}...")
                else:
                    logger.warning(f"[{transcription_id}] ⚠️ Title generation returned empty string")
            except Exception as e:
                logger.warning(f"[{transcription_id}] ⚠️ Failed to generate title: {e}", exc_info=True)
                metadata['title'] = None
            
            # Générer le résumé avec mesure du temps
            summary_time = 0.0
            try:
                logger.info(f"[{transcription_id}] 📊 Generating summary...")
                summary_start = time.time()
                summary_response = enrichment_service.generate_metadata(text_for_metadata, "summary", final_prompts, max_tokens=150)
                summary_time = round(time.time() - summary_start, 2)
                metadata['summary'] = summary_response.strip() if summary_response else None
                if metadata['summary']:
                    logger.info(f"[{transcription_id}] ✅ Summary generated: {metadata['summary'][:100]}...")
                else:
                    logger.warning(f"[{transcription_id}] ⚠️ Summary generation returned empty string")
            except Exception as e:
                logger.warning(f"[{transcription_id}] ⚠️ Failed to generate summary: {e}", exc_info=True)
                metadata['summary'] = None
            
            # Générer le score de satisfaction avec mesure du temps
            satisfaction_time = 0.0
            try:
                logger.info(f"[{transcription_id}] 📊 Generating satisfaction score...")
                satisfaction_start = time.time()
                satisfaction_response = enrichment_service.generate_metadata(text_for_metadata, "satisfaction", final_prompts, max_tokens=100)
                satisfaction_time = round(time.time() - satisfaction_start, 2)
                # Parser le JSON
                import json as json_lib
                if satisfaction_response and satisfaction_response.strip():
                    try:
                        metadata['satisfaction'] = json_lib.loads(satisfaction_response.strip())
                    except Exception as json_error:
                        logger.warning(f"[{transcription_id}] ⚠️ Failed to parse satisfaction JSON: {json_error}, using fallback")
                        metadata['satisfaction'] = {"score": None, "justification": satisfaction_response.strip()}
                else:
                    logger.warning(f"[{transcription_id}] ⚠️ Satisfaction generation returned empty string")
                    metadata['satisfaction'] = None
                if metadata['satisfaction']:
                    logger.info(f"[{transcription_id}] ✅ Satisfaction score generated: {metadata['satisfaction']}")
            except Exception as e:
                logger.warning(f"[{transcription_id}] ⚠️ Failed to generate satisfaction score: {e}", exc_info=True)
                metadata['satisfaction'] = None
            
            # Générer les bullet points avec mesure du temps
            bullet_points_time = 0.0
            try:
                logger.info(f"[{transcription_id}] 📊 Generating bullet points...")
                bullet_start = time.time()
                bullet_response = enrichment_service.generate_metadata(text_for_metadata, "bullet_points", final_prompts, max_tokens=200)
                bullet_points_time = round(time.time() - bullet_start, 2)
                # Parser le JSON
                import json as json_lib
                if bullet_response and bullet_response.strip():
                    try:
                        metadata['bullet_points'] = json_lib.loads(bullet_response.strip())
                    except Exception as json_error:
                        logger.warning(f"[{transcription_id}] ⚠️ Failed to parse bullet points JSON: {json_error}, using fallback")
                        metadata['bullet_points'] = {"points": [bullet_response.strip()]}
                else:
                    logger.warning(f"[{transcription_id}] ⚠️ Bullet points generation returned empty string")
                    metadata['bullet_points'] = None
                if metadata['bullet_points']:
                    logger.info(f"[{transcription_id}] ✅ Bullet points generated: {len(metadata['bullet_points'].get('points', []))} points")
            except Exception as e:
                logger.warning(f"[{transcription_id}] ⚠️ Failed to generate bullet points: {e}", exc_info=True)
                metadata['bullet_points'] = None
            
            metadata_time = round(time.time() - metadata_start_time, 2)
            processing_time = round(time.time() - start_time, 2)
            logger.info(f"[{transcription_id}] ✅ Metadata generation completed in {metadata_time}s")
        
        # Construire l'objet enhanced_data avec les métadonnées (enrichissement de base)
        # Toujours sauvegarder, même si toutes les métadonnées sont None (pour diagnostic)
        if enrichment_requested:
            enhanced_data = {
                "metadata": metadata
            }
            logger.info(f"[{transcription_id}] 📊 Metadata summary: title={metadata.get('title') is not None}, summary={metadata.get('summary') is not None}, satisfaction={metadata.get('satisfaction') is not None}, bullet_points={metadata.get('bullet_points') is not None}")
            
            # Construire enrichment_data au format de enrich_transcription (avec les temps individuels)
            satisfaction_score = metadata.get('satisfaction', {}).get('score') if isinstance(metadata.get('satisfaction'), dict) else None
            bullet_points_list = metadata.get('bullet_points', {}).get('points', []) if isinstance(metadata.get('bullet_points'), dict) else []
            
            enrichment_data = {
                "title": metadata.get('title'),
                "summary": metadata.get('summary'),
                "satisfaction_score": satisfaction_score,
                "bullet_points": bullet_points_list[:4] if bullet_points_list else [],  # Limiter à 4 points maximum
                "timing": {
                    "title_time": title_time,
                    "summary_time": summary_time,
                    "satisfaction_time": satisfaction_time,
                    "bullet_points_time": bullet_points_time,
                    "total_time": metadata_time
                }
            }
        else:
            enhanced_data = None
            enrichment_data = None
            logger.warning(f"[{transcription_id}] ⚠️ Enrichment not requested, enhanced_text will be null")
        
        # Mettre à jour avec les résultats
        logger.info(f"[{transcription_id}] 💾 Saving results to API...")
        update_data = {
            "status": "done",  # Mettre à jour le statut principal (comme transcription)
            "enrichment_status": "done",
            "enriched_segments": json.dumps(corrected_segments),
            "enrichment_processing_time": processing_time
        }
        
        # Ajouter enhanced_text si enrichment_requested=true (même si toutes les métadonnées sont None)
        if enhanced_data:
            update_data["enhanced_text"] = json.dumps(enhanced_data, ensure_ascii=False)
        
        # Ajouter enrichment_data au format de enrich_transcription
        if enrichment_data:
            update_data["enrichment_data"] = json.dumps(enrichment_data, ensure_ascii=False)
        
        # Ajouter enriched_text si text_correction=true
        if text_correction:
            update_data["enriched_text"] = corrected_text
        logger.info(f"[{transcription_id}] 📤 API Update payload: {json.dumps({k: v if k != 'enriched_segments' else f'<{len(corrected_segments)} segments>' for k, v in update_data.items()})}")
        
        response = api_client.update_transcription(transcription_id, update_data)
        logger.info(f"[{transcription_id}] ✅ API Update response received: status={response.get('status')}, enrichment_status={response.get('enrichment_status')}")
        logger.info(f"[{transcription_id}] 💾 Results saved to API successfully")
        
        return {
            "status": "success",
            "transcription_id": transcription_id,
            "processing_time": processing_time,
            "segments_count": len(corrected_segments),
            "mode": "classic"
        }
        
    except Exception as e:
        logger.error(f"[{transcription_id}] ❌ Error: {e}", exc_info=True)
        
        # Mettre à jour le statut à "error"
        try:
            api_client_on_error = get_api_client()
            api_client_on_error.update_transcription(transcription_id, {
                "enrichment_status": "error",
                "enrichment_error_message": str(e)
            })
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


@celery_app.task(
    bind=True,
    name='orchestrate_distributed_enrichment',
    max_retries=2,
    default_retry_delay=30,
    acks_late=True
)
def orchestrate_distributed_enrichment_task(self, transcription_id: str):
    """
    Orchestre l'enrichissement distribué : découpe les segments en chunks et crée les tâches.
    
    Args:
        transcription_id: ID de la transcription
    """
    logger.info(
        f"[{transcription_id}] 🎼 DISTRIBUTED ORCHESTRATION STARTED | "
        f"Worker: {config.instance_name} | "
        f"Task ID: {self.request.id}"
    )
    
    try:
        api_client = get_api_client()
        transcription = api_client.get_transcription(transcription_id)
        
        if not transcription:
            raise ValueError(f"Transcription {transcription_id} not found")
        
        # Récupérer les segments
        segments_json = transcription.get('segments')
        if not segments_json:
            raise ValueError(f"No segments found for transcription {transcription_id}")
        
        if isinstance(segments_json, str):
            segments = json.loads(segments_json)
        else:
            segments = segments_json
        
        if not segments:
            raise ValueError(f"Empty segments for transcription {transcription_id}")
        
        # Mettre à jour le statut
        api_client.update_transcription(transcription_id, {
            "enrichment_status": "processing",
            "enrichment_worker_id": f"{config.instance_name}-orchestrator"
        })
        
        # 1. Découper en chunks intelligents
        logger.info(
            f"[{transcription_id}] ✂️ DISTRIBUTED ORCHESTRATION | Step 1/3: Splitting segments into chunks | "
            f"Total segments: {len(segments)}"
        )
        
        splitter = ChunkSplitter(
            max_chunk_size=getattr(config, 'max_chunk_size', 500),
            max_duration=60.0
        )
        
        # Détecter si la diarisation est disponible
        use_diarization = any(seg.get('speaker') for seg in segments)
        strategy = 'speaker' if use_diarization else 'size'
        
        chunks = splitter.split(segments, strategy=strategy, use_diarization=use_diarization)
        num_chunks = len(chunks)
        
        logger.info(
            f"[{transcription_id}] ✅ DISTRIBUTED ORCHESTRATION | Step 1/3: Chunking complete | "
            f"Chunks created: {num_chunks} | "
            f"Will be distributed across available workers"
        )
        
        if num_chunks == 0:
            raise ValueError("No chunks created")
        
        # 2. Stocker les métadonnées dans Redis
        redis_manager = get_redis_manager()
        orchestration_start_time = time.time()
        
        llm_model = transcription.get('llm_model', config.llm_model)
        
        # Loguer les prompts AU DÉBUT de l'orchestration
        logger.info(f"[{transcription_id}] 📝 PROMPTS | ========== DÉBUT ORCHESTRATION DISTRIBUÉE ==========")
        
        # Récupérer les prompts personnalisés si fournis
        enrichment_prompts = None
        enrichment_prompts_str = transcription.get('enrichment_prompts')
        if enrichment_prompts_str:
            try:
                if isinstance(enrichment_prompts_str, str):
                    enrichment_prompts = json.loads(enrichment_prompts_str)
                else:
                    enrichment_prompts = enrichment_prompts_str
                logger.info(f"[{transcription_id}] 📝 PROMPTS | Reçus depuis l'interface: {list(enrichment_prompts.keys())}")
                for key, value in enrichment_prompts.items():
                    logger.info(f"[{transcription_id}] 📝 PROMPTS | '{key}': {value[:100]}..." if len(value) > 100 else f"[{transcription_id}] 📝 PROMPTS | '{key}': {value}")
            except Exception as e:
                logger.warning(f"[{transcription_id}] ⚠️ Failed to parse enrichment_prompts: {e}, using default")
                enrichment_prompts = None
        
        # Loguer les prompts par défaut qui seront utilisés
        from enrichment_service import DEFAULT_ENRICHMENT_PROMPTS
        if enrichment_prompts:
            # Fusionner pour voir ce qui sera utilisé
            final_prompts = DEFAULT_ENRICHMENT_PROMPTS.copy()
            final_prompts.update(enrichment_prompts)
            logger.info(f"[{transcription_id}] 📝 PROMPTS | Prompts finaux (défaut + interface): {list(final_prompts.keys())}")
            for key in final_prompts.keys():
                source = "surchargé par interface" if key in enrichment_prompts else "par défaut"
                prompt_text = enrichment_prompts.get(key, DEFAULT_ENRICHMENT_PROMPTS.get(key, ""))
                logger.info(f"[{transcription_id}] 📝 PROMPTS | '{key}' ({source}): {prompt_text[:100]}..." if len(prompt_text) > 100 else f"[{transcription_id}] 📝 PROMPTS | '{key}' ({source}): {prompt_text}")
        else:
            logger.info(f"[{transcription_id}] 📝 PROMPTS | Utilisation des prompts par défaut uniquement: {list(DEFAULT_ENRICHMENT_PROMPTS.keys())}")
            for key, value in DEFAULT_ENRICHMENT_PROMPTS.items():
                logger.info(f"[{transcription_id}] 📝 PROMPTS | '{key}' (par défaut): {value[:100]}..." if len(value) > 100 else f"[{transcription_id}] 📝 PROMPTS | '{key}' (par défaut): {value}")
        
        # Vérifier si la correction du texte est demandée
        text_correction = transcription.get('text_correction', False)
        logger.info(f"[{transcription_id}] 📝 PROMPTS | Correction du texte (text_correction): {text_correction}")
        logger.info(f"[{transcription_id}] 📝 PROMPTS | ========== FIN LOGS PROMPTS ==========")
        
        chunks_metadata = {
            "transcription_id": transcription_id,
            "total_chunks": num_chunks,
            "completed_chunks": 0,
            "chunks": [chunk for chunk in chunks],  # Stocker les chunks complets
            "llm_model": llm_model,
            "enrichment_prompts": enrichment_prompts,  # Stocker les prompts personnalisés
            "text_correction": text_correction,  # Stocker le flag text_correction
            "orchestration_start_time": orchestration_start_time,
            "strategy": strategy
        }
        
        ttl = getattr(config, 'redis_enrichment_ttl', 3600)
        redis_manager.store_metadata(transcription_id, chunks_metadata, ttl)
        redis_manager.reset_completed_count(transcription_id)
        
        # 3. Créer une tâche pour chaque chunk
        logger.info(
            f"[{transcription_id}] 📤 DISTRIBUTED ORCHESTRATION | Step 2/3: Creating enrichment chunk tasks | "
            f"Total chunks: {num_chunks} | "
            f"Queue: enrichment | "
            f"Tasks will be distributed automatically by Celery"
        )
        chunk_tasks = []
        from celery import current_app as celery_current_app
        
        for i, chunk in enumerate(chunks):
            chunk_task = celery_current_app.send_task(
                'enrich_chunk',
                args=[transcription_id, i, num_chunks],
                queue='enrichment'
            )
            chunk_tasks.append(chunk_task.id)
            logger.info(
                f"[{transcription_id}] 📤 DISTRIBUTED ORCHESTRATION | Chunk {i+1}/{num_chunks} enqueued | "
                f"Task ID: {chunk_task.id} | "
                f"Chunk size: {sum(len(seg.get('text', '')) for seg in chunk)} chars | "
                f"Waiting for available worker..."
            )
        
        # 4. Stocker les IDs des tâches
        redis_client = get_redis_client()
        tasks_key = f"enrichment:{transcription_id}:chunk_tasks"
        redis_client.setex(tasks_key, 3600, json.dumps(chunk_tasks))
        
        logger.info(
            f"[{transcription_id}] ✅ DISTRIBUTED ORCHESTRATION | Step 3/3: All tasks created | "
            f"Enrichment tasks: {num_chunks} | "
            f"Next: Workers will process chunks in parallel"
        )
        
        return {
            "status": "orchestrated",
            "transcription_id": transcription_id,
            "num_chunks": num_chunks,
            "chunk_tasks": chunk_tasks
        }
        
    except Exception as e:
        logger.error(f"[{transcription_id}] ❌ Orchestration error: {e}", exc_info=True)
        try:
            api_client = get_api_client()
            api_client.update_transcription(transcription_id, {
                "enrichment_status": "error",
                "enrichment_error_message": f"Orchestration failed: {str(e)}"
            })
        except:
            pass
        raise


@celery_app.task(
    bind=True,
    name='enrich_chunk',
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
    reject_on_worker_lost=True
)
def enrich_chunk_task(self, transcription_id: str, chunk_index: int, total_chunks: int):
    """
    Enrichit un seul chunk de segments.
    
    Args:
        transcription_id: ID de la transcription parente
        chunk_index: Index du chunk (0-based)
        total_chunks: Nombre total de chunks
    """
    logger.info(
        f"[{transcription_id}] 🎯 DISTRIBUTED CHUNK STARTED | "
        f"Chunk: {chunk_index+1}/{total_chunks} | "
        f"Worker: {config.instance_name} | "
        f"Task ID: {self.request.id}"
    )
    start_time = time.time()
    
    try:
        # Récupérer les métadonnées depuis Redis
        redis_manager = get_redis_manager()
        metadata = redis_manager.get_metadata(transcription_id)
        
        if not metadata:
            raise ValueError(f"Metadata not found for transcription {transcription_id}")
        
        # Récupérer le chunk
        chunks = metadata.get('chunks', [])
        if chunk_index >= len(chunks):
            raise ValueError(f"Chunk {chunk_index} not found in metadata")
        
        chunk = chunks[chunk_index]
        llm_model = metadata.get('llm_model', config.llm_model)
        
        logger.info(
            f"[{transcription_id}] ⚙️ DISTRIBUTED CHUNK | Worker {config.instance_name} processing | "
            f"Chunk: {chunk_index+1}/{total_chunks} | "
            f"Model: {llm_model} | "
            f"Segments in chunk: {len(chunk)}"
        )
        
        # Vérifier si la correction du texte est demandée
        text_correction = metadata.get('text_correction', False)
        
        # Corriger le texte UNIQUEMENT si text_correction=true
        if text_correction:
            logger.info(f"[{transcription_id}] ✏️ Text correction enabled for chunk {chunk_index+1}...")
            enrichment_service = get_llm_service(model_name=llm_model)
            enriched_chunk = enrichment_service.enrich_segments(chunk, custom_prompts=None)  # Correction du texte
        else:
            logger.info(f"[{transcription_id}] ℹ️ Text correction disabled for chunk {chunk_index+1} - Using original segments")
            enriched_chunk = chunk  # Pas de correction, garder les segments originaux
        
        processing_time = round(time.time() - start_time, 2)
        
        # Stocker le résultat dans Redis
        result = {
            "chunk_index": chunk_index,
            "enriched_segments": enriched_chunk,
            "processing_time": processing_time
        }
        
        ttl = getattr(config, 'redis_enrichment_ttl', 3600)
        redis_manager.store_chunk_result(transcription_id, chunk_index, result, ttl)
        completed_count = redis_manager.increment_completed_count(transcription_id)
        
        logger.info(
            f"[{transcription_id}] ✅ DISTRIBUTED CHUNK COMPLETED | "
            f"Chunk: {chunk_index+1}/{total_chunks} | "
            f"Worker: {config.instance_name} | "
            f"Processing time: {processing_time}s | "
            f"Progress: {completed_count}/{total_chunks} chunks done ({100*completed_count/total_chunks:.1f}%)"
        )
        
        # Si tous les chunks sont terminés, déclencher l'agrégation
        if completed_count >= total_chunks:
            if redis_manager.acquire_aggregation_lock(transcription_id):
                logger.info(
                    f"[{transcription_id}] 🎉 DISTRIBUTED MODE | All chunks completed | "
                    f"Total: {total_chunks} chunks | "
                    f"All workers finished | "
                    f"Triggering aggregation... (lock acquired by {config.instance_name})"
                )
                from celery import current_app as celery_current_app
                aggregate_task = celery_current_app.send_task(
                    'aggregate_enrichment_chunks',
                    args=[transcription_id],
                    queue='enrichment',
                    countdown=1
                )
                logger.info(
                    f"[{transcription_id}] ✅ DISTRIBUTED MODE | Aggregation task enqueued | "
                    f"Task ID: {aggregate_task.id} | "
                    f"Next: Reassembling all chunks"
                )
            else:
                logger.info(
                    f"[{transcription_id}] ℹ️ DISTRIBUTED MODE | All chunks completed but aggregation already triggered by another worker"
                )
        
        return {
            "status": "success",
            "chunk_index": chunk_index,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"[{transcription_id}] ❌ Chunk {chunk_index+1} error: {e}", exc_info=True)
        raise


@celery_app.task(
    bind=True,
    name='aggregate_enrichment_chunks',
    max_retries=2,
    default_retry_delay=30,
    acks_late=True
)
def aggregate_enrichment_chunks_task(self, transcription_id: str):
    """
    Réassemble les chunks enrichis en résultat final.
    
    Args:
        transcription_id: ID de la transcription
    """
    logger.info(
        f"[{transcription_id}] 🔗 DISTRIBUTED AGGREGATION STARTED | "
        f"Worker: {config.instance_name} | "
        f"Task ID: {self.request.id} | "
        f"Will reassemble all completed chunks"
    )
    start_time = time.time()
    
    try:
        redis_manager = get_redis_manager()
        metadata = redis_manager.get_metadata(transcription_id)
        
        if not metadata:
            raise ValueError(f"Metadata not found for transcription {transcription_id}")
        
        total_chunks = metadata['total_chunks']
        
        # Récupérer tous les résultats des chunks
        logger.info(
            f"[{transcription_id}] 📥 DISTRIBUTED AGGREGATION | Step 1/2: Collecting chunk results | "
            f"Expected chunks: {total_chunks}"
        )
        all_enriched_segments = []
        max_chunk_time = 0.0
        
        for i in range(total_chunks):
            result = redis_manager.get_chunk_result(transcription_id, i)
            if not result:
                raise ValueError(f"Result not found for chunk {i} of transcription {transcription_id}")
            
            all_enriched_segments.extend(result['enriched_segments'])
            chunk_time = result.get('processing_time', 0.0)
            max_chunk_time = max(max_chunk_time, chunk_time)
        
        # Trier les segments par timestamp
        all_enriched_segments.sort(key=lambda x: x.get('start', 0.0))
        
        # Calculer le temps réel écoulé
        orchestration_start_time = metadata.get('orchestration_start_time')
        if orchestration_start_time:
            real_elapsed_time = round(time.time() - orchestration_start_time, 2)
        else:
            real_elapsed_time = max_chunk_time
        
        logger.info(
            f"[{transcription_id}] ✅ DISTRIBUTED AGGREGATION | Step 1/2: All results collected | "
            f"Segments: {len(all_enriched_segments)} | "
            f"Max chunk time: {max_chunk_time:.1f}s (parallel) | "
            f"Real elapsed time: {real_elapsed_time:.1f}s"
        )
        
        # Construire le texte complet (corrigé si text_correction=true, sinon original)
        text_correction = metadata.get('text_correction', False)
        if text_correction:
            # Utiliser le texte corrigé si disponible
            enriched_text = " ".join(
                seg.get('enriched_text', seg.get('text', '')) 
                for seg in all_enriched_segments 
                if seg.get('enriched_text', seg.get('text', '')).strip()
            )
        else:
            # Utiliser le texte original
            enriched_text = " ".join(
                seg.get('text', '') 
                for seg in all_enriched_segments 
                if seg.get('text', '').strip()
            )
        
        # Générer les métadonnées (titre, résumé, score, bullet points) - C'EST L'ENRICHISSEMENT DE BASE
        # Les métadonnées sont TOUJOURS générées si enrichment_requested=true
        logger.info(f"[{transcription_id}] 📊 Generating metadata (title, summary, satisfaction, bullet_points) - ENRICHISSEMENT DE BASE...")
        metadata_start_time = time.time()
        # Obtenir les prompts finaux depuis les métadonnées
        from enrichment_service import DEFAULT_ENRICHMENT_PROMPTS
        enrichment_prompts = metadata.get('enrichment_prompts')
        final_prompts = DEFAULT_ENRICHMENT_PROMPTS.copy()
        if enrichment_prompts:
            final_prompts.update(enrichment_prompts)
        
        # Obtenir le modèle LLM
        llm_model = metadata.get('llm_model', config.llm_model)
        enrichment_service = get_llm_service(model_name=llm_model)
        
        metadata_result = {}
        
        # Générer le titre avec mesure du temps
        title_time = 0.0
        try:
            logger.info(f"[{transcription_id}] 📊 Generating title...")
            title_start = time.time()
            title_response = enrichment_service.generate_metadata(enriched_text, "title", final_prompts, max_tokens=50)
            title_time = round(time.time() - title_start, 2)
            metadata_result['title'] = title_response.strip() if title_response else None
            if metadata_result['title']:
                logger.info(f"[{transcription_id}] ✅ Title generated: {metadata_result['title'][:50]}...")
            else:
                logger.warning(f"[{transcription_id}] ⚠️ Title generation returned empty string")
        except Exception as e:
            logger.warning(f"[{transcription_id}] ⚠️ Failed to generate title: {e}", exc_info=True)
            metadata_result['title'] = None
        
        # Générer le résumé avec mesure du temps
        summary_time = 0.0
        try:
            logger.info(f"[{transcription_id}] 📊 Generating summary...")
            summary_start = time.time()
            summary_response = enrichment_service.generate_metadata(enriched_text, "summary", final_prompts, max_tokens=150)
            summary_time = round(time.time() - summary_start, 2)
            metadata_result['summary'] = summary_response.strip() if summary_response else None
            if metadata_result['summary']:
                logger.info(f"[{transcription_id}] ✅ Summary generated: {metadata_result['summary'][:100]}...")
            else:
                logger.warning(f"[{transcription_id}] ⚠️ Summary generation returned empty string")
        except Exception as e:
            logger.warning(f"[{transcription_id}] ⚠️ Failed to generate summary: {e}", exc_info=True)
            metadata_result['summary'] = None
        
        # Générer le score de satisfaction avec mesure du temps
        satisfaction_time = 0.0
        try:
            logger.info(f"[{transcription_id}] 📊 Generating satisfaction score...")
            satisfaction_start = time.time()
            satisfaction_response = enrichment_service.generate_metadata(enriched_text, "satisfaction", final_prompts, max_tokens=100)
            satisfaction_time = round(time.time() - satisfaction_start, 2)
            # Parser le JSON
            if satisfaction_response and satisfaction_response.strip():
                try:
                    metadata_result['satisfaction'] = json.loads(satisfaction_response.strip())
                except Exception as json_error:
                    logger.warning(f"[{transcription_id}] ⚠️ Failed to parse satisfaction JSON: {json_error}, using fallback")
                    metadata_result['satisfaction'] = {"score": None, "justification": satisfaction_response.strip()}
            else:
                logger.warning(f"[{transcription_id}] ⚠️ Satisfaction generation returned empty string")
                metadata_result['satisfaction'] = None
            if metadata_result['satisfaction']:
                logger.info(f"[{transcription_id}] ✅ Satisfaction score generated: {metadata_result['satisfaction']}")
        except Exception as e:
            logger.warning(f"[{transcription_id}] ⚠️ Failed to generate satisfaction score: {e}", exc_info=True)
            metadata_result['satisfaction'] = None
        
        # Générer les bullet points avec mesure du temps
        bullet_points_time = 0.0
        try:
            logger.info(f"[{transcription_id}] 📊 Generating bullet points...")
            bullet_start = time.time()
            bullet_response = enrichment_service.generate_metadata(enriched_text, "bullet_points", final_prompts, max_tokens=200)
            bullet_points_time = round(time.time() - bullet_start, 2)
            # Parser le JSON
            if bullet_response and bullet_response.strip():
                try:
                    metadata_result['bullet_points'] = json.loads(bullet_response.strip())
                except Exception as json_error:
                    logger.warning(f"[{transcription_id}] ⚠️ Failed to parse bullet points JSON: {json_error}, using fallback")
                    metadata_result['bullet_points'] = {"points": [bullet_response.strip()]}
            else:
                logger.warning(f"[{transcription_id}] ⚠️ Bullet points generation returned empty string")
                metadata_result['bullet_points'] = None
            if metadata_result['bullet_points']:
                logger.info(f"[{transcription_id}] ✅ Bullet points generated: {len(metadata_result['bullet_points'].get('points', []))} points")
        except Exception as e:
            logger.warning(f"[{transcription_id}] ⚠️ Failed to generate bullet points: {e}", exc_info=True)
            metadata_result['bullet_points'] = None
        
        metadata_time = round(time.time() - metadata_start_time, 2)
        logger.info(f"[{transcription_id}] ✅ Metadata generation completed in {metadata_time}s")
        
        # Construire l'objet enhanced_data avec les métadonnées (enrichissement de base)
        # Toujours sauvegarder, même si toutes les métadonnées sont None (pour diagnostic)
        enhanced_data = {
            "metadata": metadata_result
        }
        logger.info(f"[{transcription_id}] 📊 Metadata summary: title={metadata_result.get('title') is not None}, summary={metadata_result.get('summary') is not None}, satisfaction={metadata_result.get('satisfaction') is not None}, bullet_points={metadata_result.get('bullet_points') is not None}")
        
        # Construire enrichment_data au format de enrich_transcription (avec les temps individuels)
        satisfaction_score = metadata_result.get('satisfaction', {}).get('score') if isinstance(metadata_result.get('satisfaction'), dict) else None
        bullet_points_list = metadata_result.get('bullet_points', {}).get('points', []) if isinstance(metadata_result.get('bullet_points'), dict) else []
        
        enrichment_data = {
            "title": metadata_result.get('title'),
            "summary": metadata_result.get('summary'),
            "satisfaction_score": satisfaction_score,
            "bullet_points": bullet_points_list[:4] if bullet_points_list else [],  # Limiter à 4 points maximum
            "timing": {
                "title_time": title_time,
                "summary_time": summary_time,
                "satisfaction_time": satisfaction_time,
                "bullet_points_time": bullet_points_time,
                "total_time": metadata_time
            }
        }
        
        # Sauvegarder le résultat final
        api_client = get_api_client()
        aggregation_time = round(time.time() - start_time, 2)
        
        if orchestration_start_time:
            total_processing_time = round(time.time() - orchestration_start_time, 2)
        else:
            total_processing_time = round(max_chunk_time + aggregation_time, 2)
        
        update_data = {
            "status": "done",  # Mettre à jour le statut principal (comme transcription)
            "enrichment_status": "done",
            "enriched_segments": json.dumps(all_enriched_segments),
            "enrichment_processing_time": total_processing_time,
            "enhanced_text": json.dumps(enhanced_data, ensure_ascii=False),  # Toujours sauvegarder les métadonnées
            "enrichment_data": json.dumps(enrichment_data, ensure_ascii=False)  # Format enrich_transcription avec les temps
        }
        
        # Ajouter enriched_text si text_correction=true
        if text_correction:
            corrected_text = " ".join(
                seg.get('enriched_text', seg.get('text', '')) 
                for seg in all_enriched_segments 
                if seg.get('enriched_text', seg.get('text', '')).strip()
            )
            update_data["enriched_text"] = corrected_text
        logger.info(f"[{transcription_id}] 📤 DISTRIBUTED AGGREGATION | API Update payload: {json.dumps({k: v if k != 'enriched_segments' else f'<{len(all_enriched_segments)} segments>' for k, v in update_data.items()})}")
        
        response = api_client.update_transcription(transcription_id, update_data)
        logger.info(f"[{transcription_id}] ✅ DISTRIBUTED AGGREGATION | API Update response: status={response.get('status')}, enrichment_status={response.get('enrichment_status')}")
        
        logger.info(
            f"[{transcription_id}] ✅ DISTRIBUTED AGGREGATION | Step 2/2: Aggregation completed | "
            f"Total segments: {len(all_enriched_segments)} | "
            f"Real processing time: {total_processing_time:.1f}s (from orchestration start) | "
            f"Max chunk time: {max_chunk_time:.1f}s | "
            f"Aggregation time: {aggregation_time:.1f}s | "
            f"Result saved to database"
        )
        
        # Nettoyer les données Redis
        try:
            redis_manager.cleanup(transcription_id, total_chunks)
        except Exception as cleanup_error:
            logger.warning(f"[{transcription_id}] ⚠️ Cleanup error: {cleanup_error}")
        
        return {
            "status": "success",
            "transcription_id": transcription_id,
            "segments_count": len(all_enriched_segments),
            "total_processing_time": total_processing_time
        }
        
    except Exception as e:
        logger.error(f"[{transcription_id}] ❌ Aggregation error: {e}", exc_info=True)
        try:
            api_client = get_api_client()
            api_client.update_transcription(transcription_id, {
                "enrichment_status": "error",
                "enrichment_error_message": f"Aggregation failed: {str(e)}"
            })
        except:
            pass
        raise


if __name__ == "__main__":
    logger.info(f"🚀 Starting Celery enrichment worker: {config.instance_name}")
    celery_app.worker_main([
        'worker',
        f'--loglevel={config.log_level.lower()}',
        f'--concurrency={config.max_workers}',
        f'--hostname={config.instance_name}@%h',
        '--without-gossip',
        '--without-mingle',
        '-Q', 'enrichment'
    ])

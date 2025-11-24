"""
logging_config.py
Configuration centralisée du logging pour Vocalyx Enrichment.
Uniformise le format des logs pour tous les composants.
"""

import logging
import sys
from pathlib import Path

# Format uniforme pour tous les logs
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Configure le logging pour toute l'application.
    
    Args:
        log_level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Chemin optionnel vers un fichier de log
    """
    
    # Convertir le niveau de log
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configuration de base
    handlers = []
    
    # Handler pour stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(
        logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    )
    handlers.append(console_handler)
    
    # Handler optionnel pour fichier
    if log_file:
        # Créer le répertoire si nécessaire
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        )
        handlers.append(file_handler)
    
    # Configuration globale
    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True  # Override les configurations existantes
    )
    
    # Configurer les loggers spécifiques
    loggers_to_configure = [
        "vocalyx",
        "vocalyx.enrichment",
        "httpx",
        "transformers",
        "torch",
        "celery",
        "celery.task",
        "celery.worker",
        "celery.app",
    ]
    
    for logger_name in loggers_to_configure:
        log = logging.getLogger(logger_name)
        log.setLevel(numeric_level)
        log.handlers.clear()
        for handler in handlers:
            log.addHandler(handler)
        log.propagate = False
        
    # Réduire le verbosité des bibliothèques externes
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    # Filtrer le warning de sécurité Celery
    import warnings
    try:
        from celery.platforms import SecurityWarning
        warnings.filterwarnings('ignore', category=SecurityWarning, module='celery.platforms')
    except ImportError:
        pass
    warnings.filterwarnings('ignore', category=UserWarning, module='celery.platforms')
    
    logger = logging.getLogger("vocalyx.enrichment")
    logger.info("✅ Logging configuré avec succès")
    
    return logger


# Custom formatter avec couleurs (optionnel)
class ColoredFormatter(logging.Formatter):
    """Formatter avec couleurs pour le terminal"""
    
    # Codes ANSI
    COLORS = {
        'DEBUG': '\033[0;36m',    # Cyan
        'INFO': '\033[0;32m',     # Vert
        'WARNING': '\033[0;33m',  # Jaune
        'ERROR': '\033[0;31m',    # Rouge
        'CRITICAL': '\033[1;31m', # Rouge gras
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Ajouter la couleur au niveau de log
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        return super().format(record)


def setup_colored_logging(log_level: str = "INFO", log_file: str = None):
    """
    Configure le logging avec couleurs pour le terminal.
    Identique à setup_logging mais avec couleurs.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = []
    
    # Handler console avec couleurs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(
        ColoredFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    )
    handlers.append(console_handler)
    
    # Handler fichier sans couleurs
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        )
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True
    )
    
    # Configurer les loggers spécifiques
    for logger_name in ["vocalyx", "vocalyx.enrichment", "httpx", "transformers", "torch",
                       "celery", "celery.task", "celery.worker", "celery.app"]:
        log = logging.getLogger(logger_name)
        log.setLevel(numeric_level)
        log.handlers.clear()
        for handler in handlers:
            log.addHandler(handler)
        log.propagate = False
        
    # Réduire le verbosité des bibliothèques externes
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    # Filtrer le warning de sécurité Celery
    import warnings
    try:
        from celery.platforms import SecurityWarning
        warnings.filterwarnings('ignore', category=SecurityWarning, module='celery.platforms')
    except ImportError:
        pass
    warnings.filterwarnings('ignore', category=UserWarning, module='celery.platforms')
    
    logger = logging.getLogger("vocalyx.enrichment")
    logger.info("✅ Logging coloré configuré")
    
    return logger


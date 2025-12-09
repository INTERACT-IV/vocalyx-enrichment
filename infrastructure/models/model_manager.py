"""
Gestionnaire de modèles LLM pour téléchargement et gestion des modèles GGUF
Production-ready avec support Hugging Face Hub
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger("vocalyx")


class ModelManager:
    """Gère le téléchargement et la gestion des modèles GGUF"""
    
    # Modèles recommandés pour CPU (quantisés GGUF)
    RECOMMENDED_MODELS = {
        'phi-3-mini': {
            'repo_id': 'microsoft/Phi-3-mini-4k-instruct-gguf',
            'filename': 'Phi-3-mini-4k-instruct-q4.gguf',  # Nom du fichier local (correspond au fichier dans shared/)
            'size_gb': 2.3,
            'description': 'Phi-3 Mini 3.8B - Léger et rapide, idéal pour CPU'
        },
        'phi-3-medium': {
            'repo_id': 'microsoft/Phi-3-medium-4k-instruct-gguf',
            'filename': 'Phi-3-medium-4k-instruct-q4_K_M.gguf',
            'size_gb': 7.0,
            'description': 'Phi-3 Medium 14B - Meilleure qualité, plus lent'
        },
        'mistral-7b-instruct': {
            'repo_id': 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
            'filename': 'mistral-7b-instruct-v0.2.Q4_K_M.gguf',
            'size_gb': 4.1,
            'description': 'Mistral 7B Instruct - Bon équilibre qualité/vitesse'
        },
        'llama-3-8b-instruct': {
            'repo_id': 'bartowski/Llama-3-8B-Instruct-GGUF',
            'filename': 'llama-3-8b-instruct-q4_K_M.gguf',
            'size_gb': 4.6,
            'description': 'Llama 3 8B Instruct - Excellente qualité'
        },
        'gemma-2b': {
            'repo_id': 'google/gemma-2b-it-GGUF',
            'filename': 'gemma-2b-it-q4_K_M.gguf',
            'size_gb': 1.4,
            'description': 'Gemma 2B - Très léger, très rapide'
        }
    }
    
    def __init__(self, models_dir: str = './models/enrichment'):
        """
        Initialise le gestionnaire de modèles.
        
        Args:
            models_dir: Répertoire où stocker les modèles
        """
        self.models_dir = Path(models_dir)
        # Ne pas créer le répertoire s'il n'existe pas (il peut être dans shared/)
        if not self.models_dir.exists():
            # Essayer de trouver le répertoire partagé
            current_dir = Path.cwd()
            shared_dir = current_dir.parent / 'shared' / 'models' / 'enrichment'
            if shared_dir.exists():
                logger.info(f"📁 Using shared models directory: {shared_dir}")
                self.models_dir = shared_dir
            else:
                # Créer le répertoire configuré
                self.models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Model manager initialized | Directory: {self.models_dir}")
    
    def get_model_path(self, model_name: str) -> Path:
        """
        Retourne le chemin vers un modèle.
        
        Args:
            model_name: Nom du modèle ou chemin vers fichier .gguf
            
        Returns:
            Chemin Path vers le modèle
        """
        model_path = Path(model_name)
        
        # Si c'est un chemin absolu ou relatif vers un fichier existant
        if model_path.is_absolute() or (model_path.exists() and model_path.is_file()):
            return model_path
        
        # Si c'est un nom de modèle recommandé
        if model_name in self.RECOMMENDED_MODELS:
            model_info = self.RECOMMENDED_MODELS[model_name]
            model_path = self.models_dir / model_info['filename']
            
            # Si le modèle n'existe pas dans le répertoire configuré,
            # essayer de chercher dans le répertoire partagé (shared/models/enrichment)
            if not model_path.exists():
                # Chercher dans le répertoire partagé (structure vocalyx-all)
                # Essayer plusieurs chemins possibles
                current_dir = Path.cwd()
                possible_shared_dirs = [
                    self.models_dir.parent.parent / 'shared' / 'models' / 'enrichment',  # vocalyx-all/shared/models/enrichment
                    current_dir / 'shared' / 'models' / 'enrichment',  # Depuis la racine du projet
                    current_dir.parent / 'shared' / 'models' / 'enrichment',  # Depuis vocalyx-enrichment/
                    Path('../shared/models/enrichment').resolve(),  # Chemin relatif résolu
                    Path('./shared/models/enrichment').resolve(),  # Chemin relatif depuis cwd
                ]
                
                for shared_models_dir in possible_shared_dirs:
                    if shared_models_dir.exists():
                        # Chercher le fichier exact
                        shared_model_path = shared_models_dir / model_info['filename']
                        if shared_model_path.exists():
                            logger.info(f"📁 Found model in shared directory: {shared_model_path}")
                            return shared_model_path
                        
                        # Essayer avec des variations du nom de fichier
                        # Chercher tous les fichiers .gguf qui contiennent le nom du modèle
                        model_search_terms = [
                            model_name.lower().replace('-', '').replace('_', ''),
                            'phi3' if 'phi-3' in model_name.lower() else model_name.lower(),
                        ]
                        
                        for gguf_file in shared_models_dir.glob('*.gguf'):
                            file_lower = gguf_file.name.lower()
                            if any(term in file_lower for term in model_search_terms):
                                logger.info(f"📁 Found matching model in shared directory: {gguf_file}")
                                return gguf_file
            
            return model_path
        
        # Sinon, chercher dans le répertoire des modèles
        model_path = self.models_dir / model_name
        if not model_path.suffix:
            model_path = model_path.with_suffix('.gguf')
        
        # Si pas trouvé, chercher dans le répertoire partagé
        if not model_path.exists():
            shared_models_dir = self.models_dir.parent.parent / 'shared' / 'models' / 'enrichment'
            if shared_models_dir.exists():
                shared_model_path = shared_models_dir / model_path.name
                if shared_model_path.exists():
                    logger.info(f"📁 Found model in shared directory: {shared_model_path}")
                    return shared_model_path
        
        return model_path
    
    def model_exists(self, model_name: str) -> bool:
        """
        Vérifie si un modèle existe localement.
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            True si le modèle existe
        """
        model_path = self.get_model_path(model_name)
        return model_path.exists() and model_path.is_file()
    
    def download_model(self, model_name: str, use_huggingface_hub: bool = True) -> Path:
        """
        Télécharge un modèle GGUF depuis Hugging Face Hub.
        Note: Si le modèle existe déjà localement (y compris dans shared/models), 
        il ne sera pas téléchargé.
        
        Args:
            model_name: Nom du modèle (doit être dans RECOMMENDED_MODELS)
            use_huggingface_hub: Utiliser huggingface_hub si disponible
            
        Returns:
            Chemin vers le modèle téléchargé
        """
        if model_name not in self.RECOMMENDED_MODELS:
            raise ValueError(
                f"Model '{model_name}' not in recommended models. "
                f"Available: {list(self.RECOMMENDED_MODELS.keys())}"
            )
        
        # Vérifier d'abord si le modèle existe déjà (y compris dans shared/)
        model_path = self.get_model_path(model_name)
        if model_path.exists():
            logger.info(f"✅ Model already exists locally: {model_path}")
            return model_path
        
        model_info = self.RECOMMENDED_MODELS[model_name]
        
        logger.info(
            f"📥 Downloading model: {model_name} | "
            f"Size: ~{model_info['size_gb']} GB | "
            f"Description: {model_info['description']}"
        )
        
        try:
            if use_huggingface_hub:
                try:
                    from huggingface_hub import hf_hub_download
                    
                    logger.info(f"📥 Downloading from Hugging Face Hub: {model_info['repo_id']}")
                    downloaded_path = hf_hub_download(
                        repo_id=model_info['repo_id'],
                        filename=model_info['filename'],
                        local_dir=str(self.models_dir),
                        local_dir_use_symlinks=False
                    )
                    
                    # Renommer si nécessaire pour correspondre au nom attendu
                    downloaded_path = Path(downloaded_path)
                    if downloaded_path.name != model_path.name:
                        model_path.parent.mkdir(parents=True, exist_ok=True)
                        downloaded_path.rename(model_path)
                    
                    logger.info(f"✅ Model downloaded successfully: {model_path}")
                    return model_path
                    
                except ImportError:
                    logger.warning(
                        "⚠️ huggingface_hub not installed, falling back to direct download. "
                        "Install with: pip install huggingface_hub"
                    )
                    use_huggingface_hub = False
            
            # Fallback: téléchargement direct (moins fiable)
            if not use_huggingface_hub:
                logger.warning(
                    "⚠️ Direct download not implemented. "
                    "Please install huggingface_hub: pip install huggingface_hub"
                )
                raise ImportError("huggingface_hub is required for model download")
                
        except Exception as e:
            logger.error(f"❌ Failed to download model {model_name}: {e}", exc_info=True)
            raise
    
    def list_available_models(self) -> Dict[str, Dict]:
        """
        Liste les modèles recommandés disponibles.
        
        Returns:
            Dictionnaire avec les informations des modèles
        """
        return self.RECOMMENDED_MODELS.copy()
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Récupère les informations d'un modèle.
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            Dictionnaire avec les informations ou None
        """
        return self.RECOMMENDED_MODELS.get(model_name)
    
    def check_model_health(self, model_path: Path) -> bool:
        """
        Vérifie la santé d'un modèle (existence, taille, etc.).
        
        Args:
            model_path: Chemin vers le modèle
            
        Returns:
            True si le modèle est valide
        """
        if not model_path.exists():
            logger.warning(f"⚠️ Model file not found: {model_path}")
            return False
        
        if not model_path.is_file():
            logger.warning(f"⚠️ Model path is not a file: {model_path}")
            return False
        
        # Vérifier la taille (doit être > 100 MB pour un modèle GGUF)
        size_mb = model_path.stat().st_size / (1024 * 1024)
        if size_mb < 100:
            logger.warning(f"⚠️ Model file seems too small ({size_mb:.1f} MB): {model_path}")
            return False
        
        logger.info(f"✅ Model health check passed: {model_path} ({size_mb:.1f} MB)")
        return True

#!/usr/bin/env python3
"""
Script utilitaire pour trouver un modèle LLM dans les répertoires
"""

import sys
import argparse
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.models.model_manager import ModelManager


def main():
    parser = argparse.ArgumentParser(
        description='Trouve un modèle LLM dans les répertoires'
    )
    parser.add_argument(
        'model_name',
        type=str,
        nargs='?',
        default='qwen2.5-7b-instruct',
        help='Nom du modèle à chercher (défaut: qwen2.5-7b-instruct)'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='./shared/models/enrichment',
        help='Répertoire de base pour chercher les modèles (défaut: ./shared/models/enrichment)'
    )
    
    args = parser.parse_args()
    
    manager = ModelManager(models_dir=args.models_dir)
    
    print(f"🔍 Recherche du modèle: {args.model_name}\n")
    print(f"📁 Répertoire configuré: {args.models_dir}\n")
    
    # Obtenir le chemin du modèle
    model_path = manager.get_model_path(args.model_name)
    
    print(f"📍 Chemin résolu: {model_path}")
    print(f"   Existe: {'✅ OUI' if model_path.exists() else '❌ NON'}")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   Taille: {size_mb:.1f} MB")
        
        # Vérifier la santé
        if manager.check_model_health(model_path):
            print(f"   Santé: ✅ OK")
        else:
            print(f"   Santé: ⚠️  Problèmes détectés")
    else:
        print(f"\n💡 Le modèle n'a pas été trouvé.")
        print(f"   Options:")
        print(f"   1. Vérifier que le fichier existe dans: {args.models_dir}")
        print(f"   2. Utiliser un chemin absolu vers le fichier .gguf")
        print(f"   3. Télécharger le modèle avec: python scripts/download_model.py {args.model_name}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Script utilitaire pour télécharger les modèles LLM GGUF
"""

import sys
import argparse
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.models.model_manager import ModelManager


def main():
    parser = argparse.ArgumentParser(
        description='Télécharge un modèle LLM GGUF pour Vocalyx Enrichment'
    )
    parser.add_argument(
        'model_name',
        type=str,
        help='Nom du modèle à télécharger (phi-3-mini, mistral-7b-instruct, etc.)'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='./models/enrichment',
        help='Répertoire où stocker les modèles (défaut: ./models/enrichment)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='Lister les modèles disponibles'
    )
    
    args = parser.parse_args()
    
    manager = ModelManager(models_dir=args.models_dir)
    
    if args.list:
        print("\n📋 Modèles disponibles :\n")
        for name, info in manager.list_available_models().items():
            print(f"  • {name}")
            print(f"    Description: {info['description']}")
            print(f"    Taille: ~{info['size_gb']} GB")
            print(f"    Repository: {info['repo_id']}")
            print()
        return
    
    model_name = args.model_name
    
    # Vérifier si le modèle existe déjà
    if manager.model_exists(model_name):
        model_path = manager.get_model_path(model_name)
        print(f"✅ Le modèle {model_name} existe déjà : {model_path}")
        return
    
    # Vérifier si le modèle est dans la liste recommandée
    if model_name not in manager.RECOMMENDED_MODELS:
        print(f"❌ Erreur: Le modèle '{model_name}' n'est pas dans la liste recommandée.")
        print(f"\nModèles disponibles : {', '.join(manager.RECOMMENDED_MODELS.keys())}")
        print("\nUtilisez --list pour voir les détails.")
        sys.exit(1)
    
    # Télécharger le modèle
    print(f"📥 Téléchargement du modèle: {model_name}")
    print("   Cela peut prendre plusieurs minutes selon votre connexion...\n")
    
    try:
        model_path = manager.download_model(model_name)
        print(f"\n✅ Modèle téléchargé avec succès : {model_path}")
        
        # Vérifier la santé
        if manager.check_model_health(model_path):
            print("✅ Vérification de santé : OK")
        else:
            print("⚠️  Vérification de santé : Problèmes détectés")
            
    except Exception as e:
        print(f"\n❌ Erreur lors du téléchargement : {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

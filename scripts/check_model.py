#!/usr/bin/env python3
"""
Script de diagnostic pour vérifier l'état d'un modèle GGUF
"""

import sys
import argparse
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.models.model_manager import ModelManager


def check_gguf_file(file_path: Path) -> bool:
    """Vérifie si un fichier est un GGUF valide"""
    if not file_path.exists():
        print(f"❌ Fichier n'existe pas: {file_path}")
        return False
    
    if not file_path.is_file():
        print(f"❌ Chemin n'est pas un fichier: {file_path}")
        return False
    
    size_mb = file_path.stat().st_size / (1024 * 1024)
    size_gb = file_path.stat().st_size / (1024 * 1024 * 1024)
    
    print(f"📁 Fichier: {file_path}")
    print(f"   Taille: {size_gb:.2f} GB ({size_mb:.1f} MB)")
    
    # Vérifier le format GGUF
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header == b'GGUF':
                print(f"   Format: ✅ GGUF valide (magic bytes: {header.hex()})")
                return True
            else:
                print(f"   Format: ❌ Non-GGUF (magic bytes: {header.hex()}, attendu: 47475546)")
                return False
    except Exception as e:
        print(f"   Format: ❌ Erreur lors de la lecture: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Vérifie l\'état d\'un modèle GGUF'
    )
    parser.add_argument(
        'model_name',
        type=str,
        help='Nom du modèle (qwen2.5-7b-instruct, mistral-7b-instruct, phi-3-mini) ou chemin vers fichier'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='./models/enrichment',
        help='Répertoire où chercher les modèles'
    )
    
    args = parser.parse_args()
    
    manager = ModelManager(models_dir=args.models_dir)
    
    print(f"🔍 Diagnostic du modèle: {args.model_name}\n")
    
    # Obtenir le chemin du modèle
    model_path = manager.get_model_path(args.model_name)
    
    print(f"📍 Chemin résolu: {model_path}")
    print(f"   Existe: {'✅ Oui' if model_path.exists() else '❌ Non'}\n")
    
    if model_path.exists():
        # Vérifier la santé
        print("🏥 Vérification de santé:")
        is_healthy = manager.check_model_health(model_path)
        print(f"   Résultat: {'✅ Modèle valide' if is_healthy else '❌ Modèle invalide'}\n")
        
        # Vérification détaillée
        print("🔬 Vérification détaillée:")
        check_gguf_file(model_path)
    else:
        print("\n🔍 Recherche dans les emplacements possibles:")
        possible_dirs = [
            Path("/app/shared/models/enrichment"),
            Path("/app/models/enrichment"),
            Path(args.models_dir),
        ]
        
        for search_dir in possible_dirs:
            if search_dir.exists():
                files = list(search_dir.glob("*.gguf"))
                print(f"\n   {search_dir}:")
                if files:
                    print(f"      {len(files)} fichier(s) GGUF trouvé(s):")
                    for f in files:
                        size_gb = f.stat().st_size / (1024**3)
                        print(f"      - {f.name} ({size_gb:.2f} GB)")
                else:
                    print(f"      Aucun fichier GGUF trouvé")
            else:
                print(f"\n   {search_dir}: (n'existe pas)")
        
        # Si c'est un modèle recommandé, proposer de le télécharger
        if args.model_name in manager.RECOMMENDED_MODELS:
            print(f"\n💡 Le modèle '{args.model_name}' est un modèle recommandé.")
            print(f"   Vous pouvez le télécharger avec:")
            print(f"   python scripts/download_model.py {args.model_name}")


if __name__ == '__main__':
    main()


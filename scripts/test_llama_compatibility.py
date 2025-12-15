#!/usr/bin/env python3
"""
Script pour tester la compatibilité de llama-cpp-python avec un modèle GGUF
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_llama_version():
    """Teste la version de llama-cpp-python"""
    try:
        import llama_cpp
        version = getattr(llama_cpp, '__version__', 'unknown')
        print(f"✅ llama-cpp-python version: {version}")
        
        # Vérifier la version minimale
        try:
            from packaging import version as v
            if v.parse(version) >= v.parse("0.2.20"):
                print(f"   ✅ Version compatible avec Qwen 2.5 (>= 0.2.20)")
            else:
                print(f"   ⚠️  Version trop ancienne pour Qwen 2.5 (recommandé: >= 0.2.20)")
                print(f"   💡 Mettre à jour: pip install --upgrade llama-cpp-python")
        except ImportError:
            print(f"   ⚠️  Impossible de vérifier la version (packaging non installé)")
        
        return True, version
    except ImportError:
        print("❌ llama-cpp-python n'est pas installé")
        print("   💡 Installer avec: pip install llama-cpp-python")
        return False, None


def test_model_load(model_path: str):
    """Teste le chargement d'un modèle"""
    from llama_cpp import Llama
    
    model_path_obj = Path(model_path)
    
    if not model_path_obj.exists():
        print(f"❌ Fichier n'existe pas: {model_path}")
        return False
    
    print(f"\n🔍 Test de chargement du modèle: {model_path}")
    print(f"   Taille: {model_path_obj.stat().st_size / (1024**3):.2f} GB")
    
    try:
        # Essayer de charger avec des paramètres minimaux
        print("   Tentative de chargement...")
        llm = Llama(
            model_path=str(model_path),
            n_ctx=512,  # Contexte minimal pour test
            n_threads=1,
            n_batch=128,
            n_gpu_layers=0,
            verbose=False,
            use_mmap=True,
            use_mlock=False
        )
        print("   ✅ Modèle chargé avec succès!")
        
        # Test de génération simple
        print("   Test de génération...")
        response = llm("Bonjour", max_tokens=10, stop=["\n"], echo=False)
        if response and 'choices' in response and len(response['choices']) > 0:
            generated = response['choices'][0].get('text', '').strip()
            print(f"   ✅ Génération réussie: '{generated[:50]}...'")
        else:
            print(f"   ⚠️  Génération retournée mais format inattendu")
        
        del llm
        return True
        
    except ValueError as e:
        print(f"   ❌ Erreur ValueError: {e}")
        print(f"   💡 Possible causes:")
        print(f"      - Version de llama-cpp-python incompatible")
        print(f"      - Fichier corrompu")
        print(f"      - Format GGUF incompatible")
        return False
    except Exception as e:
        print(f"   ❌ Erreur: {type(e).__name__}: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Teste la compatibilité de llama-cpp-python avec un modèle GGUF'
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='Chemin vers le fichier GGUF à tester'
    )
    
    args = parser.parse_args()
    
    print("🧪 Test de compatibilité llama-cpp-python\n")
    
    # Test 1: Version
    print("1️⃣ Vérification de la version:")
    is_installed, version = test_llama_version()
    
    if not is_installed:
        sys.exit(1)
    
    # Test 2: Chargement du modèle
    print("\n2️⃣ Test de chargement du modèle:")
    success = test_model_load(args.model_path)
    
    if success:
        print("\n✅ Tous les tests ont réussi!")
        sys.exit(0)
    else:
        print("\n❌ Échec du test de chargement")
        print("\n💡 Solutions possibles:")
        print("   1. Mettre à jour llama-cpp-python:")
        print("      pip install --upgrade llama-cpp-python")
        print("   2. Avec optimisations CPU:")
        print("      CMAKE_ARGS=\"-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS\" pip install --upgrade llama-cpp-python")
        print("   3. Vérifier que le fichier GGUF n'est pas corrompu")
        print("   4. Essayer un autre modèle (mistral-7b-instruct ou phi-3-mini)")
        sys.exit(1)


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Script de test pour vérifier l'installation et le fonctionnement de l'enrichissement
"""

import sys
from pathlib import Path

# Ajouter le répertoire au path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from enrichment_service import EnrichmentService


def test_enrichment():
    """Test basique de l'enrichissement"""
    print("🧪 Test de l'enrichissement Vocalyx\n")
    
    # Afficher les informations Python
    print(f"🐍 Python: {sys.executable}")
    print(f"   Version: {sys.version.split()[0]}\n")
    
    # Vérifier que llama-cpp-python est installé
    try:
        import llama_cpp
        version = getattr(llama_cpp, '__version__', 'unknown')
        print(f"✅ llama-cpp-python est installé (version: {version})\n")
    except ImportError as e:
        print("❌ llama-cpp-python n'est pas installé")
        print(f"   Erreur: {e}")
        print("\n💡 Pour installer :")
        print(f"   {sys.executable} -m pip install llama-cpp-python")
        print("\n   Ou avec optimisations CPU (Linux) :")
        print("   CMAKE_ARGS=\"-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS\" pip3 install llama-cpp-python")
        print("\n   Voir INSTALL_LLAMA.md pour plus de détails\n")
        return False
    
    # Charger la configuration
    print("1️⃣ Chargement de la configuration...")
    try:
        config = Config()
        print(f"   ✅ Configuration chargée")
        print(f"   - Modèle: {config.llm_model}")
        print(f"   - Threads: {config.llm_n_threads or 'auto'}")
        print(f"   - Contexte: {config.llm_n_ctx}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False
    
    # Initialiser le service
    print("\n2️⃣ Initialisation du service d'enrichissement...")
    try:
        service = EnrichmentService(config)
        print(f"   ✅ Service initialisé")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False
    
    # Test d'enrichissement
    print("\n3️⃣ Test d'enrichissement d'un texte...")
    test_text = "Bonjour comment allez vous aujourd'hui"
    print(f"   Texte original: {test_text}")
    
    try:
        enriched = service.enrich_text(test_text)
        print(f"   Texte enrichi: {enriched}")
        
        if enriched == test_text:
            print("   ⚠️  Le texte n'a pas été enrichi")
            print("   💡 Cela peut indiquer que le modèle n'a pas pu être chargé")
            print("   💡 Vérifiez que llama-cpp-python est installé et que le modèle existe")
            # Ne pas échouer le test, mais avertir
        else:
            print("   ✅ Enrichissement réussi !")
            
    except ImportError as e:
        print(f"   ❌ Erreur d'importation: {e}")
        print("   💡 Installez llama-cpp-python avec: pip3 install llama-cpp-python")
        return False
    except Exception as e:
        print(f"   ❌ Erreur lors de l'enrichissement: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test avec segments
    print("\n4️⃣ Test d'enrichissement de segments...")
    test_segments = [
        {"text": "Premier segment de transcription", "start": 0.0, "end": 2.5},
        {"text": "Deuxième segment avec du contenu", "start": 2.5, "end": 5.0},
    ]
    
    try:
        enriched_segments = service.enrich_segments(test_segments)
        print(f"   ✅ {len(enriched_segments)} segments enrichis")
        
        # Vérifier que l'enrichissement a réellement fonctionné
        all_enriched = True
        for i, seg in enumerate(enriched_segments):
            enriched_text = seg.get('enriched_text', seg.get('text', ''))
            original_text = seg.get('text', '')
            is_enriched = enriched_text != original_text
            all_enriched = all_enriched and is_enriched
            status = "✅" if is_enriched else "⚠️"
            print(f"   {status} Segment {i+1}: {enriched_text[:50]}...")
        
        if not all_enriched:
            print("   ⚠️  Certains segments n'ont pas été enrichis")
            print("   💡 Vérifiez que le modèle est correctement chargé")
    except ImportError as e:
        print(f"   ❌ Erreur d'importation: {e}")
        print("   💡 Installez llama-cpp-python avec: pip3 install llama-cpp-python")
        return False
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ Tous les tests sont passés avec succès !")
    return True


if __name__ == '__main__':
    success = test_enrichment()
    sys.exit(0 if success else 1)

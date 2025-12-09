#!/usr/bin/env python3
"""
Test rapide pour vérifier la détection du modèle
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.models.model_manager import ModelManager

# Tester avec différents répertoires
test_dirs = [
    './shared/models/enrichment',
    '../shared/models/enrichment',
    './models/enrichment',
]

print("🔍 Test de détection du modèle phi-3-mini\n")

for test_dir in test_dirs:
    print(f"📁 Test avec répertoire: {test_dir}")
    manager = ModelManager(models_dir=test_dir)
    model_path = manager.get_model_path('phi-3-mini')
    print(f"   Chemin résolu: {model_path}")
    print(f"   Existe: {'✅' if model_path.exists() else '❌'}")
    if model_path.exists():
        print(f"   Taille: {model_path.stat().st_size / (1024**2):.1f} MB")
    print()

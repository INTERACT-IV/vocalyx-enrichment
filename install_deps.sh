#!/bin/bash
# Script d'installation rapide des dépendances

echo "📦 Installation des dépendances pour Vocalyx Enrichment..."

# Installer les dépendances de base
echo "1️⃣ Installation des dépendances Python de base..."
pip3 install -r requirements.txt

# Vérifier si llama-cpp-python est installé
if ! python3 -c "import llama_cpp" 2>/dev/null; then
    echo ""
    echo "2️⃣ Installation de llama-cpp-python..."
    echo "   Choisissez votre option :"
    echo "   1) Installation standard (recommandé pour débuter)"
    echo "   2) Installation optimisée CPU (OpenBLAS) - Linux uniquement"
    read -p "   Votre choix [1]: " choice
    choice=${choice:-1}
    
    if [ "$choice" = "2" ]; then
        echo "   Installation avec optimisations OpenBLAS..."
        CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip3 install llama-cpp-python
    else
        echo "   Installation standard..."
        pip3 install llama-cpp-python
    fi
else
    echo "✅ llama-cpp-python déjà installé"
fi

echo ""
echo "✅ Installation terminée !"
echo ""
echo "📝 Prochaines étapes :"
echo "   1. Vérifier que le modèle existe : python3 scripts/find_model.py phi-3-mini"
echo "   2. Tester l'installation : python3 test_enrichment.py"
echo "   3. Configurer : cp config.ini.example config.ini"

# Installation de llama-cpp-python

## Installation Standard (Recommandé pour débuter)

```bash
pip3 install llama-cpp-python
```

## Installation Optimisée CPU (Linux)

Pour de meilleures performances sur CPU, installez avec OpenBLAS :

```bash
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip3 install llama-cpp-python
```

**Note:** Cela nécessite OpenBLAS installé sur le système :
```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# CentOS/RHEL
sudo yum install openblas-devel
```

## Installation Optimisée (macOS Apple Silicon)

```bash
CMAKE_ARGS="-DLLAMA_METAL=ON" pip3 install llama-cpp-python
```

## Vérification de l'Installation

```bash
python3 -c "from llama_cpp import Llama; print('✅ llama-cpp-python installé avec succès')"
```

## Dépannage

### Erreur: "No module named 'llama_cpp'"

1. Vérifier que l'installation s'est bien passée :
```bash
pip3 show llama-cpp-python
```

2. Réinstaller si nécessaire :
```bash
pip3 uninstall llama-cpp-python
pip3 install llama-cpp-python
```

### Erreur de compilation

Si l'installation échoue, installer les dépendances de compilation :

**Ubuntu/Debian:**
```bash
sudo apt-get install build-essential cmake git
```

**CentOS/RHEL:**
```bash
sudo yum groupinstall "Development Tools"
sudo yum install cmake git
```

### Installation depuis les wheels précompilés

Si la compilation échoue, utiliser les wheels précompilés :

```bash
pip3 install llama-cpp-python --only-binary :all:
```

## Temps d'Installation

- Installation standard : 2-5 minutes
- Installation avec OpenBLAS : 5-10 minutes (compilation)
- Wheels précompilés : 30 secondes

## Après Installation

Testez que tout fonctionne :

```bash
python3 test_enrichment.py
```

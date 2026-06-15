#!/bin/bash
set -ex

# 1. Vyčištění cílového místa
rm -rf methods/smgp
mkdir -p methods

# 2. Klonování do absolutní cesty
# Použijeme `pwd` pro zjištění aktuálního pracovního adresáře
ROOT_DIR=$(pwd)
echo "--- Pracuji v: $ROOT_DIR ---"

git clone https://github.com/MichalicekPetr/SRBench-SMGPRegressor-Src-Files.git "$ROOT_DIR/methods/smgp"

# 3. DIAGNOSTIKA: Kontrola bezprostředně po klonování
cd "$ROOT_DIR/methods/smgp"
echo "--- Obsah složky po klonování (příkaz ls -la) ---"
ls -la

# Ověření cesty, kde jsme
echo "--- Aktuální absolutní cesta ---"
pwd

# 4. Instalace
pip install .
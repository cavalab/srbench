#!/bin/bash
set -ex

# 1. Vytvoř dočasnou složku pro stažení (např. v /tmp)
TMP_DIR=$(mktemp -d)

# 2. Klonuj do této dočasné složky
git clone https://github.com/MichalicekPetr/SRBench-SMGPRegressor-Src-Files.git "$TMP_DIR"

# 3. Nainstaluj balíček do systému (do site-packages)
# Pip zkopíruje soubory z TMP_DIR do systémového adresáře Pythonu
pip install "$TMP_DIR"

# 4. (Volitelné) Smaž dočasnou složku, už ji nepotřebuješ
rm -rf "$TMP_DIR"

# 5. Ověření
echo "--- Ověření instalace v systému ---"
pip show smgp
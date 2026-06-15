#!/bin/bash
set -ex # Zobrazí každý příkaz, který se spouští

# 1. Klonujeme do cílové složky
mkdir -p methods
rm -rf methods/smgp
git clone https://github.com/MichalicekPetr/SRBench-SMGPRegressor-Src-Files.git methods/smgp

# 2. PŘESNÝ PŘECHOD DO SLOŽKY
# Musíme vlézt tam, kam jsme to naklonovali!
cd methods/smgp

# 3. Instalace
pip install .

# 4. Diagnostika (pro jistotu, že tam setup.py je)
ls -la setup.py
#!/bin/bash
# Restaura un entorno a partir de requirements.txt
# Uso: ./import_env.sh nombre_del_entorno

if [ -z "$1" ]; then
    echo "❌ Debes indicar el nombre del nuevo entorno."
    echo "Ejemplo: ./import_env.sh mi_entorno"
    exit 1
fi

python -m venv $1
source $1/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
echo "✅ Entorno $1 creado con dependencias instaladas."

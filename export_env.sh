#!/bin/bash
# Exporta las dependencias del entorno actual
# Uso: ./export_env.sh nombre_del_entorno

if [ -z "$1" ]; then
    echo "❌ Debes indicar el nombre del entorno."
    echo "Ejemplo: ./export_env.sh mi_entorno"
    exit 1
fi

source $1/bin/activate
pip freeze > requirements.txt
deactivate
echo "✅ Entorno exportado a requirements.txt"

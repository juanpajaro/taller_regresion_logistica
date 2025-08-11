#!/usr/bin/env python3
"""
Evaluación específica de la función sigmoid
"""

import numpy as np
from taller import sigmoid

def test_sigmoid():
    """Evalúa la función sigmoid y verifica que coincida con los valores esperados"""
    
    # Array de entrada
    entrada = np.array([-2, -1, 0, 1, 2])
    
    # Valores esperados
    esperado = np.array([0.1192, 0.2689, 0.5, 0.7311, 0.8808])
    
    # Calcular sigmoid
    salida = sigmoid(entrada)
    
    # Mostrar resultados en el formato solicitado
    print(f"Entrada: {entrada}")
    print(f"Salida:  {np.round(salida, 4)}")
    print(f"Esperado: {esperado}")
    
    # Verificar si los resultados coinciden (con tolerancia para errores de punto flotante)
    coincide = np.allclose(np.round(salida, 4), esperado, atol=1e-4)
    
    print(f"\n¿Coincide con los valores esperados? {coincide}")
    
    if coincide:
        print("✓ La función sigmoid funciona correctamente")
    else:
        print("✗ La función sigmoid NO produce los valores esperados")
        print("Diferencias:")
        for i, (calc, esp) in enumerate(zip(np.round(salida, 4), esperado)):
            diff = abs(calc - esp)
            print(f"  Posición {i}: Calculado={calc}, Esperado={esp}, Diferencia={diff}")
    
    return coincide

if __name__ == "__main__":
    test_sigmoid()

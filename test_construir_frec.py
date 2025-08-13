#!/usr/bin/env python3
"""
Evaluación específica de la función construir_frecuencias
"""

import numpy as np
import sys
import os

# Configurar el entorno
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sol import construir_frecuencias

def test_construir_frecuencias():
    """Evalúa la función construir_frecuencias y verifica que coincida con los valores esperados"""
    
    print("="*60)
    print("EVALUACIÓN DE LA FUNCIÓN CONSTRUIR_FRECUENCIAS")
    print("="*60)
    
    # Datos de entrada exactos
    tweets_ejemplo = [
        "I love this movie",
        "I hate this movie", 
        "Amazing great wonderful",
        "Terrible bad awful",
        "Love love love",
        "Hate hate hate"
    ]
    
    etiquetas_ejemplo = np.array([[1], [0], [1], [0], [1], [0]])
    
    # Mostrar tweets de ejemplo
    print("Tweets de ejemplo:")
    for i, (tweet, etiqueta) in enumerate(zip(tweets_ejemplo, etiquetas_ejemplo.flatten())):
        sentimiento = "Positivo" if etiqueta == 1 else "Negativo"
        print(f"  {i+1}. '{tweet}' → {sentimiento}")
    
    # Construir frecuencias
    frecuencias = construir_frecuencias(tweets_ejemplo, etiquetas_ejemplo)
    
    print(f"\nTotal de pares (sentimiento, palabra) únicos: {len(frecuencias)}")
    
    # Separar frecuencias por sentimiento
    freq_positivas = {k: v for k, v in frecuencias.items() if k[0] == 1}
    freq_negativas = {k: v for k, v in frecuencias.items() if k[0] == 0}
    
    print(f"\nPalabras en tweets positivos: {len(freq_positivas)}")
    print("Frecuencias positivas:")
    for (sentimiento, palabra), frecuencia in sorted(freq_positivas.items(), key=lambda x: x[1], reverse=True):
        print(f"  '{palabra}': {frecuencia}")
    
    print(f"\nPalabras en tweets negativos: {len(freq_negativas)}")
    print("Frecuencias negativas:")
    for (sentimiento, palabra), frecuencia in sorted(freq_negativas.items(), key=lambda x: x[1], reverse=True):
        print(f"  '{palabra}': {frecuencia}")
    
    # Valores esperados
    total_esperado = 10
    positivas_esperadas = {
        'love': 4,
        'movi': 1,
        'amaz': 1,
        'great': 1,
        'wonder': 1
    }
    negativas_esperadas = {
        'hate': 4,
        'movi': 1,
        'terribl': 1,
        'bad': 1,
        'aw': 1
    }
    
    print("\n" + "="*60)
    print("VERIFICACIÓN DE RESULTADOS")
    print("="*60)
    
    # Verificar total de pares únicos
    total_correcto = len(frecuencias) == total_esperado
    print(f"Total de pares únicos: {len(frecuencias)} (esperado: {total_esperado}) {'✓' if total_correcto else '✗'}")
    
    # Verificar número de palabras positivas
    num_pos_correcto = len(freq_positivas) == len(positivas_esperadas)
    print(f"Número de palabras positivas: {len(freq_positivas)} (esperado: {len(positivas_esperadas)}) {'✓' if num_pos_correcto else '✗'}")
    
    # Verificar número de palabras negativas
    num_neg_correcto = len(freq_negativas) == len(negativas_esperadas)
    print(f"Número de palabras negativas: {len(freq_negativas)} (esperado: {len(negativas_esperadas)}) {'✓' if num_neg_correcto else '✗'}")
    
    # Verificar frecuencias positivas
    print("\nVerificación de frecuencias positivas:")
    pos_correctas = 0
    for palabra, freq_esperada in positivas_esperadas.items():
        freq_actual = freq_positivas.get((1, palabra), 0)
        correcto = freq_actual == freq_esperada
        if correcto:
            pos_correctas += 1
        print(f"  '{palabra}': {freq_actual} (esperado: {freq_esperada}) {'✓' if correcto else '✗'}")
    
    # Verificar frecuencias negativas
    print("\nVerificación de frecuencias negativas:")
    neg_correctas = 0
    for palabra, freq_esperada in negativas_esperadas.items():
        freq_actual = freq_negativas.get((0, palabra), 0)
        correcto = freq_actual == freq_esperada
        if correcto:
            neg_correctas += 1
        print(f"  '{palabra}': {freq_actual} (esperado: {freq_esperada}) {'✓' if correcto else '✗'}")
    
    # Resultado final
    print("\n" + "="*60)
    print("RESULTADO FINAL")
    print("="*60)
    
    todas_correctas = (
        total_correcto and 
        num_pos_correcto and 
        num_neg_correcto and
        pos_correctas == len(positivas_esperadas) and
        neg_correctas == len(negativas_esperadas)
    )
    
    if todas_correctas:
        print("✓ La función construir_frecuencias funciona CORRECTAMENTE")
        print("✓ Todos los valores coinciden con los esperados")
    else:
        print("✗ La función construir_frecuencias NO produce los valores esperados")
        print(f"✗ Frecuencias positivas correctas: {pos_correctas}/{len(positivas_esperadas)}")
        print(f"✗ Frecuencias negativas correctas: {neg_correctas}/{len(negativas_esperadas)}")
    
    # Mostrar diferencias si las hay
    if not todas_correctas:
        print("\nDIFERENCIAS ENCONTRADAS:")
        
        # Palabras que están de más en positivas
        palabras_extra_pos = set((k[1] for k in freq_positivas.keys())) - set(positivas_esperadas.keys())
        if palabras_extra_pos:
            print(f"Palabras positivas extra: {list(palabras_extra_pos)}")
        
        # Palabras que faltan en positivas
        palabras_faltantes_pos = set(positivas_esperadas.keys()) - set((k[1] for k in freq_positivas.keys()))
        if palabras_faltantes_pos:
            print(f"Palabras positivas faltantes: {list(palabras_faltantes_pos)}")
        
        # Palabras que están de más en negativas
        palabras_extra_neg = set((k[1] for k in freq_negativas.keys())) - set(negativas_esperadas.keys())
        if palabras_extra_neg:
            print(f"Palabras negativas extra: {list(palabras_extra_neg)}")
        
        # Palabras que faltan en negativas
        palabras_faltantes_neg = set(negativas_esperadas.keys()) - set((k[1] for k in freq_negativas.keys()))
        if palabras_faltantes_neg:
            print(f"Palabras negativas faltantes: {list(palabras_faltantes_neg)}")
    
    return todas_correctas

if __name__ == "__main__":
    test_construir_frecuencias()

#!/usr/bin/env python3
"""
Evaluación específica de la función limpiar_tweet
"""

import numpy as np
import sys
import os

# Configurar el entorno
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from taller import limpiar_tweet

def test_limpiar_tweet():
    """Evalúa la función limpiar_tweet y verifica que coincida con los valores esperados"""
    
    # Datos de prueba con sus resultados esperados
    casos_prueba = [
        {
            "tweet": "I love this movie! It's amazing #happy",
            "esperado": ['love', 'movi', 'amaz', 'happi']
        },
        {
            "tweet": "Check out this link https://example.com @username",
            "esperado": ['check', 'link']
        },
        {
            "tweet": "$100 dollars!!! This is expensive... 123",
            "esperado": ['dollar', 'expens']
        },
        {
            "tweet": "Worst day ever :( #sad #terrible",
            "esperado": ['worst', 'day', 'ever', 'sad', 'terribl']
        },
        {
            "tweet": "Amazing experience :) highly recommend!!!",
            "esperado": ['amaz', 'experi', 'highli', 'recommend']
        },
        {
            "tweet": "RT @user: Great news! https://news.com #breaking",
            "esperado": ['rt', 'great', 'news', 'break']
        },
        {
            "tweet": "I can't believe it... SO disappointed!!!",
            "esperado": ['cant', 'believ', 'disappoint']
        }
    ]
    
    print("EVALUACIÓN DE LA FUNCIÓN LIMPIAR_TWEET")
    print("="*60)
    print("Tweet original → Resultado obtenido vs Esperado")
    print("-" * 60)
    
    todos_correctos = True
    
    for i, caso in enumerate(casos_prueba, 1):
        tweet = caso["tweet"]
        esperado = caso["esperado"]
        
        # Obtener resultado de la función
        resultado = limpiar_tweet(tweet)
        
        # Mostrar resultados en el formato solicitado
        tweet_corto = tweet[:40] + "..." if len(tweet) > 40 else tweet
        print(f"'{tweet_corto}' →")
        print(f"   Obtenido: {resultado}")
        print(f"   Esperado: {esperado}")
        
        # Verificar si coinciden
        coincide = resultado == esperado
        
        if coincide:
            print(f"   ✓ CORRECTO")
        else:
            print(f"   ✗ INCORRECTO")
            todos_correctos = False
            
            # Mostrar diferencias detalladas
            print(f"   Diferencias:")
            
            # Palabras que están en el resultado pero no en lo esperado
            extras = [palabra for palabra in resultado if palabra not in esperado]
            if extras:
                print(f"     Palabras extra: {extras}")
            
            # Palabras que están en lo esperado pero no en el resultado
            faltantes = [palabra for palabra in esperado if palabra not in resultado]
            if faltantes:
                print(f"     Palabras faltantes: {faltantes}")
        
        print()
    
    # Resumen final
    print("="*60)
    print("RESUMEN DE LA EVALUACIÓN")
    print("="*60)
    
    if todos_correctos:
        print("🎉 TODAS LAS PRUEBAS PASARON CORRECTAMENTE 🎉")
        print("✓ La función limpiar_tweet produce los resultados esperados")
    else:
        print("⚠️ ALGUNAS PRUEBAS FALLARON")
        print("✗ La función limpiar_tweet NO produce todos los resultados esperados")
        print("Revisa la implementación de la función o los casos de prueba")
    
    print(f"Total de casos evaluados: {len(casos_prueba)}")
    casos_correctos = sum(1 for caso in casos_prueba if limpiar_tweet(caso["tweet"]) == caso["esperado"])
    print(f"Casos correctos: {casos_correctos}")
    print(f"Casos incorrectos: {len(casos_prueba) - casos_correctos}")
    print(f"Porcentaje de éxito: {(casos_correctos / len(casos_prueba)) * 100:.1f}%")
    
    return todos_correctos

def mostrar_casos_detallados():
    """Muestra cada caso de prueba con análisis detallado"""
    
    print("\n" + "="*60)
    print("ANÁLISIS DETALLADO DE CADA CASO")
    print("="*60)
    
    casos_prueba = [
        "I love this movie! It's amazing #happy",
        "Check out this link https://example.com @username",
        "$100 dollars!!! This is expensive... 123",
        "Worst day ever :( #sad #terrible",
        "Amazing experience :) highly recommend!!!",
        "RT @user: Great news! https://news.com #breaking",
        "I can't believe it... SO disappointed!!!"
    ]
    
    for i, tweet in enumerate(casos_prueba, 1):
        print(f"\nCASO {i}:")
        print(f"Tweet original: '{tweet}'")
        
        resultado = limpiar_tweet(tweet)
        print(f"Resultado: {resultado}")
        
        # Mostrar estadísticas
        print(f"Número de palabras procesadas: {len(resultado)}")
        
        if resultado:
            print(f"Palabras más cortas: {min(resultado, key=len)} ({len(min(resultado, key=len))} caracteres)")
            print(f"Palabras más largas: {max(resultado, key=len)} ({len(max(resultado, key=len))} caracteres)")

if __name__ == "__main__":
    # Ejecutar evaluación principal
    test_limpiar_tweet()
    
    # Mostrar análisis detallado
    mostrar_casos_detallados()

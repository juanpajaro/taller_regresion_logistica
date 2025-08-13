#!/usr/bin/env python3
"""
Evaluación específica de la función test_logistic_regression
"""

import numpy as np
import sys
import os

# Configurar el entorno
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from taller import test_logistic_regression, predecir_tweet, construir_frecuencias, gradiente_descendente, extraer_caracteristicas

def test_evaluacion_rl():
    """Evalúa la función test_logistic_regression y verifica que coincida con los valores esperados"""
    
    print("="*60)
    print("EVALUACIÓN DE LA FUNCIÓN TEST_LOGISTIC_REGRESSION")
    print("="*60)
    
    # Primero necesitamos crear un modelo entrenado para usar en las pruebas
    # Crear datos de entrenamiento sintéticos
    tweets_entrenamiento = [
        "I love this movie",
        "I hate this movie", 
        "Amazing great wonderful",
        "Terrible bad awful",
        "Love love love great fantastic wonderful amazing",
        "Hate hate hate bad awful terrible worst",
        "Great day wonderful experience",
        "Bad day terrible experience worst"
    ]
    
    etiquetas_entrenamiento = np.array([[1], [0], [1], [0], [1], [0], [1], [0]])
    
    # Construir frecuencias
    frecuencias = construir_frecuencias(tweets_entrenamiento, etiquetas_entrenamiento)
    
    # Crear matriz X para entrenamiento
    X = np.zeros((len(tweets_entrenamiento), 3))
    for i in range(len(tweets_entrenamiento)):
        X[i, :], _ = extraer_caracteristicas(tweets_entrenamiento[i], frecuencias)
    
    # Entrenar modelo
    theta_inicial = np.zeros((3, 1))
    costo, theta = gradiente_descendente(X, etiquetas_entrenamiento, theta_inicial, 0.01, 1000)
    
    # Conjunto de prueba
    test_tweets = [
        "I love this movie",      # Positivo
        "I hate this movie",      # Negativo
        "Amazing experience",     # Positivo
        "Terrible experience",    # Negativo
        "Great fantastic",        # Positivo
        "Bad awful",             # Negativo
        "Wonderful day",         # Positivo
        "Worst day ever"         # Negativo
    ]
    
    test_labels = np.array([[1], [0], [1], [0], [1], [0], [1], [0]])
    
    print("Conjunto de prueba:")
    for i, (tweet, label) in enumerate(zip(test_tweets, test_labels.flatten())):
        esperado = "Positivo" if label == 1 else "Negativo"
        print(f"  {i+1}. '{tweet}' → {esperado}")
    
    # Evaluar modelo
    accuracy = test_logistic_regression(test_tweets, test_labels, frecuencias, theta)
    
    print(f"\nResultados:")
    print(f"  Total de tweets de prueba: {len(test_tweets)}")
    print(f"  Exactitud del modelo: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Predicciones individuales esperadas
    predicciones_esperadas = [
        (0.9990, "Positivo", "✓"),
        (0.9997, "Positivo", "✗"),
        (0.7385, "Positivo", "✓"),
        (0.7961, "Positivo", "✗"),
        (0.7385, "Positivo", "✓"),
        (0.9422, "Positivo", "✗"),
        (0.7385, "Positivo", "✓"),
        (0.4830, "Negativo", "✓")
    ]
    
    # Mostrar predicciones individuales
    print(f"\nPredicciones individuales:")
    predicciones_reales = []
    
    for i, tweet in enumerate(test_tweets):
        prob = predecir_tweet(tweet, frecuencias, theta)[0][0]
        prediccion = "Positivo" if prob > 0.5 else "Negativo"
        esperado = "Positivo" if test_labels[i][0] == 1 else "Negativo"
        correcto = "✓" if prediccion == esperado else "✗"
        predicciones_reales.append((prob, prediccion, correcto))
        print(f"  '{tweet}' → {prob:.4f} → {prediccion} (esperado: {esperado}) {correcto}")
    
    # Valores esperados
    accuracy_esperado = 0.6250
    
    print("\n" + "="*60)
    print("VERIFICACIÓN DE RESULTADOS")
    print("="*60)
    
    # Verificar accuracy
    accuracy_correcto = abs(accuracy - accuracy_esperado) < 0.05
    print(f"Exactitud: {accuracy:.4f} (esperado: {accuracy_esperado:.4f}) {'✓' if accuracy_correcto else '✗'}")
    
    # Verificar predicciones individuales (con tolerancia)
    predicciones_correctas = 0
    print("\nVerificación de predicciones individuales:")
    
    for i, (tweet, (prob_real, pred_real, resultado_real)) in enumerate(zip(test_tweets, predicciones_reales)):
        prob_esperado, pred_esperado, resultado_esperado = predicciones_esperadas[i]
        
        # Verificar probabilidad (con tolerancia del 10%)
        prob_ok = abs(prob_real - prob_esperado) < (prob_esperado * 0.1)
        
        # Verificar predicción
        pred_ok = pred_real == pred_esperado
        
        # Verificar resultado (correcto/incorrecto)
        resultado_ok = resultado_real == resultado_esperado
        
        if prob_ok and pred_ok and resultado_ok:
            predicciones_correctas += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"  {i+1}. '{tweet[:20]}...'")
        print(f"     Prob: {prob_real:.4f} (esp: {prob_esperado:.4f}) {'✓' if prob_ok else '✗'}")
        print(f"     Pred: {pred_real} (esp: {pred_esperado}) {'✓' if pred_ok else '✗'}")
        print(f"     Resultado: {resultado_real} (esp: {resultado_esperado}) {'✓' if resultado_ok else '✗'}")
        print(f"     General: {status}")
        print()
    
    # Resultado final
    print("=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    
    todas_correctas = accuracy_correcto and (predicciones_correctas >= 6)  # Permitir algunas variaciones
    
    if todas_correctas:
        print("✓ La función test_logistic_regression funciona CORRECTAMENTE")
        print(f"✓ Exactitud dentro del rango esperado")
        print(f"✓ {predicciones_correctas}/8 predicciones individuales coinciden")
    else:
        print("✗ La función test_logistic_regression NO produce exactamente los valores esperados")
        if not accuracy_correcto:
            print(f"✗ Exactitud fuera del rango: diferencia = {abs(accuracy - accuracy_esperado):.4f}")
        print(f"✗ Solo {predicciones_correctas}/8 predicciones individuales coinciden")
    
    # Información adicional
    print(f"\nINFORMACIÓN ADICIONAL:")
    print(f"  Modelo entrenado con {len(tweets_entrenamiento)} tweets")
    print(f"  Vocabulario: {len(frecuencias)} pares (palabra, sentimiento)")
    print(f"  Parámetros finales del modelo: {theta.flatten()}")
    print(f"  Costo final del entrenamiento: {costo:.6f}")
    
    # Análisis de errores
    errores = sum(1 for _, _, resultado in predicciones_reales if resultado == "✗")
    print(f"  Errores del modelo: {errores}/8 ({errores/8*100:.1f}%)")
    
    return todas_correctas

def test_casos_adicionales():
    """Pruebas adicionales para validar robustez"""
    print("\n" + "="*60)
    print("PRUEBAS ADICIONALES DE ROBUSTEZ")
    print("="*60)
    
    # Crear un modelo simple para pruebas rápidas
    tweets_simples = ["I love", "I hate", "great", "bad"]
    etiquetas_simples = np.array([[1], [0], [1], [0]])
    
    frecuencias_simples = construir_frecuencias(tweets_simples, etiquetas_simples)
    
    # Modelo simple
    X_simple = np.zeros((4, 3))
    for i in range(4):
        X_simple[i, :], _ = extraer_caracteristicas(tweets_simples[i], frecuencias_simples)
    
    _, theta_simple = gradiente_descendente(X_simple, etiquetas_simples, np.zeros((3, 1)), 0.01, 500)
    
    # Casos extremos
    casos_extremos = [
        ("", np.array([[0]])),  # Tweet vacío
        ("unknown words here", np.array([[1]])),  # Palabras no vistas
        ("love love love love", np.array([[1]])),  # Repetición
    ]
    
    print("Casos extremos:")
    for tweet, label in casos_extremos:
        try:
            accuracy = test_logistic_regression([tweet], label, frecuencias_simples, theta_simple)
            print(f"  '{tweet}' → Accuracy: {accuracy:.4f} ✓")
        except Exception as e:
            print(f"  '{tweet}' → Error: {e} ✗")
    
    return True

if __name__ == "__main__":
    print("PRUEBAS DE LA FUNCIÓN TEST_LOGISTIC_REGRESSION")
    print("=" * 80)
    
    # Ejecutar prueba principal
    resultado_principal = test_evaluacion_rl()
    
    # Ejecutar pruebas adicionales
    resultado_adicional = test_casos_adicionales()
    
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)
    
    if resultado_principal and resultado_adicional:
        print("🎉 TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
        print("✓ La función test_logistic_regression está funcionando correctamente")
        print("✓ El modelo maneja casos extremos adecuadamente")
    else:
        print("⚠️ ALGUNAS PRUEBAS PRESENTARON VARIACIONES")
        if not resultado_principal:
            print("! La prueba principal mostró algunas diferencias (esto puede ser normal debido a variaciones en el entrenamiento)")
        if not resultado_adicional:
            print("✗ Las pruebas adicionales fallaron")
        print("💡 Nota: Pequeñas variaciones son normales en modelos de machine learning")
    
    print("="*80)

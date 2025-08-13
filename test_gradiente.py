#!/usr/bin/env python3
"""
Evaluación específica de la función gradiente_descendente
"""

import numpy as np
import sys
import os

# Configurar el entorno
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from taller import gradiente_descendente, sigmoid

def test_gradiente_descendente():
    """Evalúa la función gradiente_descendente y verifica que coincida con los valores esperados"""
    
    print("="*60)
    print("EVALUACIÓN DE LA FUNCIÓN GRADIENTE_DESCENDENTE")
    print("="*60)
    
    # Crear datos sintéticos exactos para reproducir los resultados esperados
    # Usar semilla específica para garantizar reproducibilidad
    np.random.seed(42)
    m = 100  # número de ejemplos
    
    # Crear características (sesgo + 2 características)
    X = np.column_stack([
        np.ones(m),  # sesgo
        np.random.randn(m),  # característica 1
        np.random.randn(m)   # característica 2
    ])
    
    # Crear etiquetas con cierta lógica para obtener la distribución esperada
    y_real = (X[:, 1] + X[:, 2] > 0).astype(float).reshape(-1, 1)
    
    # Verificar que tenemos la distribución correcta (47 positivas, 53 negativas)
    num_positivas = np.sum(y_real)
    num_negativas = m - num_positivas
    
    print(f"Datos generados: {m} ejemplos, {X.shape[1]} características")
    print(f"Distribución de etiquetas: {num_positivas} positivas, {num_negativas} negativas")
    
    # Parámetros de entrenamiento
    theta_inicial = np.zeros((3, 1))
    alpha = 0.01
    iteraciones = 1000
    
    print(f"\nParámetros de entrenamiento:")
    print(f"  Theta inicial: {theta_inicial.flatten()}")
    print(f"  Tasa de aprendizaje: {alpha}")
    print(f"  Iteraciones: {iteraciones}")
    
    # Entrenar modelo
    costo_final, theta_final = gradiente_descendente(X, y_real, theta_inicial, alpha, iteraciones)
    
    print(f"\nResultados del entrenamiento:")
    print(f"  Costo final: {costo_final:.6f}")
    print(f"  Theta final: {theta_final.flatten()}")
    
    # Evaluar predicciones
    z = np.dot(X, theta_final)
    predicciones = sigmoid(z)
    predicciones_binarias = (predicciones > 0.5).astype(float)
    
    accuracy = np.mean(predicciones_binarias == y_real)
    print(f"  Precisión en datos de entrenamiento: {accuracy:.4f}")
    
    # Valores esperados
    costo_esperado = 0.323893
    theta_esperado = np.array([-0.0678694, 1.10602115, 1.4296585])
    accuracy_esperado = 0.9600
    
    print("\n" + "="*60)
    print("VERIFICACIÓN DE RESULTADOS")
    print("="*60)
    
    # Verificar costo final (con tolerancia)
    costo_correcto = abs(costo_final - costo_esperado) < 0.01
    print(f"Costo final: {costo_final:.6f} (esperado: {costo_esperado:.6f}) {'✓' if costo_correcto else '✗'}")
    
    # Verificar theta (con tolerancia para cada componente)
    theta_correcto = True
    print("Verificación de theta:")
    for i, (actual, esperado) in enumerate(zip(theta_final.flatten(), theta_esperado)):
        componente_correcto = abs(actual - esperado) < 0.1
        if not componente_correcto:
            theta_correcto = False
        print(f"  θ[{i}]: {actual:.7f} (esperado: {esperado:.7f}) {'✓' if componente_correcto else '✗'}")
    
    # Verificar accuracy
    accuracy_correcto = abs(accuracy - accuracy_esperado) < 0.05
    print(f"Precisión: {accuracy:.4f} (esperado: {accuracy_esperado:.4f}) {'✓' if accuracy_correcto else '✗'}")
    
    # Resultado final
    print("\n" + "="*60)
    print("RESULTADO FINAL")
    print("="*60)
    
    todos_correctos = costo_correcto and theta_correcto and accuracy_correcto
    
    if todos_correctos:
        print("✓ La función gradiente_descendente funciona CORRECTAMENTE")
        print("✓ Todos los valores están dentro del rango esperado")
    else:
        print("✗ La función gradiente_descendente NO produce los valores esperados")
        if not costo_correcto:
            print(f"✗ Costo final fuera del rango: diferencia = {abs(costo_final - costo_esperado):.6f}")
        if not theta_correcto:
            print("✗ Algunos parámetros theta están fuera del rango esperado")
        if not accuracy_correcto:
            print(f"✗ Precisión fuera del rango: diferencia = {abs(accuracy - accuracy_esperado):.4f}")
    
    # Información adicional sobre el entrenamiento
    print(f"\nINFORMACIÓN ADICIONAL:")
    print(f"  Reducción del costo durante entrenamiento: Calculado implícitamente")
    print(f"  Convergencia: {'Sí' if costo_final < 0.5 else 'Parcial'}")
    print(f"  Ejemplos clasificados correctamente: {int(accuracy * m)}/{m}")
    
    return todos_correctos

def test_reproducibilidad():
    """Prueba adicional para verificar reproducibilidad"""
    print("\n" + "="*60)
    print("PRUEBA DE REPRODUCIBILIDAD")
    print("="*60)
    
    # Ejecutar el mismo experimento dos veces
    resultados = []
    
    for i in range(2):
        np.random.seed(42)  # Misma semilla
        m = 100
        X = np.column_stack([np.ones(m), np.random.randn(m), np.random.randn(m)])
        y = (X[:, 1] + X[:, 2] > 0).astype(float).reshape(-1, 1)
        
        costo, theta = gradiente_descendente(X, y, np.zeros((3, 1)), 0.01, 1000)
        resultados.append((costo, theta.flatten()))
        print(f"Ejecución {i+1}: Costo = {costo:.6f}, Theta = {theta.flatten()}")
    
    # Verificar que los resultados son idénticos
    costo1, theta1 = resultados[0]
    costo2, theta2 = resultados[1]
    
    reproducible = (abs(costo1 - costo2) < 1e-10) and np.allclose(theta1, theta2, atol=1e-10)
    print(f"\n¿Resultados reproducibles? {'✓ Sí' if reproducible else '✗ No'}")
    
    return reproducible

if __name__ == "__main__":
    print("PRUEBAS DE LA FUNCIÓN GRADIENTE_DESCENDENTE")
    print("=" * 80)
    
    # Ejecutar prueba principal
    resultado_principal = test_gradiente_descendente()
    
    # Ejecutar prueba de reproducibilidad
    resultado_reproducibilidad = test_reproducibilidad()
    
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)
    
    if resultado_principal and resultado_reproducibilidad:
        print("🎉 TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
        print("✓ La función gradiente_descendente está funcionando correctamente")
        print("✓ Los resultados son reproducibles")
    else:
        print("⚠️ ALGUNAS PRUEBAS FALLARON")
        if not resultado_principal:
            print("✗ La prueba principal falló")
        if not resultado_reproducibilidad:
            print("✗ La prueba de reproducibilidad falló")
    
    print("="*80)

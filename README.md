# Taller de Regresión Logística - PLN
**Maestría en Inteligencia Artificial - Universidad Javeriana 2025**

## 📋 Descripción del Taller

Este taller implementa un clasificador de sentimientos para tweets utilizando **Regresión Logística** desde cero. Los estudiantes aprenderán a procesar texto, extraer características y entrenar un modelo de machine learning para análisis de sentimientos.

## 🎯 Objetivos de Aprendizaje

Al completar este taller, los estudiantes serán capaces de:

1. **Preprocesar texto** eliminando ruido y normalizando tweets
2. **Extraer características** relevantes para análisis de sentimientos  
3. **Implementar regresión logística** desde cero usando NumPy
4. **Evaluar modelos** de clasificación binaria
5. **Aplicar técnicas de PLN** en problemas reales

## 🛠️ Requisitos Previos

### Conocimientos:
- Python intermedio
- Conceptos básicos de machine learning
- Álgebra lineal básica
- Procesamiento de lenguaje natural

### Librerías requeridas:
```bash
pip install nltk numpy
```

## 📚 Funciones a Implementar

### 1. **Función `limpiar_tweet(tweet)`**

**Objetivo:** Preprocesar tweets eliminando ruido y normalizando el texto.

**Tareas a completar:**
```python
def limpiar_tweet(tweet):
    # COMPLETAR:
    stemmer = ???  # Inicializar PorterStemmer
    stop_words = ???  # Cargar stopwords en inglés
    
    # Usar expresiones regulares para:
    tweet = ???  # Eliminar símbolos de monetización ($100, $USD)
    tweet = ???  # Eliminar URLs (https://...)
    tweet = ???  # Eliminar menciones (@usuario)
    tweet = ???  # Eliminar caracteres especiales y números
```

**Pistas importantes:**
- Use `PorterStemmer()` de NLTK
- Use `stopwords.words('english')` para obtener stopwords
- Expresiones regulares útiles:
  - `r'\$\w*'` para símbolos monetarios
  - `r'https?://[^\s\n\r]+'` para URLs
  - `r'@\w+'` para menciones
  - `r'[^a-zA-Z\s]'` para caracteres no alfabéticos

**Resultado esperado:**
```
Input: "I love this movie! @user #happy https://example.com"
Output: ['love', 'movi', 'happi']
```

### 2. **Función `construir_frecuencias(tweets, etiquetas)`**

**Objetivo:** Construir un diccionario de frecuencias de palabras por sentimiento.

**Tareas a completar:**
```python
def construir_frecuencias(tweets, etiquetas):
    # COMPLETAR:
    if pair in frecuencias:
        frecuencias[pair] += ???  # Incrementar contador
    else:
        frecuencias[pair] = ???  # Inicializar contador
```

**Explicación:**
- `pair = (sentimiento, palabra)` donde sentimiento es 0 (negativo) o 1 (positivo)
- El diccionario mapea `(sentimiento, palabra) → frecuencia`

**Resultado esperado:**
```
{(1, 'love'): 4, (0, 'hate'): 3, (1, 'great'): 2, ...}
```

### 3. **Función `sigmoid(z)` - ¡FUNCIÓN CLAVE!**

**Objetivo:** Implementar la función sigmoide para regresión logística.

**Tarea a completar:**
```python
def sigmoid(z):
    # COMPLETAR:
    h = ???  # Fórmula: 1 / (1 + e^(-z))
```

**Fórmula matemática:**
```
σ(z) = 1 / (1 + e^(-z))
```

**Pistas:**
- Use `np.exp()` para la exponencial
- Maneje casos donde z es muy grande o pequeño

**Resultado esperado:**
```
Input: [-2, -1, 0, 1, 2]
Output: [0.1192, 0.2689, 0.5, 0.7311, 0.8808]
```

### 4. **Función `gradiente_descendente(X, y, theta, alpha, num_iter)`**

**Objetivo:** Optimizar los parámetros del modelo usando gradiente descendente.

**Tareas a completar:**
```python
def gradiente_descendente(X, y, theta, alpha, num_iter):
    # COMPLETAR:
    m = ???  # Número de ejemplos de entrenamiento
    
    for i in range(0, num_iter):
        z = ???        # Producto punto X * theta
        h = ???        # Aplicar sigmoid a z
        J = ???        # Calcular función de costo (log-likelihood)
        theta = ???    # Actualizar parámetros
```

**Fórmulas matemáticas:**
- **Predicción:** `z = X @ theta`
- **Probabilidad:** `h = sigmoid(z)`
- **Costo:** `J = -(1/m) * [y.T @ log(h) + (1-y).T @ log(1-h)]`
- **Gradiente:** `theta = theta - (alpha/m) * X.T @ (h-y)`

**Resultado esperado:**
```
Costo final: ~0.32
Theta final: [-0.067, 1.106, 1.429]
```

### 5. **Función `test_logistic_regression(test_x, test_y, frecuencias, theta)`**

**Objetivo:** Evaluar el rendimiento del modelo entrenado.

**Tarea a completar:**
```python
def test_logistic_regression(test_x, test_y, frecuencias, theta):
    # COMPLETAR:
    accuracy = ???  # Calcular exactitud: predicciones_correctas / total
```

**Pista:**
- Compare `y_hat` (predicciones) con `test_y` (etiquetas reales)
- Use `np.mean()` para calcular el porcentaje de aciertos

## 🧪 Pruebas y Validación

El repositorio incluye archivos de prueba para validar su implementación:

- `test_sigmoid.py` - Valida la función sigmoide
- `test_trinos_limpios.py` - Valida la limpieza de tweets
- `test_construir_frec.py` - Valida la construcción de frecuencias
- `test_gradiente.py` - Valida el gradiente descendente
- `test_evaluacion_rl.py` - Valida la evaluación del modelo

**Para ejecutar las pruebas:**
```bash
python3 test_sigmoid.py
python3 test_trinos_limpios.py
python3 test_construir_frec.py
python3 test_gradiente.py
python3 test_evaluacion_rl.py
```

## 📊 Resultados Esperados

Al ejecutar `python3 taller.py`, debe obtener resultados similares a:

```
Número de tweets positivos: 5000
Número de tweets negativos: 5000
Número de palabras únicas en el vocabulario: ~9000
Probabilidades sigmoid: [0.6225 0.3775 0.8175]
Costo después del entrenamiento: 0.24921235
Exactitud del modelo: 0.9950
```

## 🎓 Criterios de Evaluación

| Función | Puntos | Criterios |
|---------|--------|-----------|
| `limpiar_tweet` | 10 | Elimina correctamente URLs, menciones, caracteres especiales |
| `construir_frecuencias` | 10 | Construye diccionario correcto de frecuencias |
| `sigmoid` | 10 | Implementa correctamente la función sigmoide |
| `gradiente_descendente` | 10 | Optimiza parámetros correctamente |
| `test_logistic_regression` | 10 | Calcula exactitud correctamente |
| **Total** | **100** | |


## 📖 Conceptos Teóricos Clave

### Regresión Logística:
- **Clasificación binaria** usando función sigmoide
- **Maximización de verosimilitud** como objetivo
- **Gradiente descendente** para optimización

### Procesamiento de Texto:
- **Tokenización** para dividir texto en palabras
- **Stemming** para normalizar variaciones morfológicas
- **Stopwords** para eliminar palabras comunes sin significado
- **Bag of Words** como representación de características

### Evaluación:
- **Exactitud (Accuracy)** como métrica principal
- **Matriz de confusión** para análisis detallado
- **Validación** en conjunto de prueba independiente

## 🚀 Entrega

1. **Archivo principal:** `taller.py` con todas las funciones implementadas
2. **Pruebas:** Todos los archivos `test_*.py` deben pasar exitosamente

**Fecha de entrega:** 25 agosto 2025, 6 pm
**Formato:** Repositorio Git con código fuente completo

---

## 📞 Soporte

Para dudas académicas, contactar:
- **Profesor:** Juan Pajaro
- **Email:** juanpajaro@javeriana.edu.co
- **Horarios de atención:** Martes 5 pm

**¡Buena suerte con el taller! 🎉**

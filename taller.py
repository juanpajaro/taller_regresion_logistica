#!/usr/bin/env python3
import nltk
nltk.download('twitter_samples')
nltk.download('stopwords')

from nltk.corpus import twitter_samples
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import string

def cargar_datos():
    
    tweets_positivos = twitter_samples.strings('positive_tweets.json')
    tweets_negativos = twitter_samples.strings('negative_tweets.json')
    return tweets_positivos, tweets_negativos

# Función que balancea los tweets positivos y negativos
def balancear_tweets(positivos, negativos, n_samples):
    test_pos = positivos[:n_samples]
    test_neg = negativos[:n_samples]
    train_pos = positivos[n_samples:]
    train_neg = negativos[n_samples:]

    train_x = train_pos + train_neg
    test_x = test_pos + test_neg

    train_y = np.append(np.ones((len(train_pos),1)), np.zeros((len(train_neg),1)), axis=0)
    test_y = np.append(np.ones((len(test_pos),1)), np.zeros((len(test_neg),1)), axis=0)

    return train_x, train_y, test_x, test_y

# Función para limpiar los tweets, eliminando URLs, menciones y caracteres especiales, también calcula el stemming y elimina stopwords
def limpiar_tweet(tweet):
    """Funcion para limpiar un tweet eliminando URLs, menciones, caracteres especiales y números,
    además de aplicar stemming y eliminar stopwords.
    Args:
        tweet (str): El tweet a limpiar.
    Returns:
        list: Lista de palabras limpias del tweet."""

    ### INICIO DE LA CREACION DE SU CODIGO ###

    stemmer = None

    stop_words = None

    ### USE EXPRESIONES REGULARES ###    

    # Eliminar simbolos de monetización
    tweet = None
    # Eliminar URLs
    tweet = None
    # Eliminar menciones
    tweet = None
    # Eliminar caracteres especiales y números
    tweet = None

    ### FIN DE LA CREACION DE SU CODIGO ###

    # Tokenizar el tweet
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_limpios =[]
    for palabra in tweet_tokens:
        if (palabra not in stop_words and palabra not in string.punctuation):
            # Aplicar stemming
            palabra_stemmed = stemmer.stem(palabra)
            tweets_limpios.append(palabra_stemmed)

    return tweets_limpios

# Función para construir las frecuencias de palabras acorde a la etiqueta, limpia los trinos, debe retornar un diccionario
def construir_frecuencias(tweets, etiquetas):

    yslista = np.squeeze(etiquetas).tolist()
    
    frecuencias = {}
    for y, tweet in zip(yslista, tweets):
        for palabra in limpiar_tweet(tweet):
            pair = (y, palabra)
            if pair in frecuencias:
                frecuencias[pair] += None
            else:
                frecuencias[pair] = None
        
    return frecuencias

#FUNCION INDISPENSABLE EN EL EJERCICIO
def sigmoid(z):
    """Función sigmoide para calcular la probabilidad de pertenencia a la clase positiva."""
    ### INICIE LA CREACION DE SU CODIGO ###
    h = None
    ### FIN DE LA CREACION DE SU CODIGO ###
    return h


def gradiente_descendente(X, y, theta, alpha, num_iter):
    """Función de gradiente descendente para optimizar los parámetros del modelo."""

    ### INICIO DE LA CREACION DE SU CODIGO ###
    m = None  # Número de ejemplos
        
    for i in range(0, num_iter):
        # Obtener Z, el producto multiplicativo de X y theta
        z = None
        # Obtener la sigmoide de Z
        h = None
        # calcular la función de costo
        J = None        
        # actualizar theta
        theta = None
    # Retornar los parámetros optimizados

    ### FIN DE LA CREACION DE SU CODIGO ###

    J = float(J)    
    return J, theta

def extraer_caracteristicas(tweet, frecuencias):
    """Extrae las características del tweet basado en la frecuencia de palabras."""

    # limpiar el tweet tokenizando, aplicando stemming y eliminando stopwords
    palabra_l = limpiar_tweet(tweet)

    # Crea un vector de ceros que representa la presencia de palabras en el tweet
    x = np.zeros((1, 3))

    # termino de sesgo
    x[0, 0] = 1
    # Itera sobre las palabras del tweet
    for palabra in palabra_l:
        # incrementa el conteo para los trinos positivos
        x[0, 1] += frecuencias.get((1, palabra), 0)
        # incrementa el conteo para los trinos negativos
        x[0, 2] += frecuencias.get((0, palabra), 0)

    return x, palabra_l

def predecir_tweet(tweet, frecuencias, theta):
    """Predice si un tweet es positivo o negativo usando el modelo entrenado."""
    x, _ = extraer_caracteristicas(tweet, frecuencias)
    # Calcular la probabilidad de que el tweet sea positivo
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred

def test_logistic_regression(test_x, test_y, frecuencias, theta):
    """Función para evaluar el modelo en el conjunto de prueba."""
    y_hat = []
    for tweet in test_x:
        y_pred = predecir_tweet(tweet, frecuencias, theta)
        if y_pred > 0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)
    y_hat = np.array(y_hat).reshape(-1, 1)

    ### INICIO DE LA CREACION DE SU CODIGO ###
    accuracy = None
    ### FIN DE LA CREACION DE SU CODIGO ###
    return accuracy

def main():
    positivos, negativos = cargar_datos()
    print(f"Número de tweets positivos: {len(positivos)}")
    print(f"Número de tweets negativos: {len(negativos)}")

    train_x, train_y, test_x, test_y = balancear_tweets(positivos, negativos, 4000)
    print(f"Número de tweets de entrenamiento: {len(train_x)}")
    print(f"Número de tweets de prueba: {len(test_x)}")
    print(f"Primer tweet de entrenamiento: {train_x[0]}")
    print("tipo train_x:", type(train_x[0]))
    print("tipo train_y:", type(train_y[0]))

    frecuencias = construir_frecuencias(train_x, train_y)
    print(f"Número de palabras únicas en el vocabulario: {len(frecuencias)}")

    # Ejemplo de uso de la función sigmoid
    ejemplo_z = np.array([0.5, -0.5, 1.5])
    probabilidades = sigmoid(ejemplo_z)
    print(f"Probabilidades de pertenencia a la clase positiva para {ejemplo_z}: {probabilidades}")

    # valores sinteticos para theta y X
    # Construct a synthetic test case using numpy PRNG functions
    #np.random.seed(1)
    # X input is 10 x 3 with ones for the bias terms
    tmp_X = np.append(np.ones((10, 1)), np.random.rand(10, 2) * 2000, axis=1)
    # Y Labels are 10 x 1
    tmp_Y = (np.random.rand(10, 1) > 0.35).astype(float)

    # Apply gradient descent
    tmp_J, tmp_theta = gradiente_descendente(tmp_X, tmp_Y, np.zeros((3, 1)), 1e-8, 700)
    print(f"The cost after training is {tmp_J:.8f}.")
    print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(tmp_theta)]}")

    # Extraer características de un tweet de ejemplo
    ejemplo_tweet = "I love programming in Python! #coding"
    print(train_x[1], train_y[1])
    caracteristicas, palabra_l = extraer_caracteristicas(train_x[1], frecuencias)
    print(f"Características extraídas del tweet: {caracteristicas}")
    print(f"Palabras del tweet: {palabra_l}")
    
    # Predecir la polaridad de un tweet de ejemplo
    for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
        print( '%s -> %f' % (tweet, predecir_tweet(tweet, frecuencias, theta)))

    my_tweet = 'I am learning happy :)'
    print(f"Probabilidad de que el tweet sea positivo: {predecir_tweet(my_tweet, frecuencias, theta)[0][0]:.8f}")

    tmp_accuracy = test_logistic_regression(test_x, test_y, frecuencias, theta)
    print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")
    

if __name__ == "__main__":
    main()
# Asegúrate de tener NLTK instalado: pip install nltk
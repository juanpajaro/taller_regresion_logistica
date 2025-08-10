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
def balancear_tweets(positivos, negativos):
    test_pos = positivos[:1000]
    test_neg = negativos[:1000]
    train_pos = positivos[1000:]
    train_neg = negativos[1000:]

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

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))    

    # Eliminar simbolos de monetización
    tweet = re.sub(r'\$\w*', '', tweet)
    # Eliminar URLs
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # Eliminar menciones
    tweet = re.sub(r'@\w+', '', tweet)
    # Eliminar caracteres especiales y números
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)

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
            if pair not in frecuencias:
                frecuencias[pair] = +1
            else:
                frecuencias[pair] = 1
        
    return frecuencias


def main():
    positivos, negativos = cargar_datos()
    print(f"Número de tweets positivos: {len(positivos)}")
    print(f"Número de tweets negativos: {len(negativos)}")

    train_x, train_y, test_x, test_y = balancear_tweets(positivos, negativos)
    print(f"Número de tweets de entrenamiento: {len(train_x)}")
    print(f"Número de tweets de prueba: {len(test_x)}")
    print(f"Primer tweet de entrenamiento: {train_x[0]}")
    print("tipo train_x:", type(train_x[0]))
    print("tipo train_y:", type(train_y[0]))

    frecuencias = construir_frecuencias(train_x, train_y)
    print(f"Número de palabras únicas en el vocabulario: {len(frecuencias)}")

if __name__ == "__main__":
    main()
# Asegúrate de tener NLTK instalado: pip install nltk
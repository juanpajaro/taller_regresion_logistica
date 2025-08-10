#!/usr/bin/env python3
import nltk
nltk.download('twitter_samples')
nltk.download('stopwords')
from nltk.corpus import twitter_samples
import numpy as np

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

def main():
    positivos, negativos = cargar_datos()
    print(f"Número de tweets positivos: {len(positivos)}")
    print(f"Número de tweets negativos: {len(negativos)}")

    train_x, train_y, test_x, test_y = balancear_tweets(positivos, negativos)
    print(f"Número de tweets de entrenamiento: {len(train_x)}")
    print(f"Número de tweets de prueba: {len(test_x)}")
    print(f"Primer tweet de entrenamiento: {train_x[0]}")
    
if __name__ == "__main__":
    main()
# Asegúrate de tener NLTK instalado: pip install nltk
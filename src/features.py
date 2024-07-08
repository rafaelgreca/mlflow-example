from gensim import downloader
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np


def get_vector_mean(df: pd.DataFrame) -> np.ndarray:
    model = downloader.load("word2vec-google-news-300")

    df["tokens"] = df["summary"].apply(word_tokenize)

    # getting the word2vec embedding of a sentence by summing the embedding of the
    # individual words and then dividing by the number of words that exists in the embedding
    def get_word2vec_embeddings(words):
        nwords = 0
        feature_vec = np.zeros((300,), dtype=np.float32)

        for word in words:
            try:
                embedding_vector = model[word]
                if embedding_vector is not None:
                    feature_vec = np.add(feature_vec, embedding_vector)
                    nwords += 1
            except KeyError:
                continue

        feature_vec = np.divide(feature_vec, nwords)
        return feature_vec

    df["embeddings"] = df["tokens"].apply(get_word2vec_embeddings)
    return np.vstack(df["embeddings"]), np.asarray(df["genre"])

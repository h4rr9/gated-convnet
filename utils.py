import numpy as np
import pickle as pk
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def mkdir_safe(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def load_embeddings_polarity():
    with open("./data/processed_data/embeddings300_polarity.pkl", "rb") as handle:
        embeddings_matrix = pk.load(handle)

    return embeddings_matrix


def load_embeddings_subjectivity():
    with open("./data/processed_data/embeddings300_subjectivity.pkl", "rb") as handle:
        embeddings_matrix = pk.load(handle)

    return embeddings_matrix


def load_embeddings_sst2():
    with open("./data/processed_data/embeddings300_sst2.pkl", "rb") as handle:
        embeddings_matrix = pk.load(handle)

    return embeddings_matrix


def split_pad_sentences(sentences, max_sentence_length):

    sentences = sentences.apply(lambda x: x.split())
    sentences = sentences.apply(
        lambda x: x + (max_sentence_length - len(x) + 1) * ["<EOF>"]
    )

    return sentences


def encode_sentences(sentences, padding_size, n_samples, path_to_mapping_file):

    with open(path_to_mapping_file, "rb") as handle:
        word2index = pk.load(handle)

    sentence_encoded = np.zeros(shape=(n_samples, padding_size), dtype=np.int32)

    for i in range(n_samples):
        for j in range(padding_size):
            sentence_encoded[i][j] = word2index[sentences[i][j]]

    return sentence_encoded


def load_polarity_data():
    train = pd.read_csv("./data/processed_data/polarity.csv")
    path_to_mapping_file = "./data/processed_data/word2index_polarity.pkl"
    n_samples = len(train)
    X = train["text"]
    Y = train["target"].values

    Y = to_categorical(Y)

    padding_size = np.max(train["text"].apply(lambda x: len(x.split())).values)

    X = split_pad_sentences(X, padding_size)

    X = encode_sentences(X, padding_size, n_samples, path_to_mapping_file)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42, shuffle=True
    )

    return (X_train, Y_train), (X_test, Y_test)


def load_subjectivity_data():
    train = pd.read_csv("./data/processed_data/subjectivity.csv")
    path_to_mapping_file = "./data/processed_data/word2index_subjectivity.pkl"
    n_samples = len(train)
    X = train["text"]
    Y = train["target"].values

    Y = to_categorical(Y)

    padding_size = np.max(train["text"].apply(lambda x: len(x.split())).values)

    X = split_pad_sentences(X, padding_size)

    X = encode_sentences(X, padding_size, n_samples, path_to_mapping_file)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42, shuffle=True
    )

    return (X_train, Y_train), (X_test, Y_test)


def load_sst2_data():
    train = pd.read_csv("./data/processed_data/sst2.csv")
    path_to_mapping_file = "./data/processed_data/word2index_sst2.pkl"
    n_samples = len(train)
    X = train["text"]
    Y = train["target"].values

    Y = to_categorical(Y)

    padding_size = np.max(train["text"].apply(lambda x: len(x.split())).values)

    X = split_pad_sentences(X, padding_size)

    X = encode_sentences(X, padding_size, n_samples, path_to_mapping_file)

    X_train, X_test, Y_train, Y_test = X[:7792], X[7792:], Y[:7792], Y[7792:]

    return (X_train, Y_train), (X_test, Y_test)

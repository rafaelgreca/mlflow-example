from gensim import downloader
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List
from transformers import BertTokenizer
import os
import torch
import pandas as pd
import numpy as np


def read_dataset(folder_path: str, file_name: str) -> pd.DataFrame:
    """
    Function responsible for reading the IMDB's Genre Classification dataset.

    Args:
        folder_path (str): the root folder path.
        file_name (str): the file name.

    Returns:
        data (pd.DataFrame): the data in a dataframe format.
    """
    data = pd.read_csv(
        filepath_or_buffer=os.path.join(folder_path, file_name),
        sep=":::",
        header=None,
        engine="python",
    )
    data.columns = ["id", "title", "genre", "summary"]
    return data


def create_dataloader(
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    labels: torch.Tensor = None,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """
    Creates a dataloader to be used with the BERT model.

    Args:
        input_ids (torch.Tensor): the input ids obtained from the BERT tokenizer.
        attention_masks (torch.Tensor): the attention mask obtained from the BERT tokenizer.
        labels (optional, torch.Tensor): the texts labels.
        batch_size (int): the batch size. Defaults to 16.
        num_workers (int): the number of workers to use. Defaults to 0.
        shuffle (bool): whether to shuffle the data or not. Defaults to True.

    Returns:
        DataLoader: the dataloader created.
    """
    if labels is not None:
        dataset = TensorDataset(input_ids, attention_masks, labels)
    else:
        dataset = TensorDataset(input_ids, attention_masks)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )

    return dataloader


def bert_preprocessing(
    texts: List, max_len: int, tokenizer: BertTokenizer
) -> Tuple[List, List]:
    """
    Preprocessing the texts to be used with BERT (using BERT Tokenizer).

    Args:
        texts (List): a list containing texts that will be processed.
        max_len (int): the max length of the text.
        tokenizer (BertTokenizer): the BERT tokenizer.

    Returns:
        Tuple[List, List]: the input ids and attention masks obtained, respectively.
    """
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_text = tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids.append(encoded_text.get("input_ids"))
        attention_masks.append(encoded_text.get("attention_mask"))

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


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

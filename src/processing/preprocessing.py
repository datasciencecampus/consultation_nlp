import os
import re
import string

import nltk
import numpy as np
import yaml
from nltk.corpus import stopwords as sw
from nltk.stem import PorterStemmer, WordNetLemmatizer
from pandas.core.series import Series


def load_config(filepath: str):
    """Loads configuration settings from given filepath to
    yaml file

    Parameters
    ----------
    filepath : str
        The relative filepath to the yaml file

    Returns
    -------
    dict
        the configuration settings with key-value pairs
    """
    if type(filepath) is not str:
        raise TypeError("filepath must be a string")

    with open(filepath, "r") as file:
        config = yaml.safe_load(file)
    return config


def remove_blank_rows(series: Series) -> Series:
    """Remove blank rows from series

    Parameters
    ----------
    series : Series
        Series of strings with blank rows

    Returns
    -------
    series
        series with blank rows removed
    """
    without_blanks = replace_blanks(series)
    without_blanks.dropna(inplace=True)
    return without_blanks


def remove_punctuation(text: str) -> str:
    """Remove punctuation from string

    Parameters
    ----------
    text : str
        string which you want to remove the punctuation from

    Returns
    -------
    str
        text string without punctuation
    """
    new_text = re.sub(string=text, pattern="[{}]".format(string.punctuation), repl="")
    return new_text


def replace_blanks(series: Series) -> Series:
    """Replace blanks within array with NaN

    Parameters
    ----------
    series : Series
        series of strings containing some blank rows

    Returns
    -------
    Series
        series with blanks replaced with 'NaN'
    """
    blanks_replaced = series.replace([r"^\s*?$"], np.NaN, regex=True)
    return blanks_replaced


def stemmer(tokens: list) -> list:
    """Stem works to their root form (e.g. flying -> fly, Beautiful -> Beauty)

    Parameters
    ----------
    tokens : list
        list of word tokens

    Returns
    -------
    stemmed_tokens
        list of root word tokens
    """
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens


def lemmatizer(tokens: list) -> list:
    """Lemmatize tokens to group words with similar meanings
    (e.g. better -> good)

    Parameters
    ----------
    tokens : list
        list of word tokens

    Returns
    -------
    lemmatized_tokens
        list of simplified word groupings
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


def initialise_nltk_stopwords() -> list:
    """fetch nltk stopwords from corpora

    Returns
    -------
    list
        list of nltk stopwords
    """
    username = os.getenv("username")
    path = "c:/Users/" + username + "/AppData/Roaming/nltk_data/corpora/stopwords"
    if not os.path.exists(path):
        nltk.download("stopwords")
    nltk.data.path.append("../local_packages/nltk_data")
    stopwords = sw.words("english")
    return stopwords


def remove_nltk_stopwords(tokens: list) -> list:
    """remove stopwords from series

    Parameters
    ----------
    tokens : list
        list of tokenized words including stopwords

    Returns
    -------
    list
        token list without stopwords
    """
    stopwords = initialise_nltk_stopwords()
    without_stopwords = [item for item in tokens if item not in stopwords]
    return without_stopwords


def rejoin_tokens(tokens: list) -> str:
    """rejoin tokens into a string

    Parameters
    ----------
    tokens : list
        list of tokenized words

    Returns
    -------
    str
        tokenized words separated by spaces
    """
    joined_tokens = " ".join(tokens)
    return joined_tokens

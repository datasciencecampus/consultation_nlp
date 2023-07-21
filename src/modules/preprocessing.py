import json
import os
import sys

import nltk
import numpy as np
import yaml
from nltk.corpus import stopwords as sw
from nltk.stem import PorterStemmer, WordNetLemmatizer
from pandas import Series


def load_config(filepath: str) -> dict:
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
        config = yaml.load(file, Loader=yaml.Loader)
    return config


def load_json(filepath: str) -> dict:
    """Loads json file as dictionary
    Parameters
    ----------
    filepath:str
        the filepath to where the json file is stored
    Returns
    -------
    dict
        the json file in dict format
    """
    if type(filepath) is not str:
        raise TypeError("filepath must be a string")
    with open(filepath, "r") as file:
        json_data = json.load(file)
    return json_data


def prepend_str_to_list_objects(list_object: list, string_x: str = "qu_"):
    """add word to the front of list of question numbers
    Parameters
    ----------
    list_object: list
        list of objects to add word to
    string: str
        string to prepend to each object
    Returns
    -------
    list
        a list with the string prepended to each object
    """
    question_list = list(map(lambda object: string_x + str(object), list_object))
    return question_list


def get_response_length(series: Series) -> Series:
    """get length of each row in series
    Parameters
    ----------
    series: Series
        series you want to get the response lengths for
    Returns
    -------
    Series
        a series of lengths
    """
    series.fillna("", inplace=True)
    response_length = series.apply(len)
    return response_length


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
    without_blanks = _replace_blanks(series)
    without_blanks.dropna(inplace=True)
    return without_blanks


def _replace_blanks(series: Series) -> Series:
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


def shorten_tokens(word_tokens: Series, lemmatize: bool = True) -> list:
    """Shorten tokens to root words
    Parameters
    ----------
    word_tokens:Series
        Series of listed word tokens
    lemmatize: bool, default = True
        whether to use lemmatizer or revert back to False (stemmer)"""
    if lemmatize:
        short_tokens = word_tokens.apply(lemmatizer)
    else:
        short_tokens = word_tokens.apply(stemmer)
    return short_tokens


def stemmer(tokens: list) -> list:
    """Stem works to their root form (e.g. flying -> fli, Beautiful -> Beauti)

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
    _initialise_nltk_component("corpora/wordnet.zip", "wordnet")
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


def _initialise_nltk_component(extension: str, download_object: str):
    """spliter function to determine which initialisation path to run
    Parameters
    ----------
    extension: str
        the filepath extension leading to where the model is saved
    download_object: str
        the object to download from nltk
    Returns
    -------
    None
    """
    if sys.platform.startswith("linux"):
        nltk.download(download_object)
        nltk.data.path.append("../home/runner/nltk_data")
    else:
        username = os.getenv("username")
        path = "C:/Users/" + username + "/AppData/Roaming/nltk_data/" + extension
        if not os.path.exists(path):
            nltk.download(download_object)
            nltk.data.path.append("../local_packages/nltk_data")
    return None


def remove_nltk_stopwords(tokens: list, additional_stopwords: list = []) -> list:
    """remove stopwords from series

    Parameters
    ----------
    tokens : list
        list of tokenized words including stopwords
    additional_stopwords:list
        additional words to add to the stopwords (defualt empty list)
    Returns
    -------
    list
        token list without stopwords
    """
    stopwords = initialise_update_stopwords(additional_stopwords)
    without_stopwords = [item for item in tokens if item not in stopwords]
    return without_stopwords


def initialise_update_stopwords(additional_stopwords: list = None) -> list:
    """initialise and update stopwords, ise this for efficient retrieval of
    stopwords, rather than calling both functions.
    Parameters
    ----------
    additional_stopwords:list
        new words to add to the words to remove list
    Returns
    -------
    list
        a list of words to remove from corpus
    """
    _initialise_nltk_component("corpora/stopwords", "stopwords")
    stopwords = sw.words("english")
    updated_stopwords = _update_nltk_stopwords(stopwords, additional_stopwords)
    return updated_stopwords


def _update_nltk_stopwords(stopwords: list, additional_stopwords: list):
    """add additional words to nltk stopwords
    Parameters
    ----------
    stopwords: list
        a list of stopwords
    additional_stopwords:list
        new words to add to the words to remove list
    Returns
    -------
    list
        a corpus of words to remove
    """
    for word in additional_stopwords:
        stopwords.append(word)
    return stopwords


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

import os
import re
import string
import sys

import nltk
import numpy as np
import textblob as tb
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


def spellcorrect_series(series: Series, additional_words: dict = {}) -> Series:
    """fix spelling across series using the norvig spell-correct method
    Parameters
    ----------
    series: Series
        the series of text strings you want to pass your spell checker on
    additional_words:dict
        a dictionary of words and weights for each word
    Returns
    -------
    Series
        a series with words spelling corrected"""
    tb.en.spelling = _update_spelling_words(additional_words)
    corrected_series = series.apply(lambda str: _correct_spelling(str))
    return corrected_series


def _correct_spelling(string: str) -> str:
    """correct spelling using norvig spell-correct method
    (it has around 70% accuracy)
    Parameters
    ----------
    string:str
        string you want to fix the spelling in
    Returns
    -------
    str
        string with the spelling fixed"""
    spelling_fixed = str(tb.TextBlob(string).correct())
    return spelling_fixed


def _update_spelling_words(additional_words: dict) -> None:
    """update word in the textblob library with commonly used business word
    Parameters
    ----------
    additional_words:dict
        words to add to the textblob dictionary, with associated weights.
        higher weights give greater precedence to the weighted word.
    Returns
    -------
    dict
        a dictionary of words and updated weights
    """
    for word, weight in additional_words.items():
        tb.en.spelling.update({word: weight})
    return tb.en.spelling


def remove_punctuation(series: Series) -> Series:
    """Remove punctuation from series of strings"""
    _initialise_nltk_component("tokenizers/punkt", "punkt")
    punct_removed = series.apply(_remove_punctuation_string)
    return punct_removed


def _remove_punctuation_string(text: str) -> str:
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


def shorten_tokens(word_tokens: list, lemmatize: bool = True) -> list:
    """Shorten tokens to root words
    Parameters
    ----------
    word_tokens:list
        list of word tokens to shorten
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
    """download nltk component from package
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
    username = os.getenv("username")
    path = "C:/Users/" + username + "/AppData/Roaming/nltk_data/" + extension
    if not os.path.exists(path):
        nltk.download(download_object)
    # Set path for runs on github actions
    if sys.platform.startswith("linux"):
        nltk.data.path.append("../home/runner/nltk_data")
    else:
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

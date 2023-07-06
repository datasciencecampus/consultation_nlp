import os
import re
import string

import nltk
import numpy as np
import textblob as tb
import yaml
from nltk.corpus import stopwords as sw
from nltk.stem import PorterStemmer, WordNetLemmatizer
from pandas.core.series import Series
from rapidfuzz.fuzz import ratio


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


def correct_spelling(string: str, additional_words: list = []) -> str:
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
    _update_spelling_words(additional_words)
    spelling_fixed = str(tb.TextBlob(string).correct())
    return spelling_fixed


def _update_spelling_words(additional_words: list) -> None:
    """update word in the textblob library with commonly used business word
    Parameters
    ----------
    additional_words:list
        words to add to the textblob dictionary
    Returns
    -------
    None
    """
    for word in additional_words:
        tb.en.spelling.update({word: 1})
        tb.en.spelling
    return None


def fuzzy_compare_ratio(base: Series, comparison: Series) -> Series:
    """compare the base series to the comparison series to get
    a similarity ratio between strings in the same column
    Parameters
    ----------
    base: Series
        the base series for comparison
    comparison: Series
        the series you want to compare against
    Returns
    -------
    Series
        a series of ratios (type:float) with scores closer to 100
        indicating complete match"""
    fuzzy_ratio = Series(map(ratio, base, comparison))
    return fuzzy_ratio


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
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


# @pytest.mark.skipif(sys.platform.startswith("linux"), reason="Cannot download file")
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
    stopwords = _initialise_nltk_stopwords()
    updated_stopwords = _update_nltk_stopwords(stopwords, additional_stopwords)
    without_stopwords = [item for item in tokens if item not in updated_stopwords]
    return without_stopwords


# @pytest.mark.skipif(sys.platform.startswith("linux"), reason="Cannot download file")
def _initialise_nltk_stopwords() -> list:
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


# @pytest.mark.skipif(sys.platform.startswith("linux"), reason="Cannot download file")
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

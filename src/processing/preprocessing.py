import os
import re
import string

import nltk
import numpy as np
import textblob as tb
import yaml
from nltk.corpus import stopwords as sw
from nltk.stem import PorterStemmer, WordNetLemmatizer
from numpy.typing import ArrayLike
from pandas import DataFrame, Series
from rapidfuzz.fuzz import ratio
from sklearn.feature_extraction.text import CountVectorizer


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
    corrected_series = series.apply(
        lambda str: _correct_spelling(str, additional_words)
    )
    return corrected_series


def _correct_spelling(string: str, additional_words: dict = {}) -> str:
    """correct spelling using norvig spell-correct method
    (it has around 70% accuracy)
    Parameters
    ----------
    string:str
        string you want to fix the spelling in
    additional_words:dict, default = None
        words to add to the textblob dictionary, with associated weights.
        higher weights give greater precedence to the weighted word.
    Returns
    -------
    str
        string with the spelling fixed"""
    tb.en.spelling = _update_spelling_words(additional_words)
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
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


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
    stopwords = _initialise_nltk_stopwords()
    updated_stopwords = _update_nltk_stopwords(stopwords, additional_stopwords)
    return updated_stopwords


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


def extract_feature_count(
    series: Series,
    max_features: int = None,
    ngram_range: tuple[float, float] = (1, 1),
    stop_words: ArrayLike = None,
    lowercase: bool = True,
    min_df: float | int = 1,
    max_df: float | int = 1.0,
):
    """create a text feature count dataframe from series
    Paramaters
    ----------
    series: Series
        Series of text strings
    max_features: int, default = None
        If not None, build a vocabulary that only consider the top max_features
        ordered by term frequency across the corpus. Otherwise, all features are used.
    ngram_range: tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different word n-grams
        or char n-grams to be extracted. All values of n such such that
        min_n <= n <= max_n will be used.
    stop_words: list, default=None
        list of stopwords to remove from text strings
    lowercase: bool, default = True
        convert all characters to lowercase before tokenizing
    min_df: float or int, default = 1
        When building the vocabulary ignore terms that have a document frequency
        strictly lower than the given threshold. This value is also called cut-off
        in the literature. If float, the parameter represents a proportion of
        documents, integer absolute counts.
        This parameter is ignored if vocabulary is not None.
    max_df: float or int, default = 1.0
        When building the vocabulary ignore terms that have a document frequency
        strictly higher than the given threshold (corpus-specific stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts. This parameter is ignored if vocabulary is not None.
    Returns
    -------
    DataFrame
        A dataframe of text feature counts, displaying the number of times a word
        appears in each element of the input series
    """

    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
        lowercase=lowercase,
        min_df=min_df,
        max_df=max_df,
    )

    fitted_vector = vectorizer.fit_transform(series)

    word_count_df = DataFrame(
        fitted_vector.toarray(), columns=vectorizer.get_feature_names_out()
    )
    return word_count_df


def get_total_feature_count(features: DataFrame) -> DataFrame:
    """sum across features to get total number of times word was used
    Parameters
    ----------
    features: DataFrame
        A dataframe of the features with each row corrosponding to a deconstructed
        string
    Returns
    -------
    DataFrame
        A dataframe of the total number of times each word is used across all
        strings"""
    total_feature_count = DataFrame()
    for column in features.columns:
        total_feature_count[column] = [features[column].sum()]
    return total_feature_count

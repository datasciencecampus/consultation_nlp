import re
import string

from pandas import Series
from spellchecker import SpellChecker

from src.modules.utils import _trim_ends


def find_word_replacements(series: Series, spell: SpellChecker) -> Series:
    """find suggested word replacements across series
    Parameters
    ----------
    series:Series
        series of strings to find suggestions for
    spell:SpellChecker
        spellchecker class object
    Returns
    -------
    Series
        a series of key value pairs for original and suggested replacement words
    """
    return series.apply(
        lambda string_x: _find_word_replacements_string(string_x, spell)
    )


def _find_word_replacements_string(string_x: str, spell: SpellChecker) -> dict:
    """find suggested word replacements across string
    Parameters
    ----------
    string_x:str
        string to analyse for word corrections
    spell:SpellChecker
        spellchecker class object
    Returns
    -------
    dict
        key value pairs for each of the original words and suggested replacements
    """
    word_tokens = _tokenize_words(string_x)
    highlighted_words = [word for word in spell.unknown(word_tokens) if len(word) > 0]
    if len(highlighted_words) > 0:
        replacements = [spell.correction(word) for word in highlighted_words]
    word_replacements = {
        highlighted_words[i]: replacements[i] for i in range(len(highlighted_words))
    }
    return word_replacements


def replace_words(series: Series, word_replacements: Series) -> Series:
    """Replace words in series with suggested word replacements
    Parameters
    ----------
    series:Series
        a series of strings to update
    word_replacements:Series
        series of dictionaries with original and suggested replacement words
    Returns
    -------
    Series
        series with suggested replacement words implemented"""
    updated_series = Series(
        map(
            lambda x: _replace_words_string(series.iloc[x], word_replacements.iloc[x]),
            [i for i in range(len(series))],
        )
    )
    return updated_series


def _replace_words_string(string_x: str, word_replacements: dict) -> str:
    """Replace words in string with suggested word replacements
    Parameters
    ----------
    string_x:str
        a text string with words to update
    word_replacements:dict
        original word (key) and it's suggested replacment (value)
    Returns
    -------
    str
        a string with original words updated with suggested replacements
    """
    for key, value in word_replacements.items():
        if value is not None:
            start_loc, end_loc = re.search(key, string_x.lower()).span()
            original = string_x[start_loc:end_loc]
            is_captialized = original[0].isupper()
            if is_captialized:
                value = value.capitalize()
            string_x = re.sub(original, value, string_x)
    return string_x


def update_spell_dictionary(additional_words: dict) -> SpellChecker:
    """update spell checker dictionary with additional words
    Parameters
    ----------
    addtional_words:dict
        dictionary of additional words and their associated weights
    Returns
    -------
    SpellChecker
        SpellChecker with added words"""
    spell = SpellChecker(distance=2)
    for key, value in additional_words.items():
        spell.word_frequency.add(key, value)
    return spell


def _tokenize_words(string_x: str) -> list:
    """tokenize words by spliting at each space
    Parameters
    ----------
    string_x:str
        string you want to tokenize
    Returns
    -------
    list
        list of word tokens"""
    return string_x.split(" ")


def remove_punctuation(string_x: str) -> str:
    """remove punctuation surrounding words
    Parameters
    ----------
    string_x:str
        string you want to remove punctuation from
    Returns
    -------
    str
        string with target punctuation removed"""
    string_x = _remove_punctuation_sentence_start(string_x)
    string_x = _remove_punctuation_space_after(string_x)
    string_x = _remove_punctuation_space_before(string_x)
    string_x = _remove_punctuation_sentence_end(string_x)
    string_x = _trim_ends(string_x)
    return string_x


def _remove_punctuation_space_before(string_x: str) -> str:
    """remove punctuation with a white space after it eg. 'hello. ' -> 'hello  '
    Parameters
    ----------
    string_x:str
        string you want to remove punctuation from
    Returns
    -------
    str
        string with target punctuation removed"""
    escaped_punctuation = re.escape(string.punctuation)
    return re.sub(rf"(?<=\s)[{escaped_punctuation}]+", " ", string_x)


def _remove_punctuation_space_after(string_x: str) -> str:
    """remove punctuation that occurs after a space e.g. 'hello ! " -> 'hello   '
    Parameters
    ----------
    string_x:str
        string you want to remove punctuation from
    Returns
    -------
    str
        string with target punctuation removed"""
    escaped_punctuation = re.escape(string.punctuation)
    return re.sub(rf"[{escaped_punctuation}]+(?=\s)", " ", string_x)


def _remove_punctuation_sentence_end(string_x: str) -> str:
    """remove punctuation from the end of a string e.g. 'hello!' -> 'hello '
    Parameters
    ----------
    string_x:str
        string you want to remove punctuation from
    Returns
    -------
    str
        string with target punctuation removed"""
    escaped_punctuation = re.escape(string.punctuation)
    return re.sub(rf"[{escaped_punctuation}]+$", " ", string_x)


def _remove_punctuation_sentence_start(string_x: str) -> str:
    """remove punctuation from the start of a string e.g. '!hello' -> ' hello'
    Parameters
    ----------
    string_x:str
        string you want to remove punctuation from
    Returns
    -------
    str
        string with target punctuation removed"""
    escaped_punctuation = re.escape(string.punctuation)
    return re.sub(rf"^[{escaped_punctuation}]+(?=\w)", " ", string_x)

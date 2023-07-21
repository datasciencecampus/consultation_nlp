import re
import string

from pandas import Series
from spellchecker import SpellChecker

from src.modules.utils import _trim_ends


def auto_correct_series(series: Series, additional_words: dict) -> Series:
    """"""
    spell = _update_spell_dictionary(additional_words)
    modifications = {}
    for n, string_x in enumerate(series):
        changes = {}
        word_tokens = _tokenize_words(string_x)
        puctuation_removed = list(map(remove_punctuation, word_tokens))
        highlighted_words = [
            word for word in spell.unknown(puctuation_removed) if len(word) > 0
        ]
        if len(highlighted_words) > 0:
            for word in highlighted_words:
                string_x, changes[word] = _auto_correct(word, string_x, spell)
        series[n] = string_x
        modifications[n + 1] = changes
    return series, modifications


def _update_spell_dictionary(additional_words: dict) -> SpellChecker:
    """update spell checker dictionary with additional words
    Parameters
    ----------
    addtional_words:dict
        dictionary of additional words and their associated weights
    Returns
    -------
    SpellChecker
        SpellChecker with added words"""
    spell = SpellChecker()
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
    return re.sub(rf"(?<=\s)[{escaped_punctuation}]", " ", string_x)


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


def _auto_correct(word: str, string_x: str, spell: SpellChecker) -> str:
    """auto-correct words based on Levenshtien Distance algorithm"""
    start, end = re.search(word, string_x.lower()).span()
    original = string_x[start:end]
    corrected = spell.correction(original)
    replacement = filter_acronymns(original, corrected)
    try:
        new_string = re.sub(original, replacement, string_x)
    except TypeError:
        new_string = string_x
        replacement = original
    if original != replacement:
        output = (new_string, replacement)
    else:
        output = (string_x, None)
    return output


def filter_acronymns(original: str, corrected: str):
    """returns original word if it meets the pattern criteria for an acronym
    otherwise returns corrected word.
    Acronym pattern: uppercase(1+),lowercase(0+), uppercase(1+), lowercase(0+).
    Parameters
    ----------
    original:str
        original word
    correct: str
        corrected word
    Returns
    -------
    str
        either original or corrected string"""
    if re.search(r"([A-Z]+[a-z]*[A-Z]+[a-z]*)", original):
        return original
    else:
        return corrected

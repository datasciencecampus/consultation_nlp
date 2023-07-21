import re
import string

from src.modules.utils import _trim_ends


def clean_string(string_x: str) -> str:
    """prepare string for automated reading by removing unwanted features
    Parameters
    ----------
    string_x:str
        string to remove unwated features from
    Returns
    -------
    str
        string without unwanted features
    """
    string_x = _remove_number_punctuation(string_x)
    string_x = _remove_elipsis(string_x)
    string_x = _replace_ampersand_plus(string_x)
    string_x = _remove_drop_lines(string_x)
    string_x = _remove_escape_chars(string_x)
    string_x = _remove_non_word_joiners(string_x)
    string_x = _remove_double_space(string_x)
    string_x = _trim_ends(string_x)
    return string_x


def _remove_number_punctuation(string_x: str) -> str:
    """remove punctuation from within continous number strings e.g. 2,000 -> 2000
    Parameters
    ----------
    string_x:str
        string to remove number punctuation from
    Returns
    -------
    str
        string with number punctuation removed
    """
    return re.sub(r"(?<=\d)[,:.](?=\d)", "", string_x)


def _remove_elipsis(string_x: str) -> str:
    """remove elipsis (...) from string
    Parameters
    ----------
    string_x:str
        string containing elipsis
    Returns
    -------
    str
        string without elipisis
    """
    return re.sub(re.escape("..."), " ", string_x)


def _replace_ampersand_plus(string_x: str) -> str:
    """replace ampersand (&) and plus (+) with 'and'
    Paramaters
    ----------
    string_x:str
        sting to replace characters from
    Returns
    -------
    str
        string with ampersands and pluses replaced with 'and'"""
    return re.sub(r"(&|\+)", " and ", string_x)


def _remove_drop_lines(string_x: str) -> str:
    """remove drop line characters (backslash n) from string
    Parameters
    ----------
    string_x:str
        string to remove drop line characters from
    Returns
    -------
    str
        string without drop line characters"""
    return re.sub(r"\n", " ", string_x)


def _remove_escape_chars(string_x: str) -> str:
    """remove escape chararacters from string
    Parameters
    ----------
    string_x:str
        string you want to remove escape characters from
    Returns
    -------
    str
        string without escape characters"""
    return re.sub(r"(\\)", " ", string_x)


def _remove_non_word_joiners(string_x: str) -> str:
    """remove punctuation from within words (except for hyphen, apostrophe,
    underscore and colon)
    Parameters
    ----------
    string_x:str
        string you want to remove punctuation from
    Returns
    -------
    str
        string without the target punctuation"""
    escaped_punctuation = re.escape(string.punctuation)
    escaped_punctuation = re.escape(re.sub("[\\-_':]", "", string.punctuation))
    return re.sub(rf"(?<=\w)[{escaped_punctuation}](?=\w)", "", string_x)


def _remove_double_space(string_x: str) -> str:
    """remove double white spaces from within strings
    Parameters
    ----------
    string_x:str
        string to remove double white spaces from
    Returns
    -------
    str
        string without double white spaces
    """
    return re.sub(r"\s\s+", " ", string_x)

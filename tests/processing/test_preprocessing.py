import sys
import unittest
from itertools import repeat

import numpy as np
import pytest
import textblob as tb
from pandas import DataFrame, Series

from src.processing.preprocessing import (
    _initialise_nltk_stopwords,
    _replace_blanks,
    _update_nltk_stopwords,
    _update_spelling_words,
    correct_spelling,
    extract_feature_count,
    fuzzy_compare_ratio,
    get_total_feature_count,
    initialise_update_stopwords,
    lemmatizer,
    load_config,
    rejoin_tokens,
    remove_blank_rows,
    remove_nltk_stopwords,
    remove_punctuation,
    stemmer,
)


class TestLoadConfig:
    def test_input_type_error(self):
        """test for assertion error"""
        bad_input = 123
        with pytest.raises(Exception) as e_info:
            load_config(bad_input)
        assert (
            str(e_info.value) == "filepath must be a string"
        ), "Did not raise TypeError"

    def test_input_file_not_found(self):
        """test for error feedback on file not found"""
        bad_input = "src/superman.yaml"
        with pytest.raises(Exception) as e_info:
            load_config(bad_input)
        assert (
            str(e_info.value.args[1]) == "No such file or directory"
        ), "Did not raise file not found error"

    def test_return_dict(self):
        assert (
            type(load_config("src/config.yaml")) is dict
        ), "output is not <class 'dict'>"


class TestRemoveBlankRows:
    def test_blank_rows_removed(self):
        """test that blank rows are removed"""
        series_with_empty_row = Series([1.0, "", 3.0])
        expected_outcome = Series([1.0, 3.0])
        actual = remove_blank_rows(series_with_empty_row)
        actual_reindexed = actual.reset_index(drop=True)
        assert all(
            actual_reindexed == expected_outcome
        ), "function does not remove blank rows"

    def test_return_series(self):
        """test that function returns a Series"""
        actual = remove_blank_rows(Series([1.0, "", 3.0]))
        assert (
            type(actual) is Series
        ), "output is not <class 'pandas.core.series.Series'>"


class TestReplaceBlanks:
    def test_blank_replacement(self):
        """test replace blanks with NaN"""
        series_with_empty_row = Series([1.0, "", 3.0])
        actual = _replace_blanks(series_with_empty_row)
        assert np.isnan(actual[1]), "did not replace blank with NaN"

    def test_return_series(self):
        """test that function returns a Series"""
        actual = remove_blank_rows(Series([1.0, "", 3.0]))
        assert (
            type(actual) is Series
        ), "output is not <class 'pandas.core.series.Series'>"


class TestCorrectSpelling:
    def test_spelling_fixed(self):
        house_str = "I live in a housr"
        corrected = correct_spelling(house_str)
        assert corrected == "I live in a house", "spelling not fixed correctly"

    def test_word_update(self):
        additional_words = ["housr"]
        house_str = "I live in a housr"
        corrected = correct_spelling(house_str, additional_words)
        assert (
            corrected == "I live in a housr"
        ), "spelling word list not correctly updated"


class TestUpdateSpellingWords:
    def test_update_word_list(self):
        additional_words = ["housr"]
        _update_spelling_words(additional_words)
        assert (
            "housr" in tb.en.spelling.keys()
        ), "spelling word list not updated correctly"


class TestFuzzyCompareRatio:
    def test_ratios(self):
        base = Series(["this is", "this isn't"])
        comparison = Series(["this is", "yellow"])
        expected = Series([100.00, 0.0])
        actual = fuzzy_compare_ratio(base, comparison)
        assert all(expected == actual), "fuzzy scoring not working correctly"


class TestRemovePunctuation:
    def test_remove_punctuation(self):
        test_string = "my #$%&()*+,-./:;<=>?@[]^_`{|}~?name"
        actual = remove_punctuation(test_string)
        expected = "my name"
        assert actual == expected, "punctuation not removed correctly"


class TestStemmer:
    def test_stemmer(self):
        word_list = ["flying", "fly", "Beautiful", "Beauty"]
        actual = stemmer(word_list)
        expected = ["fli", "fli", "beauti", "beauti"]
        assert actual == expected, "words are not being stemmed correctly"


class TestLemmatizer:
    @pytest.mark.skipif(sys.platform.startswith("linux"), reason="Cannot download file")
    def test_lemmatization(self):
        word_list = ["house", "houses", "housing"]
        actual = lemmatizer(word_list)
        expected = ["house", "house", "housing"]
        assert actual == expected, "words are not being lemmatized correctly"


class TestRemoveNLTKStopwords:
    @pytest.mark.skipif(sys.platform.startswith("linux"), reason="Cannot download file")
    def test_remove_standard_stopwords(self):
        tokens = ["my", "name", "is", "elf", "who", "are", "you"]
        actual = remove_nltk_stopwords(tokens)
        expected = ["name", "elf"]
        assert actual == expected, "core stopwords not being removed correctly"

    @pytest.mark.skipif(sys.platform.startswith("linux"), reason="Cannot download file")
    def test_remove_additional_stopwords(self):
        tokens = ["my", "name", "is", "elf", "who", "are", "you"]
        actual = remove_nltk_stopwords(tokens, ["elf"])
        expected = ["name"]
        assert actual == expected, "additional stopwords not being removed correctly"


class TestInitialiseUpdateStopwords:
    @pytest.mark.skipif(sys.platform.startswith("linux"), reason="Cannot download file")
    def test_add_word_to_stopwords(self):
        additional_words = ["elf", "santa"]
        new_stopwords = initialise_update_stopwords(additional_words)
        actual = [word in new_stopwords for word in additional_words]
        assert all(actual), "new words not added to stopwords"


class TestInitialiseNLTKStopwords:
    @pytest.mark.skipif(sys.platform.startswith("linux"), reason="Cannot download file")
    def test_return_stopwords_list(self):
        stopwords = _initialise_nltk_stopwords()
        assert isinstance(stopwords, list), "Did not return a list of stopwords"

    @pytest.mark.skipif(sys.platform.startswith("linux"), reason="Cannot download file")
    def test_key_stopwords(self):
        stopwords = _initialise_nltk_stopwords()
        expected = ["i", "we", "you"]
        actual = [word in stopwords for word in expected]
        assert all(actual), "expected key words missing from stopwords"


class TestUpdateNLTKStopwords:
    @pytest.mark.skipif(sys.platform.startswith("linux"), reason="Cannot download file")
    def test_add_word_to_stopwords(self):
        stopwords = _initialise_nltk_stopwords()
        additional_words = ["elf", "santa"]
        new_stopwords = _update_nltk_stopwords(stopwords, additional_words)
        actual = [word in new_stopwords for word in additional_words]
        assert all(actual), "new words not added to stopwords"


class TestRejoinTokens:
    def test_region_tokens(self):
        tokens = ["my", "name", "is", "elf"]
        actual = rejoin_tokens(tokens)
        expected = "my name is elf"
        assert actual == expected, "did not rejoin tokens correctly"


class TestExtractFeatureCount:
    def test_feature_count(self):
        data = Series(["My name is elf"])
        expected = DataFrame([[1, 1, 1, 1]], columns=("elf", "is", "my", "name"))
        actual = extract_feature_count(data)
        assert all(expected == actual), "Does not match expected output"

    def test_remove_stopwords(self):
        stopwords = ["is", "my"]
        data = Series(["My name is elf"])
        actual = extract_feature_count(data, stop_words=stopwords)
        expected = DataFrame([[1, 1]], columns=("elf", "name"))
        assert all(expected == actual), "Does not remove stopwords"

    def test_ngrams(self):
        data = Series(["My name is elf"])
        actual = extract_feature_count(data, ngram_range=(1, 2))
        expected = DataFrame(
            [repeat(1, 7)],
            columns=["elf", "is", "is elf", "my", "my name", "name", "name is"],
        )
        assert all(expected == actual), "Does not handle ngrams"


class testGetTotalFeatureCount:
    def test_get_total_feature_count(self):
        df = DataFrame(
            [[1, 1, 1, 1, 0], [0, 1, 1, 1, 1]],
            columns=["elf", "is", "my", "name", "santa"],
        )
        expected = DataFrame(
            [1, 2, 2, 2, 1], columns=["elf", "is", "my", "name", "santa"]
        )
        actual = get_total_feature_count(df)
        assert all(expected == actual), "Does not correctly sum total features"


if __name__ == "__main__":
    unittest.main()

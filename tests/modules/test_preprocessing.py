import numpy as np
import pytest
from nltk.corpus import stopwords as sw
from pandas import Series

from src.modules import preprocessing as prep


class TestLoadConfig:
    def test_input_type_error(self):
        """test for assertion error"""
        bad_input = 123
        with pytest.raises(Exception) as e_info:
            prep.load_config(bad_input)
        assert (
            str(e_info.value) == "filepath must be a string"
        ), "Did not raise TypeError"

    def test_input_file_not_found(self):
        """test for error feedback on file not found"""
        bad_input = "src/superman.yaml"
        with pytest.raises(Exception) as e_info:
            prep.load_config(bad_input)
        assert (
            str(e_info.value.args[1]) == "No such file or directory"
        ), "Did not raise file not found error"

    def test_return_dict(self):
        assert (
            type(prep.load_config("src/config.yaml")) is dict
        ), "output is not <class 'dict'>"


class TestLoadJson:
    def test_input_type_error(self):
        """test for assertion error"""
        bad_input = 123
        with pytest.raises(Exception) as e_info:
            prep.load_json(bad_input)
        assert (
            str(e_info.value) == "filepath must be a string"
        ), "Did not raise TypeError"

    def test_input_file_not_found(self):
        """test for error feedback on file not found"""
        bad_input = "src/superman.json"
        with pytest.raises(Exception) as e_info:
            prep.load_json(bad_input)
        assert (
            str(e_info.value.args[1]) == "No such file or directory"
        ), "Did not raise file not found error"

    def test_return_dict(self):
        assert (
            type(prep.load_json("src/spelling_words.json")) is dict
        ), "output is not <class 'dict'>"


class TestPrependStrToListObjects:
    def test_prepend_str_to_list_objects(self):
        list_x = [1, 2, 3]
        string_x = "qu_"
        expected = ["qu_1", "qu_2", "qu_3"]
        actual = prep.prepend_str_to_list_objects(list_x, string_x)
        assert actual == expected, "did not correctly prepend string to list objects"


class TestGetResponseLength:
    def test_get_response_length(self):
        series = Series(["hello", "world"])
        expected = Series([5, 5])
        actual = prep.get_response_length(series)
        assert all(
            expected == actual
        ), "Did not correctly identify the response lengths"


class TestRemoveBlankRows:
    def test_blank_rows_removed(self):
        """test that blank rows are removed"""
        series_with_empty_row = Series([1.0, "", 3.0])
        expected_outcome = Series([1.0, 3.0])
        actual = prep.remove_blank_rows(series_with_empty_row)
        actual_reindexed = actual.reset_index(drop=True)
        assert all(
            actual_reindexed == expected_outcome
        ), "function does not remove blank rows"

    def test_return_series(self):
        """test that function returns a Series"""
        actual = prep.remove_blank_rows(Series([1.0, "", 3.0]))
        assert (
            type(actual) is Series
        ), "output is not <class 'pandas.core.series.Series'>"


class TestReplaceBlanks:
    def test_blank_replacement(self):
        """test replace blanks with NaN"""
        series_with_empty_row = Series([1.0, "", 3.0])
        actual = prep._replace_blanks(series_with_empty_row)
        assert np.isnan(actual[1]), "did not replace blank with NaN"

    def test_return_series(self):
        """test that function returns a Series"""
        actual = prep.remove_blank_rows(Series([1.0, "", 3.0]))
        assert (
            type(actual) is Series
        ), "output is not <class 'pandas.core.series.Series'>"


class TestShortenTokens:
    def test_shorten_tokens_lemmatize(self):
        words = Series([["houses"]])
        expected = Series([["house"]])
        actual = prep.shorten_tokens(words, True)
        assert all(actual == expected), "Did not lemmatize correctly over series"

    def test_shorten_tokens_stemmer(self):
        words = Series([["houses"]])
        expected = Series([["hous"]])
        actual = prep.shorten_tokens(words, False)
        assert all(actual == expected), "Did not stemmer correctly over series"


class TestStemmer:
    def test_stemmer(self):
        word_list = ["flying", "fly", "Beautiful", "Beauty"]
        actual = prep.stemmer(word_list)
        expected = ["fli", "fli", "beauti", "beauti"]
        assert actual == expected, "words are not being stemmed correctly"


class TestLemmatizer:
    def test_lemmatization(self):
        word_list = ["house", "houses", "housing"]
        actual = prep.lemmatizer(word_list)
        expected = ["house", "house", "housing"]
        assert actual == expected, "words are not being lemmatized correctly"


class TestRemoveNLTKStopwords:
    def test_remove_standard_stopwords(self):
        tokens = ["my", "name", "is", "elf", "who", "are", "you"]
        actual = prep.remove_nltk_stopwords(tokens)
        expected = ["name", "elf"]
        assert actual == expected, "core stopwords not being removed correctly"

    def test_remove_additional_stopwords(self):
        tokens = ["my", "name", "is", "elf", "who", "are", "you"]
        actual = prep.remove_nltk_stopwords(tokens, ["elf"])
        expected = ["name"]
        assert actual == expected, "additional stopwords not being removed correctly"


class TestInitialiseUpdateStopwords:
    def test_add_word_to_stopwords(self):
        additional_words = ["elf", "santa"]
        new_stopwords = prep.initialise_update_stopwords(additional_words)
        actual = [word in new_stopwords for word in additional_words]
        assert all(actual), "new words not added to stopwords"


class TestUpdateNLTKStopwords:
    def test_add_word_to_stopwords(self):
        prep._initialise_nltk_component("corpora/stopwords", "stopwords")
        stopwords = sw.words("english")
        additional_words = ["elf", "santa"]
        new_stopwords = prep._update_nltk_stopwords(stopwords, additional_words)
        actual = [word in new_stopwords for word in additional_words]
        assert all(actual), "new words not added to stopwords"


class TestRejoinTokens:
    def test_region_tokens(self):
        tokens = ["my", "name", "is", "elf"]
        actual = prep.rejoin_tokens(tokens)
        expected = "my name is elf"
        assert actual == expected, "did not rejoin tokens correctly"

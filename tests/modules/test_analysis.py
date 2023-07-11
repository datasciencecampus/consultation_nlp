from itertools import repeat

from pandas import DataFrame, Series

from src.modules.analysis import (
    extract_feature_count,
    get_total_feature_count,
    retrieve_named_entities,
)


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


class TestGetTotalFeatureCount:
    def test_get_total_feature_count(self):
        df = DataFrame(
            [[1, 1, 1, 1, 0], [0, 1, 1, 1, 1]],
            columns=["elf", "is", "my", "name", "santa"],
        )
        expected = DataFrame(
            [[1, 2, 2, 2, 1]], columns=["elf", "is", "my", "name", "santa"]
        )
        actual = get_total_feature_count(df)
        assert all(expected == actual), "Does not correctly sum total features"


class TestRetrieveNamedEntities:
    def test_retrieve_named_entities(self):
        test_data = Series(
            [
                "The ONS has just released an article on the UK Government's policy.",
                "my own care for nothing",
                "Hollywood actors now have their own statue",
            ]
        )
        actual = retrieve_named_entities(test_data)
        expected = [["ONS", "the UK Government's"], [], ["Hollywood"]]
        assert actual == expected, "Did not successfully retrieve named entities"

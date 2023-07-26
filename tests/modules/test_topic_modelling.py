import os
import unittest

from pandas import DataFrame, Series
from sklearn.feature_extraction.text import CountVectorizer

from src.modules import topic_modelling as topic
from src.modules import utils


class TestTopicModel:
    def test_topic_model(self):
        model = "lda"
        question = "test"
        original_series = Series(["Hello", None, "Friend"]).dropna()
        model_series = Series(["Hello", "Friend"])
        stopwords = ["he"]
        config = {
            "models": {
                "test": {
                    "max_features": None,
                    "ngram_range": (1, 2),
                    "min_df": 1,
                    "max_df": 0.9,
                    "n_topics": 2,
                    "n_top_words": 2,
                    "max_iter": {"lda": 25, "nmf": 1000},
                    "lowercase": True,
                    "topic_labels": {"lda": None, "nmf": ["topic_k", "topic_x"]},
                }
            }
        }
        datestamp = utils._get_datestamp()
        topic.topic_model(
            model, question, original_series, model_series, stopwords, config
        )
        common_words_path = f"data/outputs/{datestamp}_test_lda_common_words.jpeg"
        top_words_path = f"data/outputs/{datestamp}_test_lda_top_words_by_topic.jpeg"
        topic_score_path = f"data/outputs/{datestamp}_test_lda_topic_score.csv"

        try:
            assert os.path.isfile(
                common_words_path
            ), "{qu_n}_{model}_common_words.jpeg file not found"
            assert os.path.isfile(
                top_words_path
            ), "{qu_n}_{model}_top_words_by_topic.jpeg file not found"
            assert os.path.isfile(
                topic_score_path
            ), "{qu_n}_{model}_topic_score.csv file not found"
        finally:
            [
                os.remove(path_x)
                for path_x in [common_words_path, top_words_path, topic_score_path]
                if os.path.exists(path_x)
            ]


class TestFitVectorizerToDF:
    def test_fit_vectorizer_to_df(self):
        vectorizer = CountVectorizer()
        series = Series(["hello", "friend"])
        fitted = vectorizer.fit_transform(series)
        expected = DataFrame({"friend": [0, 1], "hello": [1, 0]})
        actual = topic._fit_vectorizer_to_df(fitted, vectorizer)
        assert all(actual == expected), "Did not fit correctly to DataFrame"


class TestColumnwiseSum:
    def test_columnwise_sum(self):
        df = DataFrame({"values": [1, 2, 3, 4, 5]})
        expected = DataFrame({"index": ["values"], "sum": [15]})
        actual = topic._columnwise_sum(df)
        assert all(actual == expected), "Did not sum columns correctly"


class TestGenerateTopicLabels(unittest.TestCase):
    def test_topic_labels_is_none(self):
        topic_labels = None
        n_topics = 2
        actual = topic._generate_topic_labels(n_topics, topic_labels)
        expected = ["Topic_1", "Topic_2"]
        assert actual == expected, "Topic labels did not match expected"

    def test_topic_labels_preset(self):
        topic_labels = ["My Topic", "Your Topic"]
        n_topics = 2
        actual = topic._generate_topic_labels(n_topics, topic_labels)
        expected = ["My Topic", "Your Topic"]
        assert actual == expected, "Topic labels did not match expected"

    def test_raise_attribute_error(self):
        topic_labels = ["One"]
        n_topics = 2
        with (self.assertRaises(Exception)) as context:
            topic._generate_topic_labels(n_topics, topic_labels)
        self.assertTrue("Does not raise an AttributeError", context.exception)


class TestGetNColumnsAndNRows(unittest.TestCase):
    def test_raise_value_error(self):
        n_topics = 0
        with (self.assertRaises(Exception)) as context:
            topic._get_n_columns_and_n_rows(n_topics)
        self.assertTrue("Does not raise a value error", context.exception)

    def test_n_topics_5_or_less(self):
        n_topics = 4
        actual = topic._get_n_columns_and_n_rows(n_topics)
        expected = (1, 4)
        assert actual == expected, "Did not produce the correct number of rows/columns"

    def test_n_topics_above_5_with_factor(self):
        n_topics = 24
        actual = topic._get_n_columns_and_n_rows(n_topics)
        expected = (6, 4)
        assert actual == expected, "Did not produce the correct number of rows/columns"

    def test_n_topics_above_5_without_factor(self):
        n_topics = 23
        actual = topic._get_n_columns_and_n_rows(n_topics)
        expected = (6, 4)
        assert actual == expected, "Did not produce the correct number of rows/columns"


class TestGetFactors:
    def test_get_factors_of_4(self):
        test_input = 4
        actual = topic._get_factors(test_input)
        expected = [1, 2, 4]
        assert actual == expected, "Did not return the correct factors of 4"

    def test_get_factors_of_5(self):
        actual = topic._get_factors(5)
        expected = [1, 5]
        assert actual == expected, "Did not return the correct factors of 5"

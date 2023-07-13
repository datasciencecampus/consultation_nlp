import os
import unittest
from datetime import datetime as dt

import matplotlib.pyplot as plt

from src.modules.visualisation import (
    _generate_topic_labels,
    _get_factors,
    _get_fig_size,
    _get_n_columns_and_n_rows,
    save_figure,
)


class TestSaveFigure:
    def test_file_created(self):
        figure = plt.figure()
        name = "test"
        datestamp = dt.strftime(dt.now(), "%Y%m%d")
        filepath = f"data/outputs/{datestamp}_{name}.jpeg"
        save_figure(name, figure)
        assert os.path.isfile(filepath), "Did not save a file with correct filename"
        os.remove(filepath)


class TestGenerateTopicLabels(unittest.TestCase):
    def test_topic_labels_is_none(self):
        topic_labels = None
        n_topics = 2
        actual = _generate_topic_labels(n_topics, topic_labels)
        expected = ["Topic_1", "Topic_2"]
        assert actual == expected, "Topic labels did not match expected"

    def test_topic_labels_preset(self):
        topic_labels = ["My Topic", "Your Topic"]
        n_topics = 2
        actual = _generate_topic_labels(n_topics, topic_labels)
        expected = ["My Topic", "Your Topic"]
        assert actual == expected, "Topic labels did not match expected"

    def test_raise_attribute_error(self):
        topic_labels = ["One"]
        n_topics = 2
        with (self.assertRaises(Exception)) as context:
            _generate_topic_labels(n_topics, topic_labels)
        self.assertTrue("Does not raise an AttributeError", context.exception)


class TestGetNColumnsAndNRows(unittest.TestCase):
    def test_raise_value_error(self):
        n_topics = 0
        with (self.assertRaises(Exception)) as context:
            _get_n_columns_and_n_rows(n_topics)
        self.assertTrue("Does not raise a value error", context.exception)

    def test_n_topics_5_or_less(self):
        n_topics = 4
        actual = _get_n_columns_and_n_rows(n_topics)
        expected = (1, 4)
        assert actual == expected, "Did not produce the correct number of rows/columns"

    def test_n_topics_above_5_with_factor(self):
        n_topics = 24
        actual = _get_n_columns_and_n_rows(n_topics)
        expected = (6, 4)
        assert actual == expected, "Did not produce the correct number of rows/columns"

    def test_n_topics_above_5_without_factor(self):
        n_topics = 23
        actual = _get_n_columns_and_n_rows(n_topics)
        expected = (6, 4)
        assert actual == expected, "Did not produce the correct number of rows/columns"


class TestGetFactors:
    def test_get_factors_of_4(self):
        test_input = 4
        actual = _get_factors(test_input)
        expected = [1, 2, 4]
        assert actual == expected, "Did not return the correct factors of 4"

    def test_get_factors_of_5(self):
        actual = _get_factors(5)
        expected = [1, 5]
        assert actual == expected, "Did not return the correct factors of 5"


class TestGetFigSize:
    def test_get_fig_size(self):
        columns = 2
        rows = 4
        actual = _get_fig_size(columns, rows)
        expected_width = 12
        expected_height = 27
        expected_result = (expected_width, expected_height)
        assert actual == expected_result, "expected height or width is not correct"

import os
import sys
import unittest
from datetime import datetime as dt

import pytest
from matplotlib import pyplot as plt

from src.modules import utils


class TestTrimEnds:
    def test_trim_ends(self):
        test_string = " hello world "
        expected = "hello world"
        actual = utils._trim_ends(test_string)
        assert expected == actual, "Did not correctly trim ends"


class TestGetDatestamp(unittest.TestCase):
    def test_get_datestamp_is_string(self):
        date_stamp = utils._get_datestamp()
        assert type(date_stamp) is str, "datestamp is not a string"

    def test_get_datestamp_is_date(self):
        datestamp = utils._get_datestamp()
        try:
            dt.strptime(datestamp, "%Y%m%d")
        except ValueError:
            raised = True
            self.assertFalse(
                raised, "Exception raised when trying to convert date string to date"
            )


class TestSaveFigure:
    @pytest.mark.skipif(
        sys.platform.startswith("linux"), reason="Unknown error during CI"
    )
    def test_file_created(self):
        figure = plt.figure()
        name = "test"
        datestamp = dt.strftime(dt.now(), "%Y%m%d")
        filepath = f"data/outputs/{datestamp}_{name}.jpeg"
        utils._save_figure(name, figure)
        assert os.path.isfile(filepath), "Did not save a file with correct filename"
        os.remove(filepath)


class TestGetFigSize:
    def test_get_fig_size(self):
        columns = 2
        rows = 4
        actual = utils._get_fig_size(columns, rows)
        expected_width = 12
        expected_height = 27
        expected_result = (expected_width, expected_height)
        assert actual == expected_result, "expected height or width is not correct"

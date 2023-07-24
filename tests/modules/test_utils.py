import os
import sys
from datetime import datetime as dt

import pytest
from matplotlib import pyplot as plt

from src.modules import utils


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

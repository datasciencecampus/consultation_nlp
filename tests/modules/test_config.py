import pytest

from src.modules.config import Config


class TestLoadConfig:
    def test_input_type_error(self):
        """test for assertion error"""
        config = Config()
        bad_input = 123
        with pytest.raises(Exception) as e_info:
            config._load_config(bad_input)
        assert (
            str(e_info.value) == "filepath must be a string"
        ), "Did not raise TypeError"

    def test_input_file_not_found(self):
        """test for error feedback on file not found"""
        config = Config()
        bad_input = "src/superman.yaml"
        with pytest.raises(Exception) as e_info:
            config._load_config(bad_input)
        assert (
            str(e_info.value.args[1]) == "No such file or directory"
        ), "Did not raise file not found error"

    def test_return_dict(self):
        config = Config()
        assert (
            type(config._load_config("src/general.yaml")) is dict
        ), "output is not <class 'dict'>"


class TestLoadJson:
    def test_input_type_error(self):
        """test for assertion error"""
        config = Config()
        bad_input = 123
        with pytest.raises(Exception) as e_info:
            config._load_json(bad_input)
        assert (
            str(e_info.value) == "filepath must be a string"
        ), "Did not raise TypeError"

    def test_input_file_not_found(self):
        """test for error feedback on file not found"""
        config = Config()
        bad_input = "src/superman.json"
        with pytest.raises(Exception) as e_info:
            config._load_json(bad_input)
        assert (
            str(e_info.value.args[1]) == "No such file or directory"
        ), "Did not raise file not found error"

    def test_return_dict(self):
        config = Config()
        assert (
            type(config._load_json("src/spelling.json")) is dict
        ), "output is not <class 'dict'>"

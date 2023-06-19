import pytest

from src.processing import preprocessing

class TestLoadConfig:
    def test_input_type_error(self):
        '''test for correct feedback on user 
        input type error'''
        bad_input = 123
        with pytest.raises(Exception) as e_info:
            preprocessing.load_config(bad_input)

    def test_input_file_not_found(self):
        '''test for error feedback on file not found'''
        bad_input = "src/superman.yaml"
        with pytest.raises(Exception) as e_info:
            preprocessing.load_config(bad_input)

    def test_return_dict(self):
        config = preprocessing.load_config('src/config.yaml')
        assert type(config) is dict, "output is not <class 'dict'>"

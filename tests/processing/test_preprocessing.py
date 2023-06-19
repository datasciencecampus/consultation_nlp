import pytest

from src.processing import preprocessing

class TestLoadConfig:
    def test_input_type_error(self):
        '''test for assertion error '''
        bad_input = "123"
        with pytest.raises(Exception) as e_info:
            preprocessing.load_config(bad_input)
        assert str(e_info.value) == "filepath must be a string",\
              "Did not raise TypeError"
    

    def test_input_file_not_found(self):
        '''test for error feedback on file not found'''
        bad_input = "src/superman.yaml"
        with pytest.raises(Exception) as e_info:
            preprocessing.load_config(bad_input)
        assert str(e_info.value.args[1]) == 'No such file or directory',\
              "Did not raise file not found error"

    def test_return_dict(self):
        assert type(preprocessing.load_config('src/config.yaml')) is dict,\
              "output is not <class 'dict'>"

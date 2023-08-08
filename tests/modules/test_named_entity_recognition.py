import sys

import pytest
from pandas import Series

from src.modules import named_entity_recognition as ner


class TestRetrieveNamedEntities:
    @pytest.mark.skipif(
        sys.platform.startswith("linux"), reason="Unknown error during CI"
    )
    def test_retrieve_named_entities(self):
        test_data = Series(
            [
                "The ONS has just released an article on the UK Government's policy.",
                "my own care for nothing",
                "Hollywood actors now have their own statue",
            ]
        )
        actual = ner.retrieve_named_entities(test_data)
        expected = [["ONS", "the UK Government's"], [], ["Hollywood"]]
        assert actual == expected, "Did not successfully retrieve named entities"

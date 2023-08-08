from src.modules import clean_string as clean


class TestCleanString:
    def test_clean_string(self):
        string_x = " hello 2,023...& don?t ask how the-heck are you\\?"
        expected = "hello 2023 and dont ask how the-heck are you ?"
        actual = clean.clean_string(string_x)
        assert actual == expected, "did not correctly clean the string"


class TestRemoveNumberPunctuation:
    def test_remove_number_punctutation(self):
        string_x = "0:0,0.0"
        expected = "0000"
        actual = clean._remove_number_punctuation(string_x)
        assert actual == expected, "did not correctly remove number punctuation"

    def test_not_remove_other_punctuation(self):
        string_x = "hello, world."
        expected = "hello, world."
        actual = clean._remove_number_punctuation(string_x)
        assert actual == expected, "incorrectly removed other types of punctutation"


class TestRemoveElipsis:
    def test_remove_elipsis(self):
        string_x = "hello...world"
        expected = "hello world"
        actual = clean._remove_elipsis(string_x)
        assert actual == expected, "did not remove elipsis"


class TestReplaceAmpersandPlus:
    def test_replace_ampersand_plus(self):
        string_x = "hello & + world"
        expected = "hello  and   and  world"
        actual = clean._replace_ampersand_plus(string_x)
        assert actual == expected, "did not replace ampersand and plus with ' and '"


class TestRemoveDropLines:
    def test_remove_drop_lines(self):
        string_x = "hello \n world"
        expected = "hello   world"
        actual = clean._remove_drop_lines(string_x)
        assert actual == expected, "did not remove drop lines"


class TestRemoveEscapeChars:
    def test_remove_escape_chars(self):
        string_x = "hello \\ world"
        expected = "hello   world"
        actual = clean._remove_escape_chars(string_x)
        assert actual == expected, "did not remove escape characters"


class TestRemoveNonWordJoiners:
    def test_leave_word_joiners(self):
        string_x = "h-e_l'l:o world"
        expected = "h-e_l'l:o world"
        actual = clean._remove_non_word_joiners(string_x)
        assert actual == expected, "did not leave the word joining punctuation"

    def test_remove_non_word_joiners(self):
        string_x = "h?e>l)l*o world"
        expected = "hello world"
        actual = clean._remove_non_word_joiners(string_x)
        assert (
            actual == expected
        ), "did not remove punctuation which doesn't normally join words"

    def test_leave_external_punctuation(self):
        string_x = ".hello world!"
        expected = ".hello world!"
        actual = clean._remove_non_word_joiners(string_x)
        assert actual == expected, "did not leave puctuation outside of string"


class TestRemoveDoubleSpace:
    def test_remove_double_space(self):
        string_x = "hello  world"
        expected = "hello world"
        actual = clean._remove_double_space(string_x)
        assert actual == expected, "Did not remove double space"

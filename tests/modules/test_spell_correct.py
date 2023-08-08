from pandas import Series
from spellchecker import SpellChecker

from src.modules import spell_correct as spell


class TestFindWordReplacements:
    def test_find_word_replacements(self):
        series_x = Series(["ehllo world", "hello world"])
        spellchecker = SpellChecker()
        expected = Series([{"ehllo": "hello"}, {}])
        actual = spell.find_word_replacements(series_x, spellchecker)
        assert all(actual == expected), "did not find correct suggestions"


class TestFindWordReplacementsString:
    def test_find_word_replacements(self):
        string_x = "ehllo world"
        spellchecker = SpellChecker()
        expected = {"ehllo": "hello"}
        actual = spell._find_word_replacements_string(string_x, spellchecker)
        assert actual == expected, "did not find correct suggestion"


class TestReplaceWords:
    def test_replace_words(self):
        series_x = Series(["ehllo world", "hello world"])
        replacement_x = Series([{"ehllo": "hello"}, {}])
        expected = Series(["hello world", "hello world"])
        actual = spell.replace_words(series_x, replacement_x)
        assert all(actual == expected), "did not correctly replace words"


class TestReplaceWordsString:
    def test_replace_words_string(self):
        string_x = "ehllo world"
        replacement_x = {"ehllo": "hello"}
        expected = "hello world"
        actual = spell._replace_words_string(string_x, replacement_x)
        assert actual == expected, "did not correctly replace words"


class TestUpdateSpellDictionary:
    def test_update_spell_dictionary(self):
        additonal_words = {"Monfklop": 1}
        before_spellchecker = SpellChecker()
        before = before_spellchecker.unknown(["Monfklop"])
        after_spellchecker = spell.update_spell_dictionary(additonal_words)
        after = after_spellchecker.unknown(["Monfklop"])
        assert len(after) == 0 and before == {
            "monfklop"
        }, "dictionary not updated correctly"


class TestTokenizeWords:
    def test_tokenize_words(self):
        string_x = "hello world"
        expected = ["hello", "world"]
        actual = spell._tokenize_words(string_x)
        assert actual == expected, "Did not tokenize correctly"


class TestRemovePunctuation:
    def test_remove_punctuation(self):
        string_x = "!hello, ,world!"
        expected = "hello   world"
        actual = spell.remove_punctuation(string_x)
        assert actual == expected, "did not correctly remove punctuation"


class TestRemovePunctuationSpaceBefore:
    def test_remove_punctuation_space_before(self):
        string_x = "hello ,world"
        expected = "hello  world"
        actual = spell._remove_punctuation_space_before(string_x)
        assert actual == expected, "did not remove punctuation after a space"


class TestRemovePunctuationSpaceAfter:
    def test_remove_punctuation_space_after(self):
        string_x = "hello, world"
        expected = "hello  world"
        actual = spell._remove_punctuation_space_after(string_x)
        assert actual == expected, "did not remove punctuation before a space"


class TestRemovePunctuationSentenceStart:
    def test_remove_punctuation_sentence_start(self):
        string_x = "!hello world"
        expected = " hello world"
        actual = spell._remove_punctuation_sentence_start(string_x)
        assert actual == expected, "did not remove starting punctuation"


class TestRemovePunctuationSentenceEnd:
    def test_remove_punctuation_sentence_end(self):
        string_x = "hello world!"
        expected = "hello world "
        actual = spell._remove_punctuation_sentence_end(string_x)
        assert actual == expected, "did not remove ending punctuation"

import re
from importlib import reload

from pandas import DataFrame, Series

# from src.modules import preprocessing as prep
from src.modules import streamlit as stream

reload(stream)


class TestGetNTopWords:
    def test_get_n_top_words(self):

        test_df = DataFrame(
            {
                "topic_1_word_importance": [0, 1, 2],
                "topic_2_word_importance": [0, 0, 0],
                "word": ["alpha", "bravo", "charlie"],
            }
        )
        actual = stream.get_top_n_words(topic_words=test_df, n=2, topic_name="Topic 1")
        expected = Series(["bravo", "charlie"], index=[1, 2])
        assert all(actual == expected)


class TestIdentifyDominantTopics:
    def test_identify_dominant_topics(self):
        topic_names_snake = ["topic_1", "topic_2", "topic_3"]
        test_df = DataFrame(
            {
                "word": ["alpha", "bravo", "charlie"],
                "topic_1": [0, 1, 2],
                "topic_2": [2, 3, 4],
                "topic_3": [3, 2, 1],
            }
        )
        actual = stream.identify_dominant_topics(
            topic_words=test_df, topic_names_snake=topic_names_snake
        )
        expected = DataFrame(
            {
                "word": ["alpha", "bravo", "charlie"],
                "variable": ["topic_3", "topic_2", "topic_2"],
            }
        )
        assert all(actual == expected)


class TestSnakeCase:
    def test_snake_case(self):
        actual = stream.snake_case("This string")
        expected = "this_string"
        assert actual == expected


class TestGetNTopicSamples:
    def test_get_n_topic_samples(self):
        test_df = DataFrame(
            {
                "responses": ["hello word", "world hello", "hello hello"],
                "topic_1": [0, 2, 1],
            }
        )
        actual = stream.get_n_topic_samples(
            text_with_topic_df=test_df, topic_name="Topic_1", n=2
        )
        expected = DataFrame(
            {"responses": ["world hello", "hello hello"], "topic_1": [2, 1]}
        )
        assert all(actual == expected)


class TestGetResponseNo:
    def test_get_response_no(self):
        test_df = DataFrame(
            {
                "responses": ["hello word", "world hello", "hello hello"],
                "index": [455, 12, 11],
            }
        )
        actual = stream.get_response_no(topic_sample=test_df, position=1)
        expected = "Response 12"
        assert actual == expected


class TestGenerateTopScores:
    def test_generate_top_scores(self):
        test_df = DataFrame(
            {
                "responses": ["hello word", "world hello", "hello hello"],
                "index": [53, 22, 12],
                "topic_1": [0.1, 0.3, 0.01],
                "topic_2": [0.12, 0.22, 0.32],
            }
        )
        actual = stream.generate_top_scores(
            topic_sample=test_df, topic_name="Topic 1", position=1
        )
        expected = "(Topic 1; Score: 30.0%)   (Topic 2; Score: 22.0%)"
        assert actual == expected


class TestGetHexColors:
    def test_get_hex_colors_is_hex(self):
        actual = stream.get_hex_colors(n_colors=1)
        assert re.match(r"#[a-zA-Z0-9]{6}", actual[0]), "does not match hex pattern"

    def test_get_hex_colors_n_returns(self):
        actual = stream.get_hex_colors(n_colors=4)
        assert len(actual) == 4
        actual = stream.get_hex_colors(n_colors=2)
        assert len(actual) == 2


class TestGetFormattingTuple:
    def test_get_formatting_tuple(self):
        test_dominant_topics = DataFrame(
            {"variable": ["topic_1", "topic_2"]}, index=["hello", "world"]
        )
        test_topic_color_dict = {"Topic 1": "#000000", "Topic 2": "#999999"}
        actual = stream.create_formatting_tuple(
            dominant_topics=test_dominant_topics,
            word="hello",
            topic_color_dict=test_topic_color_dict,
        )

        expected = ("hello", "Topic 1", "#000000")
        assert actual == expected


class TestCreateWordStopWordCombos:
    def test_create_word_stopword_combo(self):
        test_stopwords = ["he", "her"]
        test_words = Series(["hello world", "hello"], index=[21, 42])
        actual = stream.create_word_stopword_combos(
            top_n_words=test_words, stopwords=test_stopwords
        )
        expected = ["hello he world", "hello her world", "hello world", "hello"]
        assert actual == expected


class TestInsertTuple:
    def test_insert_tuple(self):
        test_split_string = [
            "hello my name",
            "['word', 'Topic 1', '#000000']",
            "is world",
        ]
        actual = stream.insert_tuple(split_string=test_split_string)
        expected = ["hello my name", ("word", "Topic 1", "#000000"), "is world"]
        assert actual == expected


class TestAddLabelFormatting:
    def test_add_label_formatting(self):
        test_df = DataFrame(
            {"responses": ["hello world how are you", "my name is world"]}
        )
        replacement_dict = {"world": "['world', 'Topic 1', '#000000']"}
        actual = stream.add_label_formatting(
            replacement_dict=replacement_dict, topic_sample=test_df
        )
        expected = [
            ["hello ['world', 'Topic 1', '#000000'] how are you"],
            ["my name is ['world', 'Topic 1', '#000000']"],
        ]
        assert actual == expected


class TestGetSingleTopicColor:
    def test_get_single_topic_color(self):
        test_topic_names = ["Topic 1", "Topic 2"]
        topic_1 = stream.get_single_topic_color(
            topic_names=test_topic_names, topic_name="Topic 1"
        )
        topic_2 = stream.get_single_topic_color(
            topic_names=test_topic_names, topic_name="Topic 2"
        )
        assert topic_1 != topic_2


# class TestSingleTopicFormatting:
#     def test_single_topic_formatting(self):
#         top_n_words = Series(["hello", "world"])
#         topic_sample = DataFrame({"responses": ["hello world how are you",
#                                                 "hi world you are my oyster",
#                                                 "hello my world how are you"],
#                                    "topic_1": [0.9, 0.3, 0.2],
#                                    "topic_2": [0.01, 0.8,0.6]})
#         topic_name = "Topic 1"
#         topic_color = "#000000"
#         stopwords = prep.initialise_update_stopwords(["he"])
#         actual = stream.single_topic_formatting(top_n_words= top_n_words,
#                                        topic_sample = topic_sample,
#                                        topic_name= "Topic 1",
#                                        topic_color = "#000000",
#                                        stopwords=stopwords)
#         expected = [[('hello', 'Topic 1', '#000000'), \
#                      ('world', 'Topic 1', '#000000'), 'how are you'],
#                      ["hi ['world', 'Topic 1', '#000000']", 'you are my oyster']]
#         assert actual == expected

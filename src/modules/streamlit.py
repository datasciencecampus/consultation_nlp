import re

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from nltk.tokenize import word_tokenize
from pandas import DataFrame, Series


def plot_words_by_topic_bar(topic_words: DataFrame, topic_name: str) -> Figure:
    """plot words by topic bar chart

    Parameters
    ----------
    topic_words:DataFrame
        a dataframe containing the topic words by weight and frequency
    topic_name:str
        the name of the topic

    Returns
    -------
    Figure
        a bar chart of word frequency and weights
    """
    fig = plt.figure(figsize=(5, 6))
    topic_name_snake = snake_case(topic_name)
    topic_words = topic_words.sort_values(f"{topic_name_snake}_word_importance").tail(
        35
    )
    plt.barh(topic_words["word"], topic_words[topic_name_snake], alpha=0.6, color="red")
    plt.barh(
        topic_words["word"], topic_words["word_frequency"], alpha=0.4, color="grey"
    )
    plt.title(topic_name)
    return fig


def get_top_n_words(topic_words: DataFrame, n: int, topic_name: str) -> Series:
    """get the top n number of word from the topic_words dataframe

    Parameters
    ----------
    topic_words:DataFrame
        a dataframe containing the topic words by weight and frequency
    n:int
        number of top words to select
    topic_name:str
        the name of the topic

    Returns
    -------
    Series
        top n words of a particular topic
    """
    topic_name_snake = snake_case(topic_name)
    top_n_words = (
        topic_words.sort_values(f"{topic_name_snake}_word_importance").tail(n).word
    )
    return top_n_words


def identify_dominant_topics(
    topic_words: DataFrame, topic_names_snake: list
) -> DataFrame:
    """Identify the dominant topic associated with each word

    Parameters
    ----------
    topic_words:DataFrame
        a dataframe containing the topic words by weight and frequency
    topic_names_snake:list
        topic names in snake case

    Returns
    -------
    DataFrame
        a dataframe of words and their associated topics"""
    long_topics = topic_words.melt("word", value_vars=topic_names_snake)
    dominant_values = long_topics.groupby("word").max("value").reset_index()
    dominant_topics = pd.merge(dominant_values, long_topics, "left").drop(
        "value", axis=1
    )
    return dominant_topics


def snake_case(string: str) -> str:
    """transform string to snake case

    Parameters
    ----------
    string:str
        string to transform

    Returns
    -------
    str
        string in snake case"""
    return re.sub(" ", "_", string).lower()


def get_n_topic_samples(
    text_with_topic_df: DataFrame, topic_name: str, n: int
) -> DataFrame:
    """get n number of topic samples from text with topic dataframe

    Parameters
    ----------
    text_with_topic_df: DataFrame
        dataframe containing responses and scores for each topic
    topic_name:str
        topic name to extract samples for
    n:int
        number of samples to extract

    Returns
    -------
    DataFrame
        trimmed dataframe sorted by a topic with n rows"""
    topic_sample = (
        text_with_topic_df.sort_values(snake_case(topic_name), ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    return topic_sample


def get_response_no(topic_sample: DataFrame, position: int) -> str:
    """get the index number of the response

    Parameters
    ----------
    topic_sample:DataFrame
        sample of text sorted by a topic with a few number of rows
    position:int
        the row number within the dataframe to retrieve the index from

    Returns
    -------
    str
        string containing the original index number of the position row
    """
    topic_x = topic_sample.loc[position, :]
    response_no = topic_x["index"]
    return f"Response {response_no}"


def generate_top_scores(topic_sample: DataFrame, topic_name: str, position: int) -> str:
    """Generate the topic scores for the top two topics

    Parameters
    ----------
    topic_sample:DataFrame
        sample of text sorted by a topic with a few number of rows
    topic_name:str
        the name of the topic
    position:int
        the row number within the dataframe to retrieve the index from

    Returns
    -------
    str
        top two topic names and scores for 'position' row of data
    """
    topic_name_snake = snake_case(topic_name)
    topic_x = topic_sample.loc[position, :]
    top_topic_score = round(topic_x[topic_name_snake] * 100, 1)
    secondary_topic = (
        topic_x.drop(["index", topic_name_snake, "responses"]).sort_values().index[-1]
    )
    secondary_topic_capital = re.sub("_", " ", secondary_topic).capitalize()
    secondary_topic_score = round(topic_x[secondary_topic] * 100, 1)
    formatted_text_header = (
        f"({topic_name}; Score: {top_topic_score}%)"
        + "   "
        + f"({secondary_topic_capital}; Score: {secondary_topic_score}%)"
    )
    return formatted_text_header


def get_hex_colors(n_colors: int) -> str:
    """Get the hex color codes for n_colors number of colors

    Parameters
    ----------
    n_colors:int
        the number of colors to get codes for

    Returns
    -------
    list
        list of length n_colors hex codes"""
    return sns.color_palette(n_colors=n_colors).as_hex()


def create_formatting_tuple(
    dominant_topics: DataFrame, word: str, topic_color_dict: dict
) -> tuple:
    """create a formatting tuple for streamlit annotation

    Parameters
    ----------
    dominant_topics:DataFrame
        dataframe of words and their strongest associated topic
    word:str
        word to create tuple for
    topic_color_dict:dict
        dictionary of topics and their assigned colors

    Returns
    -------
    tuple
        formatting tuple containing word, topic, and color
    """
    topic_x = dominant_topics.loc[word, "variable"]
    topic_pretty = re.sub("_", " ", topic_x).capitalize()
    topic_color = topic_color_dict[topic_pretty]
    return (word, topic_pretty, topic_color)


def create_word_stopword_combos(top_n_words: Series, stopwords: list) -> list:
    """create combinations of top words with stopwords in between if length > 2

    Parameters
    ----------
    top_n_words:Series
        top n number of words with index numbers
    stopwords:list
        list of inconsequential words removed from corpus during cleaning

    Returns
    -------
    list
        returns a list of possible combinations for {word_a} {stopword} {word_b}
    """
    word_combos = [i for i in top_n_words if len(i.split()) > 1]
    word_stopword_combo = []
    for combo in word_combos:
        matches = re.findall(r"(?=(\b\w+\b\s\b\w+\b))", combo)
        for pair in matches:
            left, right = tuple(pair.split())
            word_stopword_combo.append([f"{left} {i} {right}" for i in stopwords])
    word_stopword_combo.append(top_n_words)
    unnested_stopword_combo = [num for elem in word_stopword_combo for num in elem]
    return unnested_stopword_combo


def insert_tuple(split_string: list) -> list:
    """replace string with streamlit annotate formatting tuple

    Parameters
    ----------
    split_string:list
        list of strings which have been split at tuples

    Returns
    -------
    list
        list of strings and formatting tuples
    """
    for n, i in enumerate(split_string):
        matcher = re.match(r"\['[\w\s]+',\s'\w+\s\d+',\s'#[a-zA-Z0-9]{6}'\]", i)
        if matcher:
            replacement_tuple = tuple(
                re.sub(r"\[|\]|'", "", matcher.group(0)).split(", ")
            )
            split_string[n] = replacement_tuple
    return split_string


def add_label_formatting(replacement_dict: dict, topic_sample: DataFrame) -> list:
    """add streamlit annotate label formatting within string

    Parameters
    ----------
    replacement_dict:dict
        dictionary of values to replace with their tuple replacements
    topic_sample: DataFrame
        sample of responses ordered by a particular topic

    Returns
    -------
    list
        list of strings and formatting tuples
    """
    formatted_text = []
    for sample in topic_sample["responses"]:
        for key, value in replacement_dict.items():
            sample = re.sub(rf"\s\b{key}\b", f" {value}", sample)
        formatted_text.append([sample])
    return formatted_text


def get_single_topic_color(topic_names: list, topic_name: str) -> str:
    """get the topic color for a single topic

    Parameters
    ----------
    topic_names:list
        list of topic names
    topic_name:str
        the topic name to select a color for

    Returns
    -------
    str
        hex code for the topic color"""
    n_topics = len(topic_names)
    topic_colors = get_hex_colors(n_topics).as_hex()
    topic_number = [n for n, i in enumerate(topic_names) if i == topic_name]
    topic_color = topic_colors[topic_number[0]]
    return topic_color


def single_topic_formatting(
    top_n_words: Series, topic_sample: DataFrame, topic_name: str, topic_color: str
) -> list:
    """Creates a streamlit annotate formatting setup for single topic

    Parameters
    ----------
    top_n_words:Series
        top n number of words with index numbers
    topic_sample: DataFrame
        sample of responses ordered by a particular topic
    topic_name: str
        name of the topic
    topic_color: str
        hex code for the topic

    Returns
    -------
    list
        a formatted list of strings and tuples
    """
    pattern_behind = r"[\s,](?=\['[\w\s]+',\s'\w+\s\d+',\s'#[a-zA-Z0-9]{6}'\])"
    pattern_ahead = r"(?<='#[a-zA-Z0-9]{6}'])[\s]"
    pattern_combined = "|".join([pattern_behind, pattern_ahead])
    top_n_words_x = top_n_words
    replacements = [[i, topic_name, topic_color] for i in list(top_n_words)]
    replacement_dict = dict(zip(top_n_words_x, replacements))
    initial_formatted = add_label_formatting(replacement_dict, topic_sample)
    for idx in range(len(initial_formatted)):
        split_string = re.split(pattern_combined, initial_formatted[idx][0])
        split_string = insert_tuple(split_string)
        initial_formatted[idx] = split_string
    return initial_formatted


def multitopic_formatting(
    dominant_topics: DataFrame, topic_sample: DataFrame, topic_names: list
) -> list:
    """Creates a streamlit annotate formatting setup for multiple topics

    Parameters
    ----------
    dominant_topics:DataFrame
        dataframe of words and their strongest associated topic
    topic_sample: DataFrame
        sample of responses ordered by a particular topic
    topic_names: list
        list of topic names

    Returns
    -------
    list
        a formatted list of strings and tuples
    """
    n_topics = len(topic_names)
    hex_colors = get_hex_colors(n_topics)
    topic_color_dict = dict(zip(topic_names, hex_colors))
    formatted_text = []
    dominant_topics_i = dominant_topics.set_index("word")
    for response in topic_sample["responses"]:
        formatted_response = []
        for word in word_tokenize(response):
            if word in list(set(dominant_topics_i.index)):
                formatting_tuple = create_formatting_tuple(
                    dominant_topics_i, word, topic_color_dict
                )
                formatted_response.append(formatting_tuple)
                formatted_response.append(" ")
            else:
                formatted_response.append(word + " ")
        formatted_text.append(formatted_response)
    return formatted_text

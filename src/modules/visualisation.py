import typing
from datetime import datetime as dt

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud


def create_wordcloud(text: str, name: str = "wordcloud") -> None:
    """generate a wordcloud with the given filename
    Parameters
    ----------
    text: str
        text for wordcloud
    filename: str
        the name and path you want to save the wordcloud to
    Returns:
    None (message to console on location of file)
    """
    wordcloud = WordCloud().generate(text)
    figure = plt.figure(figsize=(5, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    save_figure(name, figure)
    return None


def save_figure(name: str, fig: Figure) -> None:
    """save figure with datestamp
    Parameters
    ----------
    name: str
        name of the figure
    fig
        the figure object
    Returns
    -------
    None (message to console on location of file)
    """
    datestamp = dt.strftime(dt.now(), "%Y%m%d")
    filename = f"data/outputs/{datestamp}_{name}.jpeg"
    fig.savefig(filename, bbox_inches="tight")
    print(f"{name} plot saved as {filename}")
    return None


def plot_top_words(
    model: LatentDirichletAllocation,
    feature_names: list,
    n_topics: int,
    title: str,
    n_top_words: int = 10,
    topic_labels: list = None,
) -> None:
    """Plot topics by their most frequent words
    Parameters
    ----------
    model
        the lda model components
    feature_names:list
        a list of the most frequent words (from bag of words model)
    n_topics:int
        number of topics to include in the chart
    title:str
        the title for the chart
    n_top_words:int, (default = 10)
        the number of top words to include in each topic plot
    topic_labels:list, (default = None)
        a list of labels to override the existing labels
    Returns
    -------
    None (message to console on location of file)
    """
    topic_labels = _generate_topic_labels(n_topics, topic_labels)
    labelled_components = dict(zip(topic_labels, model.components_))
    rows, columns = _get_n_columns_and_n_rows(n_topics)
    fig, axes = plt.subplots(
        rows, columns, figsize=_get_fig_size(columns, rows), sharex=True
    )
    axes = axes.flatten()
    for number, (topic_label, component) in enumerate(labelled_components.items()):
        top_features_ind = component.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = component[top_features_ind]
        ax = axes[number]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(topic_label, fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)
    save_figure("lda_top_words", fig)
    return None


def _generate_topic_labels(n_topics: int, topic_labels: list = None) -> list:
    """Generate topic labels from n_topics
    Parameters
    ----------
    n_topics: int
        number of topics
    topic_labels:list (default=None)
        list of topic_labels
    Returns
    -------
    list
        list of topic labels
    """
    if topic_labels is None:
        topic_labels = [f"Topic_{n}" for n in range(1, n_topics)]
    else:
        if len(topic_labels) != n_topics:
            raise AttributeError("len(topic_labels) does not equal n_topics")
    return topic_labels


def _get_n_columns_and_n_rows(n_topics: int) -> int:
    """calculate the optimal number of rows and columns for n_topics
    Parameters
    ----------
    n_topics: int
        number of topics
    Returns
    -------
    int
        optimal number of columns
    int
        optimal number of rows
    """
    if n_topics <= 0:
        raise ValueError("Value must be an integer greater than 0")
    if n_topics <= 5:
        n_columns = n_topics
        n_rows = 1
    else:
        factors = [factor for factor in _get_factors(n_topics) if 1 < factor <= 5]
        if len(factors) > 0:
            n_columns = factors[-1]
            n_rows = int(n_topics / n_columns)
        else:
            factors = [
                factor for factor in _get_factors(n_topics + 1) if 1 < factor <= 5
            ]
            n_columns = factors[-1]
            n_rows = int((n_topics / n_columns) + 1)
    return n_rows, n_columns


def _get_factors(x: int) -> list:
    """retrieve factors of a given integer (x)
    Parameters
    ----------
    x:int
        integer
    Returns
    -------
    list
        a list of factors of x
    """
    return [i for i in range(1, x + 1) if x % i == 0]


def _get_fig_size(columns: int, rows: int) -> typing.Tuple[int, int]:
    """get figure size from number of columns and rows
    Parameters
    ----------
    columns:int
        number of columns
    rows: int
        number of rows
    Returns
    -------
    int
        width of fig
    int
        height of fig"""
    width = columns * 6
    height = (rows * 6) + 3
    return (width, height)

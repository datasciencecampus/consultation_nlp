import re

import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from scipy.sparse._csr import csr_matrix
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.modules import utils


def topic_model(
    model: str,
    question: str,
    original_series: Series,
    model_series: Series,
    stopwords: list,
    config: dict,
):
    """Complete topic modelling on model series and save outputs

    Parameters
    ----------
    model:str
        model id
    question:str
        question id
    original_series:Series
        series with original indexing
    model_series:Series
        series which has been cleaned and pre-processed
    stopwords:list
        list of stopwords to remove
    config:dict
        configuration settings for the system

    Returns
    -------
    None (prints messages to console on location of outputs)"""
    settings = config["question_settings"][question]
    vectorizer_class = {"lda": CountVectorizer, "nmf": TfidfVectorizer}
    model_class = {"lda": LatentDirichletAllocation, "nmf": NMF}
    vectorizer = vectorizer_class[model](
        max_features=settings["max_features"],
        ngram_range=settings["ngram_range"],
        min_df=settings["min_df"],
        max_df=settings["max_df"],
        lowercase=settings["lowercase"],
        stop_words=stopwords,
    )
    fitted_vectorizer = vectorizer.fit_transform(model_series)
    vectorized_df = _fit_vectorizer_to_df(fitted_vectorizer, vectorizer)
    vectorized_sum = _columnwise_sum(vectorized_df)
    _plot_common_words(
        vectorized_sum,
        n=settings["max_features"],
        name=f"{question}_{model}_common_words",
    )
    model_cl = model_class[model](
        n_components=settings["n_topics"],
        max_iter=settings["max_iter"][model],
        random_state=179,
    )
    fitted_model = model_cl.fit_transform(fitted_vectorizer)
    topic_labels = _generate_topic_labels(
        settings["n_topics"], settings["topic_labels"][model]
    )
    topic_scores = DataFrame(fitted_model, columns=topic_labels)
    _combine_text_with_topics(original_series, topic_scores, question, model)
    _plot_top_words(
        model=model_cl,
        feature_names=list(vectorized_df.columns),
        n_topics=settings["n_topics"],
        title=f"{question} {model} - Top Words by Topic",
        n_top_words=settings["n_top_words"],
        topic_labels=topic_labels,
        filename=f"{question}_{model}_top_words_by_topic",
    )
    return None


def _fit_vectorizer_to_df(
    fitted: csr_matrix, vectorizer: CountVectorizer | TfidfVectorizer
) -> DataFrame:
    """Transform fitted vectorizer to a readable dataframe

    Parameters
    ----------
    fitted:crs_matrix
        a fitted vectorizer
    vectorizer
        an initialised vectorizer class object

    Returns
    -------
    DataFrame
        a readable dataframe with each word and frequency/weight"""
    return DataFrame(fitted.toarray(), columns=vectorizer.get_feature_names_out())


def _columnwise_sum(dataframe: DataFrame) -> DataFrame:
    """sum all values in each column in Dataframe

    Parameters
    ----------
    dataframe:DataFrame
        a dataframe you want to sum the columns of

    Returns
    -------
    DataFrame
        A dataframe with the all values in each column summed"""
    return dataframe.sum(axis=0).reset_index(name="sum")


def _plot_common_words(total_features: DataFrame, n: int = 20, name: str = "top_words"):
    """plot top 'n' number of words
    Parameters
    ----------
    total_features:DataFrame
        Dataframe containing words and their frequency count
    n:int, default = 20
        number of top words to include
    name:str, default = "top_words"
        name of the chart to output
    Returns
    -------
    None (message to console on location of chart)
    """
    if n is None:
        n = 20
    top_n_features = total_features.sort_values(["sum"], ascending=[False]).head(n)
    top_words = plt.figure(figsize=(10, 10))
    plt.barh(top_n_features["index"], top_n_features["sum"])
    plt.title(re.sub("_", " ", name))
    utils._save_figure(name, top_words)


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
        topic_labels = [f"Topic_{n}" for n in range(1, n_topics + 1)]
    else:
        if len(topic_labels) != n_topics:
            raise AttributeError("len(topic_labels) does not equal n_topics")
    return topic_labels


def _combine_text_with_topics(
    original: Series, topic_scores: DataFrame, question: str, model: str
) -> None:
    """Combine original text series with topic scores dataframe

    Parameters
    ----------
    original: Series
        text data series with original indexing
    topic_scores:DataFrame
        dataframe with n_topics columns and n_document rows
    question:str
        the question id
    model:str
        the model id

    Returns
    -------
    None (prints location of file to console)"""
    text_with_topic_df = original.reset_index().join(topic_scores)
    datestamp = utils._get_datestamp()
    filename = f"data/outputs/{datestamp}_{question}_{model}_topic_score.csv"
    text_with_topic_df.to_csv(filename, index=False)
    print(f"Topics file saved as {filename}")
    return None


def _plot_top_words(
    model: LatentDirichletAllocation,
    feature_names: list,
    n_topics: int,
    title: str,
    n_top_words: int = 10,
    topic_labels: list = None,
    filename: str = "top_words_by_topic",
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
    topic_labels = utils._generate_topic_labels(n_topics, topic_labels)
    labelled_components = dict(zip(topic_labels, model.components_))
    rows, columns = _get_n_columns_and_n_rows(n_topics)
    fig, axes = plt.subplots(
        rows, columns, figsize=utils._get_fig_size(columns, rows), sharex=True
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
    utils._save_figure(filename, fig)
    return None


def _get_n_columns_and_n_rows(n_topics: int) -> int:
    """calculate the optimal number of rows and columns for n_topics
    Parameters
    ----------
    n_topics: int
        number of topics (must be integer greater than 0)
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

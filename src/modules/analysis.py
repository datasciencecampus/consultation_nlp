import typing

import spacy
from numpy.typing import ArrayLike
from pandas import DataFrame, Series
from scipy.sparse._csr import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def extract_feature_count(
    series: Series,
    max_features: int = None,
    ngram_range: tuple[float, float] = (1, 1),
    stop_words: ArrayLike = None,
    lowercase: bool = True,
    min_df=1,
    max_df=1.0,
) -> typing.Tuple[CountVectorizer, DataFrame]:
    """create a text feature count dataframe from series
    Paramaters
    ----------
    series: Series
        Series of text strings
    max_features: int, default = None
        If not None, build a vocabulary that only consider the top max_features
        ordered by term frequency across the corpus. Otherwise, all features are used.
    ngram_range: tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different word n-grams
        or char n-grams to be extracted. All values of n such such that
        min_n <= n <= max_n will be used.
    stop_words: list, default=None
        list of stopwords to remove from text strings
    lowercase: bool, default = True
        convert all characters to lowercase before tokenizing
    min_df: float or int, default = 1
        When building the vocabulary ignore terms that have a document frequency
        strictly lower than the given threshold. This value is also called cut-off
        in the literature. If float, the parameter represents a proportion of
        documents, integer absolute counts.
        This parameter is ignored if vocabulary is not None.
    max_df: float or int, default = 1.0
        When building the vocabulary ignore terms that have a document frequency
        strictly higher than the given threshold (corpus-specific stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts. This parameter is ignored if vocabulary is not None.
    Returns
    -------
    DataFrame
        A dataframe of text feature counts, displaying the number of times a word
        appears in each element of the input series
    """

    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
        lowercase=lowercase,
        min_df=min_df,
        max_df=max_df,
    )

    fitted_vector = vectorizer.fit_transform(series)

    word_count_df = DataFrame(
        fitted_vector.toarray(), columns=vectorizer.get_feature_names_out()
    )
    return (fitted_vector, word_count_df)


def retrieve_named_entities(series: Series) -> list:
    """retrieve any named entities from the series
    Parameters
    ----------
    series:Series
        A series of text strings to analyse for named entities
    Returns
    -------
    list[list[str]]
        a list of lists containing strings for each named entitity"""
    nlp = spacy.load("en_core_web_sm")
    entities = []
    for doc in nlp.pipe(series):
        entities.append([str(ent) for ent in doc.ents])
    return entities


def latent_dirichlet_allocation(
    n_topics: int, max_iter: int, fitted_vector: csr_matrix
) -> LatentDirichletAllocation:
    """fit latent direchlet allocation model on fitted vector
    Parameters
    ----------
    n_topics:int
        number of components to include in model
    max_iter: int
        maximum number of passes over the training data
    fitted_vector:csr_matrix
        fitted vector from CountVectorizer
    Returns
    -------
    LatentDirichletAllocation
        fitted lda model
    """
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        max_iter=max_iter,
        random_state=179,
    )

    lda.fit(fitted_vector)
    return lda

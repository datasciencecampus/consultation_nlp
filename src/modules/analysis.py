from numpy.typing import ArrayLike
from pandas import DataFrame, Series
from sklearn.feature_extraction.text import CountVectorizer


def extract_feature_count(
    series: Series,
    max_features: int = None,
    ngram_range: tuple[float, float] = (1, 1),
    stop_words: ArrayLike = None,
    lowercase: bool = True,
    min_df=1,
    max_df=1.0,
):
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
    return word_count_df


def get_total_feature_count(features: DataFrame) -> DataFrame:
    """sum across features to get total number of times word was used
    Parameters
    ----------
    features: DataFrame
        A dataframe of the features with each row corrosponding to a deconstructed
        string
    Returns
    -------
    DataFrame
        A dataframe of the total number of times each word is used across all
        strings"""
    total_feature_count = DataFrame()
    for column in features.columns:
        total_feature_count[column] = [features[column].sum()]
    return total_feature_count

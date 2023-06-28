# import re
# import string

from importlib import reload

# import matplotlib.pyplot as plt
# import mglearn
# import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

# from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.processing import preprocessing

reload(preprocessing)

test_array = pd.array(
    [
        "my name is Colin",
        "Hello!",
        "What do you mean?",
        " ",
        "How dare you. I liked that rabbit!",
        "beauty, beautiful, pretty, beaut?",
    ]
)
test_df = pd.DataFrame({"test_data": test_array})
raw_series = test_df["test_data"]


def run_pipeline():
    """run entire consultation nlp pipeline"""
    config = preprocessing.load_config("src/config.yaml")
    raw_data = pd.read_csv(config["raw_data_path"], encoding="cp1252")
    raw_series = raw_data["qu_3"]
    # TODO add clean_data parent function
    lower_series = raw_series.str.lower()
    without_blank_rows = preprocessing.remove_blank_rows(lower_series)
    no_punctuation_series = without_blank_rows.apply(preprocessing.remove_punctuation)
    # TODO add a spelling fixer function
    word_tokens = no_punctuation_series.apply(word_tokenize)
    stemmed_tokens = word_tokens.apply(preprocessing.stemmer)
    # TODO consider lemmatized_tokens = word_tokens.apply(preprocessing.lemmatizer)
    # defaults to look at nouns, but can change it to look at adjectives if needed
    without_stopwords = stemmed_tokens.apply(preprocessing.remove_nltk_stopwords)
    # TODO add list of stopwords to design info
    rejoined_words = without_stopwords.apply(preprocessing.rejoin_tokens)
    print(rejoined_words)

    """#Topic Modelling"""

    vect = CountVectorizer(max_features=5)
    coliv_wordsbows = vect.fit(raw_series)

    print(coliv_wordsbows.vocabulary_)


#    lda5 = LatentDirichletAllocation(
#        n_components=5, learning_method="batch", max_iter=25, random_state=0
#    )
#
#    document_topics5 = lda5.fit_transform(coliv_wordsbows)
#
#    topics = np.array([0, 1, 2, 3, 4])
#
#    sorting = np.argsort(lda5.components_, axis=1)[:, ::-1]
#    feature_names = np.array(vect.get_feature_names())
#    mglearn.tools.print_topics(
#        topics=topics,
#        feature_names=feature_names,
#        sorting=sorting,
#        topics_per_chunk=5,
#        n_words=10,
#    )
#
#     document_topics5
#
#
#    censtranf_respns = nlp_censtranf[
#        "cens_test_1"
#    ]
#    censtranf_respns = nlp_censtranf.reset_index(drop=True)
#
#
#
#
#    def topic_summary(
#        topic_number,
#    ):
#
#        topics = [topic_number]
#        mglearn.tools.print_topics(
#            topics=topics,
#            feature_names=feature_names,
#            sorting=sorting,
#            topics_per_chunk=5,
#            n_words=10,
#        )
#
#        responses = np.argsort(document_topics5[:, topic_number])[::-1]
#
#        for i in responses[:5]:
#            print(coliv_respns[i], ".\n")
#
#
#    for i in range(5):
#        topic_summary(i)
#
#    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
#    topic_names = [
#        "{:>2} ".format(i) + " ".join(words)
#        for i, words in enumerate(feature_names[sorting[:, :2]])
#    ]
#
#    ax.barh(np.arange(5), np.sum(document_topics5, axis=0))
#    ax.set_yticks(np.arange(5))
#    ax.set_yticklabels(topic_names, ha="left", va="top")
#    ax.invert_yaxis()
#    ax.set_xlim(0, 300)
#    yax = ax.get_yaxis()
#    yax.set_tick_params(pad=130)
#    plt.tight_layout()
#
#
#    topic_labels = [
#        "The first label",
#        "The second label",
#        "The second label",
#        "The third label",
#        "The fourth label",
#    ]
#
#
#    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
#    topic_names = ["{:>2} {}".format(i, label) for i, label in enumerate(topic_labels)]
#
#    ax.barh(np.arange(5), np.mean(document_topics5, axis=0))
#    ax.set_yticks(np.arange(5))
#    ax.set_yticklabels(topic_names, ha="right", va="center")
#    ax.invert_yaxis()
#    ax.set_xlim(0, 0.5)
#    yax = ax.get_yaxis()
#    yax.set_tick_params(pad=10)
#    plt.tight_layout()

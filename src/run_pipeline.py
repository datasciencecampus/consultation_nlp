# import re
# import string
# import matplotlib.pyplot as plt
# import mglearn
# import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

from src.processing.preprocessing import remove_punctuation  # ,
from src.processing.preprocessing import (  # stemmer,
    correct_spelling,
    fuzzy_compare_ratio,
    lemmatizer,
    load_config,
    rejoin_tokens,
    remove_blank_rows,
    remove_nltk_stopwords,
)

# from sklearn.decomposition import LatentDirichletAllocation
# from importlib import reload
# reload(preprocessing)


def run_pipeline():
    """run entire consultation nlp pipeline"""
    config = load_config("src/config.yaml")
    raw_data = pd.read_csv(config["raw_data_path"], encoding="cp1252")
    raw_series = raw_data["qu_3"]
    # TODO add clean_data parent function
    lower_series = raw_series.str.lower()
    without_blank_rows = remove_blank_rows(lower_series)
    spelling_fixed = without_blank_rows.apply(
        correct_spelling, config["business_terminology"]
    )
    impact_of_spell_correction = fuzzy_compare_ratio(without_blank_rows, spelling_fixed)

    # TODO consider whether there are words we need to fix manually? i.e timliness
    # print_row_by_row(without_blank_rows,spelling_fixed)
    no_punctuation_series = spelling_fixed.apply(remove_punctuation)
    word_tokens = no_punctuation_series.apply(word_tokenize)
    # stemmed_tokens = word_tokens.apply(stemmer)
    lemmatized_tokens = word_tokens.apply(lemmatizer)
    # defaults to look at nouns, but can change it to look at adjectives if needed
    without_stopwords = lemmatized_tokens.apply(remove_nltk_stopwords)
    # TODO add list of stopwords to design info
    rejoined_words = without_stopwords.apply(rejoin_tokens)

    # just printing to overcome qa aspect
    print(rejoined_words, impact_of_spell_correction)

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

import re
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from annotated_text import annotated_text
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.modules import clean_string as clean
from src.modules import preprocessing as prep
from src.modules import spell_correct as spell
from src.modules import streamlit as stream
from src.modules import topic_modelling as topic
from src.modules.config import Config

# Page configuration
issue_link = (
    "https://github.com/datasciencecampus/consultation_nlp/issues/"
    + "new?assignees=&labels=&projects=&template=bug_report.md&title="
)

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Report a bug": issue_link},
)

with open("src/modules/style.css") as f:  # noqa w605
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Side bar
st.sidebar.subheader(":blue[Natural Language Processing - Analysis]")


question_dict = {
    "Please explain how you currently use our statistics?": "qu_12",
    "How will our proposals better meet your information needs?": "qu_15",
    "What new things would these proposals enable you to do?": "qu_17",
    "What needs would not be met by these new proposals?": "qu_22",
}

question = st.sidebar.selectbox("Select a question:", list(question_dict.keys()))

st.sidebar.subheader("Topic Modelling")


vectorizer_selection = st.sidebar.selectbox(
    "Vectorizer:", ["Count Vectorizer", "TF-IDF Vectorizer"]
)

model_selection = st.sidebar.selectbox(
    "Model:", ["Latent Dirichlet Allocation", "Non-Negative Matrix Factorization"]
)
sb_b1, sb_b2, sb_b3 = st.sidebar.columns(3)

n_topics = sb_b1.number_input("Number of topics:", min_value=1, step=1, value=3)

n_iter = sb_b2.number_input(
    "Iterations:",
    min_value=10,
    max_value=10000,
    step=1,
    value=25,
    help="Number of times the model re-runs to "
    + "attempt to allow the model outputs to converge",
)

max_features = sb_b3.number_input(
    "Max features:", min_value=1, max_value=100000, value=10000, step=1
)

st.sidebar.write("Word combinations (N-grams)")
sb_c1, sb_c2 = st.sidebar.columns(2)

ngram_help_text = (
    "N-gram range sets if features to be used to characterize "
    + "texts will be: Unigrams or words (n-gram size = 1) Bigrams or terms "
    + "compounded by two words (n-gram size = 2)"
)

ngram_start_range = sb_c1.slider(
    "Start range:", help=ngram_help_text, min_value=1, max_value=10, value=1
)

ngram_end_range = sb_c2.slider(
    "End range:", help=ngram_help_text, min_value=1, max_value=10, value=2
)

st.sidebar.write("Filter by document frequency")
sb_d1, sb_d2 = st.sidebar.columns(2)

min_df = sb_d1.slider(
    "Minimum %:",
    help="Ignore terms that appear in fewer than this percentage"
    + " of responses (0% means no bottom end filter)",
    min_value=0,
    max_value=100,
    value=0,
    step=1,
)

max_df = sb_d2.slider(
    "Maximum %:",
    help="Ignore terms that appear in more than this percentage"
    + " of responses (100% means no top end filter)",
    min_value=0,
    max_value=100,
    value=100,
    step=1,
)

model_lookup = {
    "Latent Dirichlet Allocation": {"class": LatentDirichletAllocation, "short": "lda"},
    "Non-Negative Matrix Factorization": {"class": NMF, "short": "nmf"},
}

vectorizer_lookup = {
    "Count Vectorizer": {"class": CountVectorizer, "short": "count"},
    "TF-IDF Vectorizer": {"class": TfidfVectorizer, "short": "tfidf"},
}

model_short = model_lookup[model_selection]["short"]

models = {
    question_dict[question]: {
        "max_features": max_features,
        "ngram_range": (ngram_start_range, ngram_end_range),
        "min_df": min_df,
        "max_df": max_df,
        "n_topics": n_topics,
        "n_top_words": 10,
        "max_iter": {"lda": n_iter, "nmf": n_iter},
        "lowercase": True,
        "topic_labels": {
            "lda": None,
            "nmf": None,
            "model_selection": model_lookup[model_selection]["short"],
            "vectorizer_selection": vectorizer_lookup[vectorizer_selection]["short"],
        },
    }
}
######################################################################
# for testing and bug-fixing - unhash below to emulate user inputs
######################################################################
# topic_name = "Topic 1"
# question = "Please explain how you currently use our statistics?"
# vectorizer_selection = "Count Vectorizer"
# model_selection = "Non-Negative Matrix Factorization"
# n_samples = 4
# models = {question_dict[question]:{
#     "max_features": 10000,
#     "ngram_range": (1, 2),
#     "min_df":0.0,
#     "max_df":1.0,
#     "n_topics": 3,
#     "n_top_words": 10,
#     "max_iter":{
#         "lda":25,
#         "nmf":25
#     },
#     "lowercase": True,
#     "topic_labels":{
#         "lda":None,
#         "nmf":None,
#     "model_selection":model_selection,
#     "vectorizer_selection": vectorizer_selection
#     }}}
#######################################################################
#  Don't forget to re-hash it after you are finished
#######################################################################
# Main Body
st.error("**Official Sensitive:** Do not share without permission.", icon="⚠️")
with st.spinner("Updating report..."):

    # Model Processing
    config = Config().settings
    config["models"] = models
    colnames = [f"qu_{number+1}" for number in range(0, 54)]
    raw_data = pd.read_csv(
        config["general"]["raw_data_path"], names=colnames, skiprows=1
    )
    questions = list(config["models"].keys())
    spell_checker = spell.update_spell_dictionary(config["spelling"])
    question_short = question_dict[question]
    raw_series = raw_data[question_short]
    response_char_lengths = prep.get_response_length(raw_series)
    average_response_char_length = response_char_lengths.mean()
    # Cleaning
    no_ans_removed = prep.remove_no_answer(raw_series)
    without_blank_rows = prep.remove_blank_rows(no_ans_removed)
    punct_removed = without_blank_rows.apply(spell.remove_punctuation)
    cleaned_series = punct_removed.apply(clean.clean_string)
    word_replacements = spell.find_word_replacements(cleaned_series, spell_checker)
    spelling_fixed = spell.replace_words(cleaned_series, word_replacements)
    stopwords = prep.initialise_update_stopwords(
        config["general"]["additional_stopwords"]
    )
    settings = config["models"][question_short]
    # Vectorization
    vect_setup = vectorizer_lookup[vectorizer_selection]["class"](
        max_features=settings["max_features"],
        ngram_range=settings["ngram_range"],
        min_df=settings["min_df"],
        max_df=settings["max_df"],
        lowercase=settings["lowercase"],
        stop_words="english",
    )
    fitted_vect = vect_setup.fit(spelling_fixed)
    transformed_vect = fitted_vect.transform(spelling_fixed)
    fitted_vect_df = topic._fit_vectorizer_to_df(transformed_vect, fitted_vect)
    # Model fitting
    warning = None
    model_setup = model_lookup[model_selection]["class"](
        n_components=settings["n_topics"],
        max_iter=settings["max_iter"][model_lookup[model_selection]["short"]],
        random_state=179,
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        model_fitted = model_setup.fit(transformed_vect)
        model_transformed = model_fitted.transform(transformed_vect)
        topic_names = [f"Topic {i+1}" for i in range(settings["n_topics"])]
        topic_names_snake = [re.sub(" ", "_", name_x).lower() for name_x in topic_names]
        model_fitted_df = pd.DataFrame(model_transformed, columns=topic_names_snake)
        for warn in caught_warnings:
            category = re.sub(r"[\<\>\']|class", "", str(warn.category)).split(".")[-1]
            warning = f"**{category}:**  \n{warn.message}"
    # Topic Words DataFrame
    text_with_topic_df = spelling_fixed.reset_index().join(model_fitted_df)
    text_with_topic_df = text_with_topic_df.rename(columns={0: "responses"})
    topic_weights = (
        model_setup.components_ / model_setup.components_.sum(axis=1)[:, np.newaxis]
    )
    topic_words = pd.DataFrame()
    topic_words["word"] = fitted_vect.get_feature_names_out()
    for i in range(len(model_fitted.components_)):
        topic_words[f"topic_{i+1}"] = model_fitted.components_[i]
        topic_words[f"topic_{i+1}_word_importance"] = topic_weights[i]
    topic_words["word_frequency"] = topic._columnwise_sum(fitted_vect_df)["sum"]
    topic_words = topic_words[["word", "word_frequency"]].join(
        topic_words.iloc[:, 1:-1]
    )
    topic_words_final = topic_words.drop(topic_names_snake, axis=1).sort_values(
        "word_frequency", ascending=False
    )

    # Summary information
    st.header("Summary information")
    a1, a2, a3, a4, a5 = st.columns(5)
    with a1:
        st.metric("Total Responses", len(raw_data))
    with a2:
        st.metric(
            "Organisation Responses",
            str(
                round(len(raw_data[raw_data["qu_1"] == "Yes"]) / len(raw_data) * 100, 2)
            )
            + "%",
        )
    with a3:
        st.metric(
            "Private Responses",
            str(round(len(raw_data[raw_data["qu_1"] == "No"]) / len(raw_data) * 100, 2))
            + "%",
        )
    a4.empty()
    a5.empty()
    st.divider()

    # Topic Word Dataframe configuration
    if warning is not None:
        st.warning(warning, icon="⚠️")
        st.toast(warning, icon="⚠️")
    st.header("Topic Words")
    keys = list(topic_words.columns)
    values = ["Word", "Word frequency"]
    for i in range(settings["n_topics"]):
        name = f"Topic {i+1}"
        col_config = st.column_config.ProgressColumn(
            label=f"Weight - Topic {i+1}",
            help="How important this word is to the overall topic"
            + " (weights sum to 1 within topic)",
        )
        values.append(name)
        values.append(col_config)
    my_dict = dict(zip(keys, values))
    st.dataframe(
        topic_words_final,
        column_config=my_dict,
        hide_index=True,
        use_container_width=True,
    )

    # Responses by topic
    st.header("Responses by topic")
    st.dataframe(
        text_with_topic_df.style.background_gradient(subset=topic_names_snake),
        hide_index=True,
        use_container_width=True,
    )

    # Explore Topic options
    box1, box2, box3 = st.columns([0.4, 0.3, 0.3])

    topic_name = box1.selectbox("Explore Topic:", index=0, options=topic_names)

    n_samples = box2.number_input(
        "Number of responses:", min_value=1, max_value=20, value=3
    )
    label_option = box3.radio(
        "Topic Labels:", ["Off", "Single Topic", "Multi-Topic"], horizontal=True
    )

    # Topic Processing

    words_by_topic_fig = stream.plot_words_by_topic_bar(topic_words, topic_name)
    top_n_words = stream.get_top_n_words(topic_words, 20, topic_name)
    dominant_topics = stream.identify_dominant_topics(topic_words, topic_names_snake)
    topic_sample = stream.get_n_topic_samples(text_with_topic_df, topic_name, n_samples)
    word_stopword_combos = stream.create_word_stopword_combos(top_n_words, stopwords)
    topic_color = stream.get_single_topic_color(topic_names, topic_name)
    formatted_topic_single = stream.single_topic_formatting(
        top_n_words, topic_sample, topic_name, topic_names, stopwords
    )
    formatted_text = stream.multitopic_formatting(
        dominant_topics, topic_sample, topic_names
    )

    c1, c2 = st.columns([0.3, 0.7])
    with c1:
        st.pyplot(words_by_topic_fig)
    with c2:
        for i in range(n_samples):
            st.subheader(stream.get_response_no(topic_sample, i))
            st.caption(stream.generate_top_scores(topic_sample, topic_name, i))
            if label_option == "Off":
                topic_sample.loc[i, "responses"]
            if label_option == "Single Topic":
                annotated_text(formatted_topic_single[i])
            elif label_option == "Multi-Topic":
                annotated_text(formatted_text[i])

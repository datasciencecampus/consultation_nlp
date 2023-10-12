import re
import warnings
from datetime import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
from annotated_text import annotated_text
from sklearn.decomposition import LatentDirichletAllocation
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
    page_title="NLP App",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Report a bug": issue_link},
)

with open("src\modules\style.css") as f:  # noqa w605
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Side bar
st.sidebar.subheader(":blue[Natural Language Processing - Analysis]", divider="rainbow")

uploaded_file = st.sidebar.file_uploader(
    "Upload your file here:",
    type=["csv"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)
if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file, encoding="cp1252")

question = st.sidebar.selectbox("Select question column:", raw_data.columns)

st.sidebar.subheader("Vectorization Parameters")


vectorizer_selection = st.sidebar.selectbox(
    "Method:", ["Count Vectorizer", "TF-IDF Vectorizer"]
)
sb_a1, sb_a2 = st.sidebar.columns([0.6, 0.4])
min_freq_measure = sb_a1.radio(
    "Minimum document frequency measure:",
    ("Absolute", "Relative"),
    index=0,
    help="'Absolute' selects the minimum number of documents a term must "
    + "appear in, whereas 'Relative' selects the minimum percentage of total "
    + "documents a term must appear in.",
)
if min_freq_measure == "Absolute":
    min_df = sb_a2.number_input("Min frequency:", min_value=1, value=1, step=1)
else:
    min_df = sb_a2.slider(
        "Min frequency (%):",
        help="Ignore terms that appear in fewer than this percentage"
        + " of responses (0% means no bottom end filter)",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
    )

sb_b1, sb_b2 = st.sidebar.columns([0.6, 0.4])

max_freq_measure = sb_b1.radio(
    "Maximum document frequency measure:",
    ("Absolute", "Relative"),
    index=1,
    help="'Absolute' selects the maximum number of documents a term must "
    + "appear in, whereas 'Relative' selects the maximum percentage of total "
    + "documents a term must appear in.",
)
if max_freq_measure == "Absolute":
    max_df = sb_b2.number_input("Max Frequency:", min_value=1, value=100, step=1)
else:
    max_df = sb_b2.slider(
        "Max frequency (%):",
        help="Ignore terms that appear in more than this percentage"
        + " of responses (100% means no top end filter)",
        min_value=0,
        max_value=100,
        value=100,
        step=1,
    )

sb_c1, sb_c2, sb_c3 = st.sidebar.columns([0.5, 0.25, 0.25])
max_features = sb_c1.number_input(
    "Max features:",
    min_value=1,
    max_value=1000000,
    value=100000,
    step=1,
    help="The maximum number of ngram (word combination) features to include",
)
ngram_help_text = (
    "An N-gram is a sequence of N words. An N-gram range of 1-1, will capture "
    + "one word at a time (uni-grams)' e.g. 'use'; 'admin'; 'data'. A range of "
    + "1-2 will capture single and double word sequences e.g. 'use'; "
    + "'admin'; 'data; 'use admin'; 'admin data', which can provide helpful "
    + "context and more features which can help strengthen weaker datasets"
)

ngram_start_range = sb_c2.slider(
    "Ngram start range:", help=ngram_help_text, min_value=1, max_value=5, value=1
)

ngram_end_range = sb_c3.slider(
    "Ngram end range:", help=ngram_help_text, min_value=1, max_value=5, value=2
)


st.sidebar.subheader("Topic Modelling - Parameters")

(
    sb_d1,
    sb_d2,
    sb_d3,
    sb_d4,
) = st.sidebar.columns(4)
n_topics = sb_d1.number_input("Number of topics:", min_value=1, step=1, value=3)


alpha = sb_d2.slider(
    "Alpha:",
    help="Alpha represents document-topic density - with a higher alpha, "
    + "documents are made up of more topics, and with lower alpha, documents "
    + "contain fewer topics",
    min_value=0.0,
    max_value=1.0,
    value=(1 / n_topics),
    step=0.01,
)

beta = sb_d3.slider(
    "Beta:",
    help="Beta represents topic-word density - with a high beta, topics are "
    + "made up of most of the words in the corpus, and with a low beta they "
    + "consist of few words.",
    min_value=0.0,
    max_value=1.0,
    value=(1 / n_topics),
    step=0.01,
)

n_iter = sb_d4.number_input(
    "Max iterations:",
    min_value=10,
    max_value=10000,
    step=1,
    value=25,
    help="The maximum number of times the model re-runs to "
    + "attempt to allow the model outputs to converge. If it converges before it"
    + " will stop sooner.",
)


vectorizer_lookup = {
    "Count Vectorizer": {"class": CountVectorizer, "short": "count"},
    "TF-IDF Vectorizer": {"class": TfidfVectorizer, "short": "tfidf"},
}

# ######################################################################
# # for testing and bug-fixing - unhash below to emulate user inputs
# ######################################################################
# # topic_name = "Topic 1"
# # question = "Please explain how you currently use our statistics?"
# # vectorizer_selection = "Count Vectorizer"
# # model_selection = "Non-Negative Matrix Factorization"
# # n_samples = 4
# # models = {question_dict[question]:{
# #     "max_features": 10000,
# #     "ngram_range": (1, 2),
# #     "min_df":0.0,
# #     "max_df":1.0,
# #     "n_topics": 3,
# #     "n_top_words": 10,
# #     "max_iter":{
# #         "lda":25,
# #         "nmf":25
# #     },
# #     "lowercase": True,
# #     "topic_labels":{
# #         "lda":None,
# #         "nmf":None,
# #     "model_selection":model_selection,
# #     "vectorizer_selection": vectorizer_selection
# #     }}}
# #######################################################################
# #  Don't forget to re-hash it after you are finished
# #######################################################################
# Main Body
st.error("**Official Sensitive:** Do not share without permission.", icon="⚠️")
with st.spinner("Updating report..."):
    # Model Processing
    config = Config().settings
    spell_checker = spell.update_spell_dictionary(config["spelling"])
    raw_series = raw_data[question]
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
    # Vectorization
    vect_setup = vectorizer_lookup[vectorizer_selection]["class"](
        max_features=max_features,
        ngram_range=(ngram_start_range, ngram_end_range),
        min_df=min_df,
        max_df=max_df,
        lowercase=True,
        stop_words=stopwords,
    )
    fitted_vect = vect_setup.fit(spelling_fixed)
    transformed_vect = fitted_vect.transform(spelling_fixed)
    fitted_vect_df = topic._fit_vectorizer_to_df(transformed_vect, fitted_vect)
    # Model fitting
    warning = None
    model_setup = LatentDirichletAllocation(
        n_components=n_topics,
        doc_topic_prior=alpha,
        topic_word_prior=beta,
        max_iter=n_iter,
        random_state=179,
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        model_fitted = model_setup.fit(transformed_vect)
        model_transformed = model_fitted.transform(transformed_vect)
        topic_names = [f"Topic {i+1}" for i in range(n_topics)]
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
    st.divider()

    # Topic Word Dataframe configuration
    if warning is not None:
        st.warning(warning, icon="⚠️")
        st.toast(warning, icon="⚠️")
    st.header("Topic Words")
    keys = list(topic_words.columns)
    values = ["Word", "Word frequency"]
    for i in range(n_topics):
        name = f"Topic {i+1}"
        col_config = st.column_config.ProgressColumn(
            label=f"Weight - Topic {i+1}",
            help="How important this word is to the overall topic"
            + " (weights sum to 1 within topic)",
        )
        values.append(name)
        values.append(col_config)
    my_dict = dict(zip(keys, values))
    todays_date = dt.today().strftime("%Y%m%d")
    st.dataframe(
        topic_words_final,
        column_config=my_dict,
        hide_index=True,
        use_container_width=True,
    )
    topic_words_a, topic_words_b = st.columns([0.9, 0.1])
    topic_words_b.download_button(
        "Export table",
        topic_words_final.to_csv().encode("utf-8"),
        f"{todays_date}_words_by_topic.csv",
    )
    st.divider()
    # Responses by topic
    st.header("Responses by topic")
    st.dataframe(
        text_with_topic_df.style.background_gradient(subset=topic_names_snake),
        hide_index=True,
        use_container_width=True,
    )
    topic_table_a, topic_table_b = st.columns([0.9, 0.10])
    topic_table_b.download_button(
        "Export table",
        text_with_topic_df.to_csv().encode("utf-8"),
        f"{todays_date}_responses_by_topic.csv",
    )
    st.divider()
    # Explore Topic options
    box1, box2, box3 = st.columns([0.4, 0.3, 0.3])

    topic_name = box1.selectbox("Explore Topic:", index=0, options=topic_names)

    n_samples = box2.number_input(
        "Number of responses:", min_value=1, max_value=20, value=8
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

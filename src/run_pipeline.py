import pandas as pd
from nltk.tokenize import word_tokenize

from src.modules.analysis import (
    extract_feature_count,
    latent_dirichlet_allocation,
    retrieve_named_entities,
)
from src.modules.preprocessing import (
    get_response_length,
    initialise_update_stopwords,
    load_config,
    prepend_str_to_list_objects,
    rejoin_tokens,
    remove_blank_rows,
    remove_nltk_stopwords,
    remove_punctuation,
    shorten_tokens,
    spellcorrect_series,
)
from src.modules.quality_checks import compare_spelling
from src.modules.visualisation import (
    create_wordcloud,
    plot_common_words,
    plot_top_words,
)


def run_pipeline():
    """run consultation nlp pipeline"""
    config = load_config("src/config.yaml")
    question_config = load_config("src/question_config.yaml")
    colnames = [f"qu_{number+1}" for number in range(0, 54)]
    raw_data = pd.read_csv(
        config["raw_data_path"], encoding="cp1252", names=colnames, skiprows=1
    )
    questions = prepend_str_to_list_objects(
        question_config["questions_to_interpret"], "qu_"
    )
    for question in questions:
        raw_series = raw_data[question]
        # TODO add clean_data parent function
        response_char_lengths = get_response_length(raw_series)
        average_response_char_length = response_char_lengths.mean()
        without_blank_rows = remove_blank_rows(raw_series)
        spelling_fixed = spellcorrect_series(
            without_blank_rows, config["buisness_terminology"]
        )
        compare_spelling(without_blank_rows, spelling_fixed, filename=question)
        lower_series = spelling_fixed.str.lower()
        no_punctuation_series = remove_punctuation(lower_series)
        word_tokens = no_punctuation_series.apply(word_tokenize)
        short_tokens = shorten_tokens(word_tokens, config["lemmatize"])
        without_stopwords = short_tokens.apply(
            lambda x: remove_nltk_stopwords(x, config["additional_stopwords"])
        )
        rejoined_words = without_stopwords.apply(rejoin_tokens)
        all_text_combined = " ".join(rejoined_words)
        create_wordcloud(all_text_combined, f"{question}_wordcloud")
        stopwords = initialise_update_stopwords(config["additional_stopwords"])
        fitted_vector, features = extract_feature_count(
            series=spelling_fixed,
            ngram_range=config["feature_count"]["ngram_range"],
            min_df=config["feature_count"]["min_df"],
            max_df=config["feature_count"]["max_df"],
            max_features=config["feature_count"]["max_features"],
            lowercase=config["feature_count"]["lowercase"],
            stop_words=stopwords,
        )
        total_features = features.sum(axis=0).reset_index(name="count")
        plot_common_words(
            total_features,
            n=config["feature_count"]["max_features"],
            name=f"{question}_most_common_words",
        )
        entities = retrieve_named_entities(without_blank_rows)
        lda = latent_dirichlet_allocation(
            n_topics=config["lda"]["n_topics"],
            max_iter=config["lda"]["max_iter"],
            fitted_vector=fitted_vector,
        )
        plot_top_words(
            model=lda,
            feature_names=list(features.columns),
            n_topics=config["lda"]["n_topics"],
            title=config["lda"]["title"],
            n_top_words=config["lda"]["n_top_words"],
            topic_labels=config["lda"]["topic_labels"],
            filename=f"{question}_top_words_by_topic",
        )

        print(
            response_char_lengths,
            average_response_char_length,
            total_features,
            entities,
        )


# code to execute script from terminal
if __name__ == "__main__":
    run_pipeline()

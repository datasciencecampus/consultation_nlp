import pandas as pd
from nltk.tokenize import word_tokenize

from src.modules import analysis
from src.modules import clean_string as clean
from src.modules import preprocessing as prep
from src.modules import quality_checks as quality
from src.modules import spell_correct as spell
from src.modules import visualisation as vis


def run_pipeline():
    """run consultation nlp pipeline"""
    config = prep.load_config("src/config.yaml")
    question_config = prep.load_config("src/question_config.yaml")
    colnames = [f"qu_{number+1}" for number in range(0, 54)]
    raw_data = pd.read_csv(config["raw_data_path"], names=colnames, skiprows=1)
    questions = prep.prepend_str_to_list_objects(
        question_config["questions_to_interpret"], "qu_"
    )
    for question in questions:
        raw_series = raw_data[question]
        response_char_lengths = prep.get_response_length(raw_series)
        average_response_char_length = response_char_lengths.mean()
        without_blank_rows = prep.remove_blank_rows(raw_series)
        cleaned_series = without_blank_rows.apply(clean.clean_string)
        spelling_fixed, modifications = spell.auto_correct_series(
            cleaned_series, config["buisness_terminology"]
        )
        quality.compare_spelling(
            without_blank_rows, spelling_fixed, modifications, filename=question
        )
        lower_series = spelling_fixed.str.lower()
        no_punctuation_series = lower_series.apply(spell.remove_punctuation)
        word_tokens = no_punctuation_series.apply(word_tokenize)
        short_tokens = prep.shorten_tokens(word_tokens, config["lemmatize"])
        without_stopwords = short_tokens.apply(
            lambda x: prep.remove_nltk_stopwords(x, config["additional_stopwords"])
        )
        rejoined_words = without_stopwords.apply(prep.rejoin_tokens)
        all_text_combined = " ".join(rejoined_words)  # rejoin_tokens
        vis.create_wordcloud(all_text_combined, f"{question}_wordcloud")
        stopwords = prep.initialise_update_stopwords(config["additional_stopwords"])
        fitted_vector, features = analysis.extract_feature_count(
            series=without_blank_rows,
            ngram_range=config["feature_count"]["ngram_range"],
            min_df=config["feature_count"]["min_df"],
            max_df=config["feature_count"]["max_df"],
            max_features=config["feature_count"]["max_features"],
            lowercase=config["feature_count"]["lowercase"],
            stop_words=stopwords,
        )
        total_features = features.sum(axis=0).reset_index(name="count")
        vis.plot_common_words(
            total_features,
            n=config["feature_count"]["max_features"],
            name=f"{question}_most_common_words",
        )
        entities = analysis.retrieve_named_entities(cleaned_series)
        lda = analysis.latent_dirichlet_allocation(
            n_topics=config["lda"]["n_topics"],
            max_iter=config["lda"]["max_iter"],
            fitted_vector=fitted_vector,
        )
        vis.plot_top_words(
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
            modifications,
        )


# code to execute script from terminal
if __name__ == "__main__":
    run_pipeline()

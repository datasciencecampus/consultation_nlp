import pandas as pd
from nltk.tokenize import word_tokenize

from src.modules.analysis import (
    extract_feature_count,
    get_total_feature_count,
    latent_dirichlet_allocation,
    retrieve_named_entities,
)
from src.modules.preprocessing import (
    initialise_update_stopwords,
    load_config,
    rejoin_tokens,
    remove_blank_rows,
    remove_nltk_stopwords,
    remove_punctuation,
    shorten_tokens,
    spellcorrect_series,
)
from src.modules.quality_checks import fuzzy_compare_ratio  # print_row_by_row,
from src.modules.visualisation import create_wordcloud, plot_top_words


def run_pipeline():
    """run consultation nlp pipeline"""
    config = load_config("src/config.yaml")
    colnames = [f"qu_{number+1}" for number in range(0, 33)]
    raw_data = pd.read_csv(
        config["raw_data_path"], encoding="cp1252", names=colnames, skiprows=1
    )
    raw_series = raw_data["qu_11"]
    # TODO add clean_data parent function
    without_blank_rows = remove_blank_rows(raw_series)
    spelling_fixed = spellcorrect_series(
        without_blank_rows, config["buisness_terminology"]
    )
    impact_of_spell_correction = fuzzy_compare_ratio(without_blank_rows, spelling_fixed)
    lower_series = spelling_fixed.str.lower()
    #      print_row_by_row(without_blank_rows,spelling_fixed)
    no_punctuation_series = remove_punctuation(lower_series)
    word_tokens = no_punctuation_series.apply(word_tokenize)
    short_tokens = shorten_tokens(word_tokens, config["lemmatize"])
    without_stopwords = short_tokens.apply(
        lambda x: remove_nltk_stopwords(x, config["additional_stopwords"])
    )
    rejoined_words = without_stopwords.apply(rejoin_tokens)
    all_text_combined = " ".join(rejoined_words)
    create_wordcloud(all_text_combined)
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
    total_features = get_total_feature_count(features)
    entities = retrieve_named_entities(without_blank_rows)
    lda, document_topics = latent_dirichlet_allocation(
        n_components=10, max_iter=50, fitted_vector=fitted_vector
    )
    plot_top_words(
        model=lda,
        feature_names=list(features.columns),
        n_topics=10,
        title="Top words by topic",
        n_top_words=10,
        topic_labels=None,
    )

    print(impact_of_spell_correction, total_features, entities)


# code to execute script from terminal
if __name__ == "__main__":
    run_pipeline()

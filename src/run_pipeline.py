import pandas as pd
from nltk.tokenize import word_tokenize

from src.modules import clean_string as clean
from src.modules import named_entity_recognition as ner
from src.modules import preprocessing as prep
from src.modules import quality_checks as quality
from src.modules import spell_correct as spell
from src.modules import topic_modelling as topic
from src.modules import word_cloud as wc


def run_pipeline():
    """run consultation nlp pipeline"""
    config = prep.load_config("src/config.yaml")
    config["spelling_words"] = prep.load_json("src/spelling_words.json")
    config["question_settings"] = prep.load_config("src/question_config.yaml")
    colnames = [f"qu_{number+1}" for number in range(0, 54)]
    raw_data = pd.read_csv(config["raw_data_path"], names=colnames, skiprows=1)
    questions = list(config["question_settings"].keys())
    spell_checker = spell.update_spell_dictionary(config["spelling_words"])
    # TODO add forloop for question in questions:
    question = "qu_12"
    raw_series = raw_data[question]
    response_char_lengths = prep.get_response_length(raw_series)
    average_response_char_length = response_char_lengths.mean()
    without_blank_rows = prep.remove_blank_rows(raw_series)
    punct_removed = without_blank_rows.apply(spell.remove_punctuation)
    cleaned_series = punct_removed.apply(clean.clean_string)
    word_replacements = spell.find_word_replacements(cleaned_series, spell_checker)
    spelling_fixed = spell.replace_words(cleaned_series, word_replacements)
    quality.compare_spelling(
        without_blank_rows, spelling_fixed, word_replacements, filename=question
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
    wc.create_wordcloud(all_text_combined, f"{question}_wordcloud")
    stopwords = prep.initialise_update_stopwords(config["additional_stopwords"])
    model_data = without_blank_rows

    [
        topic.topic_model(
            model, question, without_blank_rows, model_data, stopwords, config
        )
        for model in ["nmf", "lda"]
    ]

    entities = ner.retrieve_named_entities(model_data)

    print(response_char_lengths, average_response_char_length, entities, questions)


# code to execute script from terminal
if __name__ == "__main__":
    run_pipeline()

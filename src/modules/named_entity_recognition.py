import spacy
from pandas import Series


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

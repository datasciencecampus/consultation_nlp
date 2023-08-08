import matplotlib.pyplot as plt
from wordcloud import WordCloud

from src.modules import utils


def create_wordcloud(text: str, name: str = "wordcloud") -> None:
    """generate a wordcloud with the given filename
    Parameters
    ----------
    text: str
        text for wordcloud
    filename: str
        the name and path you want to save the wordcloud to
    Returns:
    None (message to console on location of file)
    """
    wordcloud = WordCloud().generate(text)
    figure = plt.figure(figsize=(5, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    utils._save_figure(name, figure)
    return None

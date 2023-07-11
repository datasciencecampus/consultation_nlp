from datetime import datetime as dt

import matplotlib.pyplot as plt
from wordcloud import WordCloud


def create_wordcloud(text: str, filename: str = "wordcloud"):
    """generate a wordcloud with the given filename
    Parameters
    ----------
    text: str
        text for wordcloud
    filename: str
        the name and path you want to save the wordcloud to
    Returns:
       prints message to console saying where file is saved
    """
    wordcloud = WordCloud().generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    datestamp = dt.strftime(dt.now(), "%Y%m%d")
    filename_datestamp_ext = "data/outputs/" + datestamp + "_" + filename + ".jpeg"
    plt.savefig(filename_datestamp_ext, bbox_inches="tight")
    print(f"Wordcloud saved to {filename_datestamp_ext}")
    return None

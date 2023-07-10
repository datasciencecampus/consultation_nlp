import matplotlib.pyplot as plt
from pandas import Series
from wordcloud import WordCloud


def print_row_by_row(base: Series, comparison: Series) -> None:
    """print each pair of words row by row
    Parameters
    ----------
    base: Series
        the base series for comparison
    comparison: Series
        the series you want to compare against
    Returns
    -------
    None
    """
    for i in base.index:
        print(base[i])
        print(comparison[i])
    return None


def create_wordcloud(text: str, filename: str = "data/outputs/wordcloud.jpeg"):
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
    plt.savefig(filename, bbox_inches="tight")
    print(f"Wordcloud saved to {filename}")

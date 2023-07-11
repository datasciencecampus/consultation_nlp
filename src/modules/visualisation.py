import matplotlib.pyplot as plt
from wordcloud import WordCloud


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

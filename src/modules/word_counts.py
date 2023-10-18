import plotly.graph_objects as go
from matplotlib.figure import Figure
from pandas import Series


def plot_word_counts(data: Series) -> Figure:
    """Create a word count boxplot

    Parameters
    ----------
    data : Series
        a series of word counts corrosponding to the original data

    Returns
    -------
    Figure
        a boxplot of word counts for the responses
    """
    fig = go.Figure()
    fig.add_trace(go.Box(x=data, name=""))
    fig.update_layout(title=go.layout.Title(text="Response word counts"), height=300)

    return fig

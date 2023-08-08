import typing
from datetime import datetime

from matplotlib.figure import Figure

from src.modules import utils


def _trim_ends(string_x: str) -> str:
    """remove whitespace at the start and end of string
    Parameters
    ----------
    string_x:str
        string you want to remove whitespace from
    Returns
    -------
    str
        string with no whitespace at the ends"""
    return string_x.strip()


def _get_datestamp():
    """get a datestamp for current time

    Returns
    -------
    str
        Datestamp in YYYYMMDD format"""
    return datetime.strftime(datetime.now(), "%Y%m%d")


def _save_figure(name: str, fig: Figure) -> None:
    """save figure with datestamp
    Parameters
    ----------
    name: str
        name of the figure
    fig
        the figure object
    Returns
    -------
    None (message to console on location of file)
    """
    datestamp = utils._get_datestamp()
    filename = f"data/outputs/{datestamp}_{name}.jpeg"
    fig.savefig(filename, bbox_inches="tight")
    print(f"{name} plot saved as {filename}")
    return None


def _get_fig_size(columns: int, rows: int) -> typing.Tuple[int, int]:
    """get figure size from number of columns and rows
    Parameters
    ----------
    columns:int
        number of columns
    rows: int
        number of rows
    Returns
    -------
    int
        width of fig
    int
        height of fig"""
    width = columns * 6
    height = (rows * 6) + 3
    return (width, height)

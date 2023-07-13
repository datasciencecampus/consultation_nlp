from pandas import Series
from rapidfuzz.fuzz import ratio


def fuzzy_compare_ratio(base: Series, comparison: Series) -> Series:
    """compare the base series to the comparison series to get
    a similarity ratio between strings in the same column
    Parameters
    ----------
    base: Series
        the base series for comparison
    comparison: Series
        the series you want to compare against
    Returns
    -------
    Series
        a series of ratios (type:float) with scores closer to 100
        indicating complete match"""
    fuzzy_ratio = Series(map(ratio, base, comparison))
    return fuzzy_ratio


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

from datetime import datetime as dt

from pandas import DataFrame, Series
from rapidfuzz.fuzz import ratio


def compare_spelling(
    before: Series, after: Series, filename: str = "spelling_corrections.csv"
) -> None:
    """Create a csv with the before and after spellings compared
    Parameters
    ----------
    before: Series
        the before spell checker series for comparison
    comparison: Series
        the 'after' series you want to compare against
    Returns
    -------
    None (message to console on location of saved file)
    """
    fuzzy_ratio = _fuzzy_compare_ratio(before, after)
    spelling_table = DataFrame(
        {"fuzzy_ratio": fuzzy_ratio, "before_spelling": before, "after_spelling": after}
    )
    spelling_table = spelling_table[spelling_table["fuzzy_ratio"] != 100.000000]
    datestamp = dt.strftime(dt.now(), "%Y%m%d")
    full_filename = f"data/outputs/{datestamp}_{filename}_spelling_table.csv"
    spelling_table.to_csv(full_filename)
    print(f"spelling table saved to {full_filename}")


def _fuzzy_compare_ratio(base: Series, comparison: Series) -> Series:
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

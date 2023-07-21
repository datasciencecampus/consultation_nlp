from datetime import datetime as dt

from pandas import DataFrame, Series
from rapidfuzz.fuzz import ratio


def compare_spelling(
    before: Series,
    after: Series,
    modifications: dict,
    filename: str = "spelling_corrections.csv",
) -> None:
    """Create a csv with the before and after spellings compared
    Parameters
    ----------
    before: Series
        the before spell checker series for comparison
    after: Series
        the 'after' series you want to compare against
    modifications: dict
        dictionary of before and after word modifications
    Returns
    -------
    None (message to console on location of saved file)
    """
    comparison_dataframe = _rebase_comparison_index(before)
    comparison_dataframe["after"] = after.sort_index()
    comparison_dataframe["fuzzy_ratio"] = _fuzzy_compare_ratio(
        comparison_dataframe["before"], comparison_dataframe["after"]
    )
    comparison_dataframe["spelling_replacements"] = list(modifications.values())
    datestamp = dt.strftime(dt.now(), "%Y%m%d")
    full_filename = f"data/outputs/{datestamp}_{filename}_spelling_table.csv"
    comparison_dataframe.to_csv(full_filename, index=False)
    print(f"spelling table saved to {full_filename}")


def _rebase_comparison_index(series: Series) -> DataFrame:
    """create a column with the old index of the before series
    Parameters
    ----------
    series:Series
        a series of data which you want to reindex/rebase
    Returns
    -------
    Dataframe
        a dataframe with two columns original_index and before"""
    return series.reset_index().set_axis(["original_index", "before"], axis=1)


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

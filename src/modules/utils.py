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

from pandas import Series

from src.modules.quality_checks import fuzzy_compare_ratio


class TestFuzzyCompareRatio:
    def test_ratios(self):
        base = Series(["this is", "this isn't"])
        comparison = Series(["this is", "yellow"])
        expected = Series([100.00, 0.0])
        actual = fuzzy_compare_ratio(base, comparison)
        assert all(expected == actual), "fuzzy scoring not working correctly"

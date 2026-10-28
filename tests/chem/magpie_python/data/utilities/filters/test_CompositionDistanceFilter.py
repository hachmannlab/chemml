from math import sqrt
from chemml.chem.magpie_python.data.materials.CompositionEntry import CompositionEntry
from chemml.chem.magpie_python.data.utilities.filters.CompositionDistanceFilter import \
    CompositionDistanceFilter

class TestCompositionDistanceFilter:
    def test_label(self):
        # Make sample list of CompositionEntry's.
        entries =[CompositionEntry(composition="Fe50Al50"), CompositionEntry(
            composition="Fe55Al45"), CompositionEntry(composition="Fe25Al75")]

        # Make filter.
        cdf = CompositionDistanceFilter()
        cdf.set_target_composition(CompositionEntry(composition="Fe49Al51"))
        cdf.set_distance_threshold(1.1)

        # Test #1.
        res = cdf.label(entries)
        assert res[0]
        assert not (res[1])
        assert not (res[2])

        # Test #2.
        cdf.set_distance_threshold(7)
        res = cdf.label(entries)
        assert res[0]
        assert res[1]
        assert not (res[2])

    def test_distances(self):
        # Test binary distance.
        assert abs(0.0 - CompositionDistanceFilter.compute_distance(CompositionEntry(composition='Fe'), CompositionEntry(composition='Fe'), p=-1)) <= 1e-06

        assert abs(0.0 - CompositionDistanceFilter.compute_distance(CompositionEntry(composition='Fe'), CompositionEntry(composition='Fe'), p=0)) <= 1e-06

        assert abs(0.0 - CompositionDistanceFilter.compute_distance(CompositionEntry(composition='Fe'), CompositionEntry(composition='Fe'), p=2)) <= 1e-06

        assert abs(0.5 - CompositionDistanceFilter.compute_distance(CompositionEntry(composition='FeO'), CompositionEntry(composition='Fe'), p=-1)) <= 1e-06

        assert abs(2.0 - CompositionDistanceFilter.compute_distance(CompositionEntry(composition='FeO)'), CompositionEntry(composition='Fe'), p=0)) <= 1e-06

        assert abs(sqrt(0.5) - CompositionDistanceFilter.compute_distance(CompositionEntry(composition='FeO'), CompositionEntry(composition='Fe'), p=2)) <= 1e-06
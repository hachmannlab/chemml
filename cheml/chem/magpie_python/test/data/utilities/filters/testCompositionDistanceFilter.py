import unittest
from math import sqrt
from .....data.materials.CompositionEntry import CompositionEntry
from .....data.utilities.filters.CompositionDistanceFilter import \
    CompositionDistanceFilter

class testCompositionDistanceFilter(unittest.TestCase):
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
        self.assertTrue(res[0])
        self.assertFalse(res[1])
        self.assertFalse(res[2])

        # Test #2.
        cdf.set_distance_threshold(7)
        res = cdf.label(entries)
        self.assertTrue(res[0])
        self.assertTrue(res[1])
        self.assertFalse(res[2])

    def test_distances(self):
        # Test binary distance.
        self.assertAlmostEquals(0.0,
                                CompositionDistanceFilter.compute_distance(
                                    CompositionEntry(composition="Fe"),
                                    CompositionEntry(composition="Fe"),
                                    p=-1), delta=1e-6)

        self.assertAlmostEquals(0.0,
                                CompositionDistanceFilter.compute_distance(
                                    CompositionEntry(composition="Fe"),
                                    CompositionEntry(composition="Fe"),
                                    p=0), delta=1e-6)

        self.assertAlmostEquals(0.0,
                                CompositionDistanceFilter.compute_distance(
                                    CompositionEntry(composition="Fe"),
                                    CompositionEntry(composition="Fe"),
                                    p=2), delta=1e-6)

        self.assertAlmostEquals(0.5,
                                CompositionDistanceFilter.compute_distance(
                                    CompositionEntry(composition="FeO"),
                                    CompositionEntry(composition="Fe"),
                                    p=-1), delta=1e-6)

        self.assertAlmostEquals(2.0,
                                CompositionDistanceFilter.compute_distance(
                                    CompositionEntry(composition="FeO)"),
                                    CompositionEntry(composition="Fe"),
                                    p=0), delta=1e-6)

        self.assertAlmostEquals(sqrt(0.5),
                                CompositionDistanceFilter.compute_distance(
                                    CompositionEntry(composition="FeO"),
                                    CompositionEntry(composition="Fe"),
                                    p=2), delta=1e-6)
import unittest
from math import log, sqrt
import numpy.testing as np
from .....attributes.generators.composition.GCLPAttributeGenerator import \
    GCLPAttributeGenerator
from .....data.materials.CompositionEntry import CompositionEntry

class testGCLPAttributeGenerator(unittest.TestCase):
    def test_attribute_generator(self):
        # Create the hull data set.
        g = GCLPAttributeGenerator()
        g.set_phases([CompositionEntry(composition="NiAl")],[-1.0])

        # Create the test set.
        entries = [CompositionEntry(composition="Al"), CompositionEntry(
            composition="Ni3Al"), CompositionEntry(composition="NiAl")]

        # Compute features.
        features = g.generate_features(entries)

        # Test results.
        np.assert_array_almost_equal([0.0, 1, 0, 0, 0], features.values[0])
        np.assert_array_almost_equal([-0.5, 2, sqrt(0.125), sqrt(0.125),
                                      log(0.5)], features.values[1])
        np.assert_array_almost_equal([-1.0, 1, 0, 0, 0], features.values[2])

        # Set "no-count".
        g.set_count_phases(False)

        # Compute features.
        features = g.generate_features(entries)

        # Test results.
        np.assert_array_almost_equal([0.0, 0, 0], features.values[0])
        np.assert_array_almost_equal([-0.5, sqrt(0.125), sqrt(0.125)],
                                     features.values[1])
        np.assert_array_almost_equal([-1.0, 0, 0], features.values[2])
# -*- coding: utf-8 -*-
from chemml.chem.magpie_python import IonicityAttributeGenerator
from chemml.chem.magpie_python import CompositionEntry

class TestIonicityAttributeGenerator:
    def test_attribute_generator(self):
        # Make a list of CompositionEntry's.
        entries = [CompositionEntry(composition="NaCl"), CompositionEntry(
            composition="Al2MnCu"), CompositionEntry(composition="Fe")]

        # Make the generator.
        ig = IonicityAttributeGenerator()

        # Run the generator.
        features = ig.generate_features(entries)

        # Test results.
        assert 3 == len(features.columns)

        # Results for NaCl.
        assert abs(1 - features.values[0][0]) <= 1e-06
        assert abs(0.7115 - features.values[0][1]) <= 0.01
        assert abs(0.3557 - features.values[0][2]) <= 0.01

        # Results for Al2MnCu.
        assert abs(0 - features.values[1][0]) <= 1e-06
        assert abs(0.0301 - features.values[1][1]) <= 0.01
        assert abs(0.0092 - features.values[1][2]) <= 0.01

        # Results for Fe.
        assert abs(0 - features.values[2][0]) <= 1e-06
        assert abs(0 - features.values[2][1]) <= 0.01
        assert abs(0 - features.values[2][2]) <= 0.01
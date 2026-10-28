# -*- coding: utf-8 -*-
from chemml.chem.magpie_python import ElementFractionAttributeGenerator
from chemml.chem.magpie_python import CompositionEntry
from chemml.chem.magpie_python.data.materials.util.LookUpData import LookUpData

class TestElementFractionAttributeGenerator:
    def test_attribute_generator(self):
        # Make list of CompositionEntry's.
        entries = [CompositionEntry(composition="NaCl"), CompositionEntry(
            composition="Fe")]

        # Make generator.
        el = ElementFractionAttributeGenerator()

        # Run generator.
        features = el.generate_features(entries)

        # Test results.
        assert len(LookUpData.element_names) == features.values[0].size
        assert abs(1.0 - sum(features.values[0])) <= 1e-06
        assert abs(0.0 - min(features.values[0])) <= 1e-06
        assert abs(0.5 - features.values[0][10]) <= 1e-06
        assert abs(1.0 - sum(features.values[1])) <= 1e-06
        assert abs(0.0 - min(features.values[1])) <= 1e-06
        assert abs(1.0 - features.values[1][25]) <= 1e-06
import unittest
from .....attributes.generators.crystal.ChemicalOrderingAttributeGenerator \
    import ChemicalOrderingAttributeGenerator
from .....data.materials.CrystalStructureEntry import CrystalStructureEntry
from .....vassal.data.Atom import Atom
from .....vassal.data.Cell import Cell

class testChemicalOrderingAttributeGenerator(unittest.TestCase):
    def test_results(self):
        # Structure of a B2 crystal.
        structure = Cell()
        structure.add_atom(Atom([0, 0, 0], 0))
        structure.add_atom(Atom([0.5, 0.5, 0.5], 1))
        structure.set_type_name(0, "Al")
        structure.set_type_name(1, "Ni")

        entry = CrystalStructureEntry(structure, name="B2", radii=None)
        entries = [entry]

        # Create feature generator.
        gen = ChemicalOrderingAttributeGenerator()
        gen.set_weighted(False)
        gen.set_shells([1, 2])

        # Generate features.
        features = gen.generate_features(entries)

        # Test results.
        self.assertAlmostEquals(0.142857, features.values[0][0], delta=1e-6)
        self.assertAlmostEquals(0.04, features.values[0][1], delta=1e-6)

        # Now with weights.
        gen.set_weighted(True)
        features = gen.generate_features(entries)

        # Test results.
        self.assertAlmostEquals(0.551982, features.values[0][0], delta=1e-6)
        self.assertAlmostEquals(0.253856, features.values[0][1], delta=1e-6)
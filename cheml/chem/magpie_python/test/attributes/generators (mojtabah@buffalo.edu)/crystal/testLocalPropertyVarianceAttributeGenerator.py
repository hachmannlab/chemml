import numpy.testing as np_tst
from .....attributes.generators.crystal\
    .LocalPropertyVarianceAttributeGenerator \
    import LocalPropertyVarianceAttributeGenerator
from .....data.materials.CrystalStructureEntry import CrystalStructureEntry
from .testCoordinationNumberAttributeGenerator import \
    testCoordinationNumberAttributeGenerator
from .....vassal.data.Atom import Atom
from .....vassal.data.Cell import Cell

class testLocalPropertyVarianceAttributeGenerator(
    testCoordinationNumberAttributeGenerator):
    def get_generator(self):
        gen = LocalPropertyVarianceAttributeGenerator()
        gen.clear_shells()
        gen.add_shells([1, 2])
        gen.add_elemental_property("Electronegativity")
        return gen

    def expected_count(self):
        return 10

    def test_results2(self):
        # Create a L12-H3He structure.
        # Structure of  L12.
        structure = Cell()
        structure.add_atom(Atom([0, 0, 0], 1))
        structure.add_atom(Atom([0.5, 0.5, 0], 0))
        structure.add_atom(Atom([0.5, 0, 0.5], 0))
        structure.add_atom(Atom([0, 0.5, 0.5], 0))
        structure.set_type_name(0, "H")
        structure.set_type_name(1, "He")

        entries = [
            CrystalStructureEntry(structure, name="L12-HHe", radii=None)]

        # Get the feature generator.
        gen = self.get_generator()
        gen.clear_shells()
        gen.clear_elemental_properties()
        gen.add_shell(1)
        gen.add_elemental_property("Number")

        # Generate features.
        features = gen.generate_features(entries)

        # Test the results.
        self.assertEquals(5, features.shape[1])
        np_tst.assert_array_almost_equal(
            [0.166666667, 0.083333333, 0, 2.0 / 9,
             2.0 / 9], features.values[0])
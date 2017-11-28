import unittest
from data.materials.CrystalStructureEntry import CrystalStructureEntry
from vassal.data.Atom import Atom
from vassal.data.Cell import Cell

class testAtomicStructureEntry(unittest.TestCase):
    def test_replacement(self):
        # Make B2-CuZr
        cell = Cell()
        cell.add_atom(Atom([0, 0, 0], 0))
        cell.add_atom(Atom([0.5, 0.5, 0.5], 1))
        cell.set_type_name(0, "Cu")
        cell.set_type_name(1, "Zr")
        CuZr = CrystalStructureEntry(cell, "CuZr", None)

        # Run Voronoi tessellation.
        CuZr.compute_voronoi_tessellation()

        # Make B2-NiZr
        changes = {"Cu":"Ni"}
        NiZr = CuZr.replace_elements(changes)

        # Make sure the tessellation object did not change.
        self.assertTrue(CuZr.compute_voronoi_tessellation() is
                        NiZr.compute_voronoi_tessellation())

        # Make sure the two are still unchanged.
        self.assertAlmostEquals(0.5, CuZr.get_element_fraction(name="Cu"),
                                delta=1e-6)
        self.assertAlmostEquals(0.0, CuZr.get_element_fraction(name="Ni"),
                                delta=1e-6)
        self.assertAlmostEquals(0.0, NiZr.get_element_fraction(name="Cu"),
                                delta=1e-6)
        self.assertAlmostEquals(0.5, NiZr.get_element_fraction(name="Ni"),
                                delta=1e-6)

        # Now, change the structure such that it has fewer types.
        changes["Ni"] = "Zr"

        bccZr = NiZr.replace_elements(changes)

        # Make sure the structure only has one type.
        self.assertAlmostEquals(1.0, bccZr.get_element_fraction(name="Zr"),
                                delta=1e-6)
        self.assertEquals(1, bccZr.get_structure().n_types())

        self.assertFalse(NiZr.compute_voronoi_tessellation() is
                         bccZr.compute_voronoi_tessellation())

    def test_to_string(self):
        # Make B2-CuZr
        cell = Cell()
        cell.add_atom(Atom([0, 0, 0], 0))
        cell.add_atom(Atom([0.5, 0.5, 0.5], 1))
        cell.set_type_name(0, "Cu")
        cell.set_type_name(1, "Zr")
        CuZr = CrystalStructureEntry(cell, "B2", None)

        name = CuZr.__str__()
        print name
        self.assertTrue("CuZr" in name)
        self.assertTrue("B2" in name)
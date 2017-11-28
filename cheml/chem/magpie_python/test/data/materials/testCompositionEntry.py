import unittest
from itertools import permutations
import numpy.testing as np
import os
from data.materials.CompositionEntry import CompositionEntry

class testCompositionEntry(unittest.TestCase):
    def test_parsing(self):
        entry = CompositionEntry(composition="Fe")
        self.assertEquals(1, len(entry.get_element_ids()))
        self.assertAlmostEquals(1.0, entry.get_element_fraction(name="Fe"),
                                delta=1e-6)

        entry = CompositionEntry(composition="FeO0")
        self.assertEquals(1, len(entry.get_element_ids()))
        self.assertAlmostEquals(1.0, entry.get_element_fraction(name="Fe"),
                                delta=1e-6)

        entry = CompositionEntry(composition="FeCl3")
        self.assertEquals(2, len(entry.get_element_ids()))
        self.assertAlmostEquals(0.25, entry.get_element_fraction(name="Fe"),
                                delta=1e-6)
        self.assertAlmostEquals(0.75, entry.get_element_fraction(name="Cl"),
                                delta=1e-6)

        entry = CompositionEntry(composition="Fe1Cl_3")
        self.assertEquals(2, len(entry.get_element_ids()))
        self.assertAlmostEquals(0.25, entry.get_element_fraction(name="Fe"),
                                delta=1e-6)
        self.assertAlmostEquals(0.75, entry.get_element_fraction(name="Cl"),
                                delta=1e-6)

        entry = CompositionEntry(composition="FeCl_3")
        self.assertEquals(2, len(entry.get_element_ids()))
        self.assertAlmostEquals(0.25, entry.get_element_fraction(name="Fe"),
                                delta=1e-6)
        self.assertAlmostEquals(0.75, entry.get_element_fraction(name="Cl"),
                                delta=1e-6)

        entry = CompositionEntry(composition="FeClCl2")
        self.assertEquals(2, len(entry.get_element_ids()))
        self.assertAlmostEquals(0.25, entry.get_element_fraction(name="Fe"),
                                delta=1e-6)
        self.assertAlmostEquals(0.75, entry.get_element_fraction(name="Cl"),
                                delta=1e-6)

        entry = CompositionEntry(composition="Ca(NO3)2")
        self.assertEquals(3, len(entry.get_element_ids()))
        self.assertAlmostEquals(1.0 / 9, entry.get_element_fraction(name="Ca"),
                                delta=1e-6)
        self.assertAlmostEquals(2.0 / 9, entry.get_element_fraction(name="N"),
                                delta=1e-6)
        self.assertAlmostEquals(6.0 / 9, entry.get_element_fraction(name="O"),
                                delta=1e-6)

        entry = CompositionEntry(composition="Ca(N[O]3)2")
        self.assertEquals(3, len(entry.get_element_ids()))
        self.assertAlmostEquals(1.0 / 9, entry.get_element_fraction(name="Ca"),
                                delta=1e-6)
        self.assertAlmostEquals(2.0 / 9, entry.get_element_fraction(name="N"),
                                delta=1e-6)
        self.assertAlmostEquals(6.0 / 9, entry.get_element_fraction(name="O"),
                                delta=1e-6)

        entry = CompositionEntry(composition="Ca(N(O1.5)2)2")
        self.assertEquals(3, len(entry.get_element_ids()))
        self.assertAlmostEquals(1.0 / 9, entry.get_element_fraction(name="Ca"),
                                delta=1e-6)
        self.assertAlmostEquals(2.0 / 9, entry.get_element_fraction(name="N"),
                                delta=1e-6)
        self.assertAlmostEquals(6.0 / 9, entry.get_element_fraction(name="O"),
                                delta=1e-6)

        entry = CompositionEntry(composition="Ca{N{O1.5}2}2")
        self.assertEquals(3, len(entry.get_element_ids()))
        self.assertAlmostEquals(1.0 / 9, entry.get_element_fraction(name="Ca"),
                                delta=1e-6)
        self.assertAlmostEquals(2.0 / 9, entry.get_element_fraction(name="N"),
                                delta=1e-6)
        self.assertAlmostEquals(6.0 / 9, entry.get_element_fraction(name="O"),
                                delta=1e-6)

        entry = CompositionEntry(composition="CaO-0.01Ni")
        self.assertEquals(3, len(entry.get_element_ids()))
        self.assertAlmostEquals(1.0 / 2.01, entry.get_element_fraction(
            name="Ca"), delta=1e-6)
        self.assertAlmostEquals(0.01 / 2.01, entry.get_element_fraction(
            name="Ni"), delta=1e-6)
        self.assertAlmostEquals(1.0 / 2.01, entry.get_element_fraction(
            name="O"), delta=1e-6)

        entry = CompositionEntry(composition="CaO"+str(chr(183))+"0.01Ni")
        self.assertEquals(3, len(entry.get_element_ids()))
        self.assertAlmostEquals(1.0 / 2.01, entry.get_element_fraction(
            name="Ca"), delta=1e-6)
        self.assertAlmostEquals(0.01 / 2.01, entry.get_element_fraction(
            name="Ni"), delta=1e-6)
        self.assertAlmostEquals(1.0 / 2.01, entry.get_element_fraction(
            name="O"), delta=1e-6)

        entry = CompositionEntry(composition="Ca(N(O1.5)2)2-2H2O")
        self.assertEquals(4, len(entry.get_element_ids()))
        self.assertAlmostEquals(1.0 / 15, entry.get_element_fraction(name="Ca"),
                                delta=1e-6)
        self.assertAlmostEquals(2.0 / 15, entry.get_element_fraction(name="N"),
                                delta=1e-6)
        self.assertAlmostEquals(8.0 / 15, entry.get_element_fraction(name="O"),
                                delta=1e-6)
        self.assertAlmostEquals(4.0 / 15, entry.get_element_fraction(name="H"),
                                delta=1e-6)

        entry = CompositionEntry(composition="Ca(N(O1.5)2)2-2.1(H)2O")
        self.assertEquals(4, len(entry.get_element_ids()))
        self.assertAlmostEquals(1.0 / 15.3, entry.get_element_fraction(
            name="Ca"), delta=1e-6)
        self.assertAlmostEquals(2.0 / 15.3, entry.get_element_fraction(
            name="N"), delta=1e-6)
        self.assertAlmostEquals(8.1 / 15.3, entry.get_element_fraction(
            name="O"), delta=1e-6)
        self.assertAlmostEquals(4.2 / 15.3, entry.get_element_fraction(
            name="H"), delta=1e-6)

        entry = CompositionEntry(composition="{[("
                            "Fe0.6Co0.4)0.75B0.2Si0.05]0.96Nb0.04}96Cr4")
        self.assertEquals(6, len(entry.get_element_ids()))
        self.assertAlmostEquals(0.41472, entry.get_element_fraction(
            name="Fe"), delta=1e-6)
        self.assertAlmostEquals(0.27648, entry.get_element_fraction(
            name="Co"), delta=1e-6)
        self.assertAlmostEquals(0.18432, entry.get_element_fraction(
            name="B"), delta=1e-6)
        self.assertAlmostEquals(0.04608, entry.get_element_fraction(
            name="Si"), delta=1e-6)
        self.assertAlmostEquals(0.0384, entry.get_element_fraction(
            name="Nb"), delta=1e-6)
        self.assertAlmostEquals(0.04, entry.get_element_fraction(
            name="Cr"), delta=1e-6)

    def test_set_composition(self):
        # One element.
        elem = [0]
        frac = [1]
        entry = CompositionEntry(element_ids=elem, fractions=frac)
        self.assertEquals(1, len(entry.get_element_ids()))
        self.assertAlmostEquals(1.0, entry.get_element_fraction(name="H"),
                                delta=1e-6)

        # One element with duplicates.
        elem = [0, 0]
        frac = [0.5, 0.5]
        entry = CompositionEntry(element_ids=elem, fractions=frac)
        self.assertEquals(1, len(entry.get_element_ids()))
        self.assertAlmostEquals(1.0, entry.get_element_fraction(name="H"),
                                delta=1e-6)

        # One element with zero.
        elem = [0, 1]
        frac = [1, 0]
        entry = CompositionEntry(element_ids=elem, fractions=frac)
        self.assertEquals(1, len(entry.get_element_ids()))
        self.assertAlmostEquals(1.0, entry.get_element_fraction(name="H"),
                                delta=1e-6)

        # Two elements.
        elem = [16, 10]
        frac = [1, 1]
        entry = CompositionEntry(element_ids=elem, fractions=frac)
        self.assertEquals(2, len(entry.get_element_ids()))
        self.assertAlmostEquals(0.5, entry.get_element_fraction(name="Na"),
                                delta=1e-6)
        self.assertAlmostEquals(0.5, entry.get_element_fraction(name="Cl"),
                                delta=1e-6)
        np.assert_array_equal([10, 16], entry.get_element_ids())
        np.assert_array_almost_equal([0.5, 0.5], entry.get_element_fractions())
        self.assertAlmostEquals(2, entry.number_in_cell, delta=1e-6)

        # Two elements with duplicates.
        elem = [11, 16, 16]
        frac = [1, 1, 1]
        entry = CompositionEntry(element_ids=elem, fractions=frac)
        self.assertEquals(2, len(entry.get_element_ids()))
        self.assertAlmostEquals(1.0 / 3, entry.get_element_fraction(name="Mg"),
                                delta=1e-6)
        self.assertAlmostEquals(2.0 / 3, entry.get_element_fraction(name="Cl"),
                                delta=1e-6)
        np.assert_array_equal([11, 16], entry.get_element_ids())
        np.assert_array_almost_equal([1.0 / 3, 2.0 / 3],
                                     entry.get_element_fractions())
        self.assertAlmostEquals(3, entry.number_in_cell, delta=1e-6)

        # Two elements with zero.
        elem = [11, 16, 16]
        frac = [1, 2, 0]
        entry = CompositionEntry(element_ids=elem, fractions=frac)
        self.assertEquals(2, len(entry.get_element_ids()))
        self.assertAlmostEquals(1.0 / 3, entry.get_element_fraction(name="Mg"),
                                delta=1e-6)
        self.assertAlmostEquals(2.0 / 3, entry.get_element_fraction(name="Cl"),
                                delta=1e-6)
        np.assert_array_equal([11, 16], entry.get_element_ids())
        np.assert_array_almost_equal([1.0 / 3, 2.0 / 3],
                                     entry.get_element_fractions())
        self.assertAlmostEquals(3, entry.number_in_cell, delta=1e-6)

    def test_sort_and_normalize(self):
        # Make an example composition.
        elem = [1, 2, 3, 4, 5]
        frac = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Make first composition.
        entry = CompositionEntry(element_ids=elem, fractions=frac)
        entry_elems = entry.get_element_ids()
        entry_fracs = entry.get_element_fractions()
        for i in range(5):
            self.assertAlmostEquals(entry_fracs[i], entry_elems[i] / 15.0,
                                    delta=1e-6)

        # Iterate through all permutations.
        for perm in permutations([0, 1, 2, 3, 4]):
            # Make a new version of elem and frac.
            new_elem = list(elem)
            new_frac = list(frac)
            for i in range(len(new_elem)):
                new_elem[i] = elem[perm[i]]
                new_frac[i] = frac[perm[i]]

            # Make sure it parses the same.
            new_entry = CompositionEntry(element_ids=elem, fractions=frac)
            self.assertEquals(new_entry, entry)
            self.assertEquals(0, new_entry.__cmp__(entry))
            np.assert_array_equal(entry_elems, new_entry.get_element_ids())
            np.assert_array_almost_equal(entry_fracs,
                                         new_entry.get_element_fractions())

    def test_compare(self):
        this_file_path = os.path.dirname(__file__)
        abs_path = os.path.join(this_file_path, "../../test-files/")
        entries = CompositionEntry.import_composition_list(
            abs_path+"small_set_comp.txt")
        for e1 in range(len(entries)):
            self.assertEquals(0, entries[e1].__cmp__(entries[e1]))
            for e2 in range(e1 + 1, len(entries)):
                self.assertEquals(entries[e1].__cmp__(entries[e2]),
                                  -1 * entries[e2].__cmp__(entries[e1]))
                if entries[e1].__cmp__(entries[e2]) == 0:
                    self.assertEquals(entries[e1].__hash__(), entries[
                        e2].__hash__())
                    self.assertTrue(entries[e1].__eq__(entries[e2]))
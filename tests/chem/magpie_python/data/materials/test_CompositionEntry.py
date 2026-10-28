from itertools import permutations
import numpy.testing as np
import os
from importlib.resources import files

from chemml.chem.magpie_python.data.materials.CompositionEntry import CompositionEntry

class TestCompositionEntry:
    def test_parsing(self):
        entry = CompositionEntry(composition="Fe")
        assert 1 == len(entry.get_element_ids())
        assert abs(1.0 - entry.get_element_fraction(name='Fe')) <= 1e-06

        entry = CompositionEntry(composition="FeO0")
        assert 1 == len(entry.get_element_ids())
        assert abs(1.0 - entry.get_element_fraction(name='Fe')) <= 1e-06

        entry = CompositionEntry(composition="FeCl3")
        assert 2 == len(entry.get_element_ids())
        assert abs(0.25 - entry.get_element_fraction(name='Fe')) <= 1e-06
        assert abs(0.75 - entry.get_element_fraction(name='Cl')) <= 1e-06

        entry = CompositionEntry(composition="Fe1Cl_3")
        assert 2 == len(entry.get_element_ids())
        assert abs(0.25 - entry.get_element_fraction(name='Fe')) <= 1e-06
        assert abs(0.75 - entry.get_element_fraction(name='Cl')) <= 1e-06

        entry = CompositionEntry(composition="FeCl_3")
        assert 2 == len(entry.get_element_ids())
        assert abs(0.25 - entry.get_element_fraction(name='Fe')) <= 1e-06
        assert abs(0.75 - entry.get_element_fraction(name='Cl')) <= 1e-06

        entry = CompositionEntry(composition="FeClCl2")
        assert 2 == len(entry.get_element_ids())
        assert abs(0.25 - entry.get_element_fraction(name='Fe')) <= 1e-06
        assert abs(0.75 - entry.get_element_fraction(name='Cl')) <= 1e-06

        entry = CompositionEntry(composition="Ca(NO3)2")
        assert 3 == len(entry.get_element_ids())
        assert abs(1.0 / 9 - entry.get_element_fraction(name='Ca')) <= 1e-06
        assert abs(2.0 / 9 - entry.get_element_fraction(name='N')) <= 1e-06
        assert abs(6.0 / 9 - entry.get_element_fraction(name='O')) <= 1e-06

        entry = CompositionEntry(composition="Ca(N[O]3)2")
        assert 3 == len(entry.get_element_ids())
        assert abs(1.0 / 9 - entry.get_element_fraction(name='Ca')) <= 1e-06
        assert abs(2.0 / 9 - entry.get_element_fraction(name='N')) <= 1e-06
        assert abs(6.0 / 9 - entry.get_element_fraction(name='O')) <= 1e-06

        entry = CompositionEntry(composition="Ca(N(O1.5)2)2")
        assert 3 == len(entry.get_element_ids())
        assert abs(1.0 / 9 - entry.get_element_fraction(name='Ca')) <= 1e-06
        assert abs(2.0 / 9 - entry.get_element_fraction(name='N')) <= 1e-06
        assert abs(6.0 / 9 - entry.get_element_fraction(name='O')) <= 1e-06

        entry = CompositionEntry(composition="Ca{N{O1.5}2}2")
        assert 3 == len(entry.get_element_ids())
        assert abs(1.0 / 9 - entry.get_element_fraction(name='Ca')) <= 1e-06
        assert abs(2.0 / 9 - entry.get_element_fraction(name='N')) <= 1e-06
        assert abs(6.0 / 9 - entry.get_element_fraction(name='O')) <= 1e-06

        entry = CompositionEntry(composition="CaO-0.01Ni")
        assert 3 == len(entry.get_element_ids())
        assert abs(1.0 / 2.01 - entry.get_element_fraction(name='Ca')) <= 1e-06
        assert abs(0.01 / 2.01 - entry.get_element_fraction(name='Ni')) <= 1e-06
        assert abs(1.0 / 2.01 - entry.get_element_fraction(name='O')) <= 1e-06

        entry = CompositionEntry(composition="CaO"+str(chr(183))+"0.01Ni")
        assert 3 == len(entry.get_element_ids())
        assert abs(1.0 / 2.01 - entry.get_element_fraction(name='Ca')) <= 1e-06
        assert abs(0.01 / 2.01 - entry.get_element_fraction(name='Ni')) <= 1e-06
        assert abs(1.0 / 2.01 - entry.get_element_fraction(name='O')) <= 1e-06

        entry = CompositionEntry(composition="Ca(N(O1.5)2)2-2H2O")
        assert 4 == len(entry.get_element_ids())
        assert abs(1.0 / 15 - entry.get_element_fraction(name='Ca')) <= 1e-06
        assert abs(2.0 / 15 - entry.get_element_fraction(name='N')) <= 1e-06
        assert abs(8.0 / 15 - entry.get_element_fraction(name='O')) <= 1e-06
        assert abs(4.0 / 15 - entry.get_element_fraction(name='H')) <= 1e-06

        entry = CompositionEntry(composition="Ca(N(O1.5)2)2-2.1(H)2O")
        assert 4 == len(entry.get_element_ids())
        assert abs(1.0 / 15.3 - entry.get_element_fraction(name='Ca')) <= 1e-06
        assert abs(2.0 / 15.3 - entry.get_element_fraction(name='N')) <= 1e-06
        assert abs(8.1 / 15.3 - entry.get_element_fraction(name='O')) <= 1e-06
        assert abs(4.2 / 15.3 - entry.get_element_fraction(name='H')) <= 1e-06

        entry = CompositionEntry(composition="{[("
                            "Fe0.6Co0.4)0.75B0.2Si0.05]0.96Nb0.04}96Cr4")
        assert 6 == len(entry.get_element_ids())
        assert abs(0.41472 - entry.get_element_fraction(name='Fe')) <= 1e-06
        assert abs(0.27648 - entry.get_element_fraction(name='Co')) <= 1e-06
        assert abs(0.18432 - entry.get_element_fraction(name='B')) <= 1e-06
        assert abs(0.04608 - entry.get_element_fraction(name='Si')) <= 1e-06
        assert abs(0.0384 - entry.get_element_fraction(name='Nb')) <= 1e-06
        assert abs(0.04 - entry.get_element_fraction(name='Cr')) <= 1e-06

    def test_set_composition(self):
        # One element.
        elem = [0]
        frac = [1]
        entry = CompositionEntry(element_ids=elem, fractions=frac)
        assert 1 == len(entry.get_element_ids())
        assert abs(1.0 - entry.get_element_fraction(name='H')) <= 1e-06

        # One element with duplicates.
        elem = [0, 0]
        frac = [0.5, 0.5]
        entry = CompositionEntry(element_ids=elem, fractions=frac)
        assert 1 == len(entry.get_element_ids())
        assert abs(1.0 - entry.get_element_fraction(name='H')) <= 1e-06

        # One element with zero.
        elem = [0, 1]
        frac = [1, 0]
        entry = CompositionEntry(element_ids=elem, fractions=frac)
        assert 1 == len(entry.get_element_ids())
        assert abs(1.0 - entry.get_element_fraction(name='H')) <= 1e-06

        # Two elements.
        elem = [16, 10]
        frac = [1, 1]
        entry = CompositionEntry(element_ids=elem, fractions=frac)
        assert 2 == len(entry.get_element_ids())
        assert abs(0.5 - entry.get_element_fraction(name='Na')) <= 1e-06
        assert abs(0.5 - entry.get_element_fraction(name='Cl')) <= 1e-06
        np.assert_array_equal([10, 16], entry.get_element_ids())
        np.assert_array_almost_equal([0.5, 0.5], entry.get_element_fractions())
        assert abs(2 - entry.number_in_cell) <= 1e-06

        # Two elements with duplicates.
        elem = [11, 16, 16]
        frac = [1, 1, 1]
        entry = CompositionEntry(element_ids=elem, fractions=frac)
        assert 2 == len(entry.get_element_ids())
        assert abs(1.0 / 3 - entry.get_element_fraction(name='Mg')) <= 1e-06
        assert abs(2.0 / 3 - entry.get_element_fraction(name='Cl')) <= 1e-06
        np.assert_array_equal([11, 16], entry.get_element_ids())
        np.assert_array_almost_equal([1.0 / 3, 2.0 / 3],
                                     entry.get_element_fractions())
        assert abs(3 - entry.number_in_cell) <= 1e-06

        # Two elements with zero.
        elem = [11, 16, 16]
        frac = [1, 2, 0]
        entry = CompositionEntry(element_ids=elem, fractions=frac)
        assert 2 == len(entry.get_element_ids())
        assert abs(1.0 / 3 - entry.get_element_fraction(name='Mg')) <= 1e-06
        assert abs(2.0 / 3 - entry.get_element_fraction(name='Cl')) <= 1e-06
        np.assert_array_equal([11, 16], entry.get_element_ids())
        np.assert_array_almost_equal([1.0 / 3, 2.0 / 3],
                                     entry.get_element_fractions())
        assert abs(3 - entry.number_in_cell) <= 1e-06

    def test_sort_and_normalize(self):
        # Make an example composition.
        elem = [1, 2, 3, 4, 5]
        frac = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Make first composition.
        entry = CompositionEntry(element_ids=elem, fractions=frac)
        entry_elems = entry.get_element_ids()
        entry_fracs = entry.get_element_fractions()
        for i in range(5):
            assert abs(entry_fracs[i] - entry_elems[i] / 15.0) <= 1e-06

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
            assert new_entry == entry
            assert 0 == new_entry.__cmp__(entry)
            np.assert_array_equal(entry_elems, new_entry.get_element_ids())
            np.assert_array_almost_equal(entry_fracs,
                                         new_entry.get_element_fractions())

    def test_compare(self):
        # this_file_path = os.path.dirname(__file__)
        # abs_path = os.path.join(this_file_path, "../../test-files/")
        abs_path = str(files('chemml').joinpath('datasets', 'data', 'magpie_python_test'))
        entries = CompositionEntry.import_composition_list(
            os.path.join(abs_path, "small_set_comp.txt"))
        for e1 in range(len(entries)):
            assert 0 == entries[e1].__cmp__(entries[e1])
            for e2 in range(e1 + 1, len(entries)):
                assert entries[e1].__cmp__(entries[e2]) == -1 * entries[e2].__cmp__(entries[e1])
                if entries[e1].__cmp__(entries[e2]) == 0:
                    assert entries[e1].__hash__() == entries[e2].__hash__()
                    assert entries[e1].__eq__(entries[e2])
from ...data.materials.util.LookUpData import LookUpData
from ...data.utilities.generators.PhaseDiagramCompositionEntryGenerator \
    import PhaseDiagramCompositionEntryGenerator
from ...utility.tools.OxidationStateGuesser import OxidationStateGuesser

class IonicCompoundFinder:
    """Class to find nearby compositions from a given nominal composition that
    can be charge neutral.
    Works by finding all combinations of elements in
    the supplied composition that are within a certain distance of the target
    composition with less than a certain number of atoms per unit cell.
    Distance is computed as the L_1 distance of the composition vector.
    Example: Fe3Al and FeAl are 0.5 apart.

    Attributes
    ----------
    nominal_composition : CompositionEntry
        Nominal composition.
    maxmium_distance : float
        Maximum acceptable distance from nominal composition.
    max_formula_unit : int
        Maximum number of atoms in formula unit.

    """

    # Nominal composition.
    nominal_composition = None

    # Maximum acceptable distance from nominal composition.
    maximum_distance = 0.1

    # Maximum number of atoms in formula unit.
    max_formula_unit_size = 5

    def set_nominal_composition(self, entry):
        """Function to set the target composition of the ionic compound.

        Parameters
        ----------
        entry : CompositionEntry
            Desired nominal composition with element names and fractions as
            keys and values respectively.

        Returns
        -------

        """
        if len(entry.get_element_ids()) < 2:
            raise ValueError("Must be at least a binary compound.")
        self.nominal_composition = entry

    def set_maximum_distance(self, dist):
        """Function to set the allowed maximum distance from the target value.
        Note, the distance is computed as the L_1 norm of the composition
        vector assuming one of the elements is a balance (i.e., only sum the
        difference for N-1 elements).

        Parameters
        ----------
        dist : float
            Maximum allowed distance.

        """
        self.maximum_distance = dist

    def set_max_formula_unit_size(self, size):
        """Function to set maximum number of atoms in formula unit. Example:
        NaCl has 2.

        Parameters
        ----------
        size : int
            Maximum allowed size.

        """
        self.max_formula_unit_size = size

    def find_all_compounds(self):
        """Function to find all the compounds in the vicinity of the target
        composition.

        Returns
        -------
        accepted : array-like
            A list of CompositionEntry's.
        """

        # Get elements in the nominal compound.
        elems = self.nominal_composition.get_element_ids()
        fracs = self.nominal_composition.get_element_fractions()

        # Get list of all possible compositions.
        gen = PhaseDiagramCompositionEntryGenerator()
        gen.set_elements_by_index(elems)
        gen.set_even_spacing(False)
        gen.set_order(1, len(elems))
        gen.set_size(self.max_formula_unit_size)
        all_possibilities = gen.generate_entries()

        hits = []
        # Find which ones fit the desired tolerance.
        for entry in all_possibilities:
            # See if it is close enough in composition.
            dist = 0.0
            for e in range(len(elems)):
                dist += abs(fracs[e] - entry.get_element_fraction(id=elems[e]))

            if dist > self.maximum_distance:
                continue

            # See if it is ionically neutral.
            ox_g = OxidationStateGuesser()
            en = LookUpData.load_property("Electronegativity")
            os = LookUpData.load_property("OxidationStates")
            ox_g.set_electronegativity(en)
            ox_g.set_oxidationstates(os)
            can_form_ionic = len(ox_g.get_possible_states(entry)) > 0

            if can_form_ionic:
                hits.append((dist, entry))

        # Sort such that closest is first.
        hits.sort()

        # Get only compositions.
        accepted = [i[1] for i in hits]
        return accepted

from data.materials.CompositionEntry import CompositionEntry
import numpy as np
from data.materials.util.LookUpData import LookUpData
from vassal.analysis.VoronoiCellBasedAnalysis import VoronoiCellBasedAnalysis

class AtomicStructureEntry(CompositionEntry):
    """
    Class to represent a crystal structure.
    """

    # Crystal structure.
    structure = None

    # Name of entry.
    name = None

    # Link to atomic radii array.
    radii = None

    # Voronoi tessellation of this structure.
    voronoi = None

    def __init__(self, structure, name, radii):
        """
        Function to create an entry given its crystal structure.
        :param structure: Structure of this entry.
        :param name: Name of the structure (used for debugging purposes).
        :param radii: Radii to use for each element (null to leave radii
        unchanged).
        """
        super(AtomicStructureEntry, self).__init__()
        self.structure = structure
        self.name = name
        if structure.n_atoms() == 0:
            raise Exception("Cannot handle blank crystal structures.")
        self.radii = radii

        # Compute the composition of this crystal.
        self.compute_composition()

    def compute_composition(self):
        """
        Function to compute the composition of this crystal.
        :return:
        """

        # Get the composition.
        n_t = self.structure.n_types()
        elems = np.zeros(n_t, dtype=int)
        count = np.zeros(n_t, dtype=float)
        for i in range(n_t):
            name = self.structure.get_type_name(i)
            if name in LookUpData.element_ids:
                elems[i] = LookUpData.element_ids[name]
            else:
                raise Exception("Element name not recognized "+name)

            # Get the number of that atom.
            count[i] = float(self.structure.number_of_type(i))

            # Set the atomic radius of that atom.
            if self.radii is not None:
                self.structure.set_type_radius(i, self.radii[elems[i]])

        # Before reordering compositions.
        self.set_composition(count, element_ids=elems, to_sort=False)

    def replace_elements(self, replacements):
        """
        Function to create a new entry by replacing elements on this entry.
        :param replacements: Map of elements to replace. Key: Old element,
        Value: New element.
        :return: New entry.
        """

        # Create new entry.
        new_entry = self.__copy__()
        new_entry.structure.replace_type_names(replacements)
        new_entry.structure.merge_like_types()
        new_entry.compute_composition()

        # If Voronoi tessellation has already been computed, create a tool
        # for the new entry w/o recomputing the tessellation.
        if self.voronoi is not None and (self.structure.n_types() !=
                                             new_entry.structure.n_types()):
            new_entry.voronoi = None
        return new_entry

    def __copy__(self):
        """
        Function to create a copy of this instance.
        :return: A copy.
        """
        x = AtomicStructureEntry(self.structure.__copy__(), self.name,
            self.radii)
        x.element_ids = list(self.element_ids)
        x.fractions = list(self.fractions)
        x.voronoi = self.voronoi
        return x

    def __eq__(self, other):
        """
        Function to check if this instance equals another instance.
        :param other: The other instance.
        :return: Whether they are equal or not.
        """
        if isinstance(other, AtomicStructureEntry):
            return self.structure.__eq__(other.structure) and \
                   super(AtomicStructureEntry, self).__eq__(other)
        return False

    def __cmp__(self, other):
        """
        Function to compare this instance with another instance.
        :param other: The other instance.
        :return: -1 if self < other , 1 if self > other or 0 if self = other.
        """
        if isinstance(other, AtomicStructureEntry):
            # First: Check for equality.
            if self.__eq__(other):
                return 0

            # Second: Check the composition / attributes.
            super_comp = super(AtomicStructureEntry, self).__cmp__(other)
            if super_comp != 0:
                return super_comp

            # Third: Extreme measures.
            if self.structure.n_atoms() != other.structure.n_atoms():
                return self.structure.n_atoms() - other.structure.n_atoms()
            if self.structure.n_types() != other.structure.n_types():
                return self.structure.n_types() - other.structure.n_types()
            for i in range(3):
                for j in range(3):
                    v1 = self.structure.get_basis()[i][j]
                    v2 = other.structure.get_basis()[i][j]
                    if (v1 - v2) ** 2 < 1e-30:
                        c = 0
                    elif v1 < v2:
                        c = -1
                    else:
                        c = +1

                    if c != 0:
                        return c

            for i in range(self.structure.n_atoms()):
                my_atom = self.structure.get_atom(i)
                your_atom = other.structure.get_atom(i)
                if my_atom.get_type() > your_atom.get_type():
                    return my_atom.get_type() - your_atom.get_type()
                my_pos = my_atom.get_position()
                your_pos = your_atom.get_position()
                for i in range(3):
                    v1 = my_pos[i]
                    v2 = your_pos[i]
                    if (v1 - v2) ** 2 < 1e-30:
                        c = 0
                    elif v1 < v2:
                        c = -1
                    else:
                        c = +1
            raise Exception("These entries were supposed to be unequal.")
        else:
            return super(AtomicStructureEntry, self).__cmp__(other)

    def __hash__(self):
        """
        Function to compute the hashcode of this instance.
        :return: Hashcode.
        """
        return super(AtomicStructureEntry, self).__hash__() ^ \
               self.structure.__hash__()

    def get_structure(self):
        """
        Function to get link to the structure.
        :return: Structure this entry represents.
        """
        return self.structure

    def get_name(self):
        """
        Function to get the name of this entry.
        :return: Name.
        """
        return self.name

    def compute_voronoi_tessellation(self):
        """
        Function to compute the voronoi tessellation of this structure.
        :return: Tool used to query properties of the tessellation.
        """
        if self.voronoi is None:
            self.voronoi = VoronoiCellBasedAnalysis(radical=False)
            self.voronoi.analyze_structure(self.structure)
        elif not self.voronoi.tessellation_is_converged():
            raise Exception("Tessellation did not converge.")
        return self.voronoi

    def clear_representations(self):
        """
        Function to clear out the representations used when computing
        attributes.
        :return:
        """
        self.voronoi = None

    def __str__(self):
        """
        Function the generate the string representation of this instance.
        :return: String representation.
        """
        comp = super(AtomicStructureEntry, self).__str__()
        return self.name+":"+comp
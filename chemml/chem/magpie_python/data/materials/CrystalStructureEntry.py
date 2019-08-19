from os.path import join, isfile
from os import listdir
from ...data.materials.CompositionEntry import CompositionEntry
import numpy as np
from ...data.materials.util.LookUpData import LookUpData
from ...vassal.analysis.VoronoiCellBasedAnalysis import \
    VoronoiCellBasedAnalysis
from ...vassal.io.VASP5IO import VASP5IO

class CrystalStructureEntry(CompositionEntry):
    """Class to represent a crystal structure.

    Attributes
    ----------
    structure : Cell
        Crystal structure represented in the form of a Cell.
    name : str
        Name given to denote this structure. Mainly used for book-keeping.
    radii : array-like
        List of radii (floats) for various atoms in the periodic table.
    voronoi : VoronoiCellBasedAnalysis
        Tool used to query properties of the tessellation.

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
        """Function to create an entry given its crystal structure.

        Parameters
        ----------
        structure : Cell
            Crystal structure represented in the form of a Cell.
        name : str
            Name given to denote this structure. Mainly used for book-keeping.
        radii : array-like
            List of radii (floats) for various atoms in the periodic table.
            Use None to leave radii unchanged.

        """
        super(CrystalStructureEntry, self).__init__()
        self.structure = structure
        self.name = name
        if structure.n_atoms() == 0:
            raise Exception("Cannot handle blank crystal structures.")
        self.radii = radii

        # Compute the composition of this crystal.
        self.compute_composition()

    def compute_composition(self):
        """Function to compute the composition of this crystal.

        Raises
        ------
        Exception
            If element name is not recognized.

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
        """Function to create a new entry by replacing elements on this entry.

        Parameters
        ----------
        replacements : dict
            Dictionary of elements to replace. Key: Old element, Value: New
            element.

        Returns
        -------
        new_entry : CrystalStructureEntry
            New entry formed by replacing the current elements with the
            replacement map.

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
        """Function to create a copy of this instance.

        Returns
        -------
        x : CrystalStructureEntry
            A copy of this instance.

        """

        x = type(self)(self.structure, self.name, self.radii)
        x.__dict__.update(self.__dict__)
        x.structure = self.structure.__copy__()
        return x

    def __eq__(self, other):
        """Function to check if this instance equals another instance.

        Parameters
        ----------
        other : CrystalStructureEntry
            Other composition entry to compare.

        Returns
        -------
        output : bool
            True if they are equal and False otherwise.

        """
        if isinstance(other, CrystalStructureEntry):
            return self.structure.__eq__(other.structure) and \
                   super(CrystalStructureEntry, self).__eq__(other)
        return False

    def __cmp__(self, other):
        """Function to compare this instance with another instance.

        Parameters
        ----------
        other : CrystalStructureEntry
            Other composition entry to compare.

        Returns
        -------
        output : int
            -1 if self < other , 1 if self > other or 0 if self = other.

        Raises
        ------
        Exception
            If the entries are equal.
        """

        if isinstance(other, CrystalStructureEntry):
            # First: Check for equality.
            if self.__eq__(other):
                return 0

            # Second: Check the composition / attributes.
            super_comp = super(CrystalStructureEntry, self).__cmp__(other)
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
            return super(CrystalStructureEntry, self).__cmp__(other)

    def __hash__(self):
        """Function to compute the hashcode of this instance.

        Returns
        -------
        output : int
            Hashcode of this instance.

        """
        return super(CrystalStructureEntry, self).__hash__() ^ \
               self.structure.__hash__()

    def get_structure(self):
        """Function to get link to the structure.

        Returns
        -------
        structure : Cell
            Structure this entry represents.
        """

        return self.structure

    def get_name(self):
        """Function to get the name of this entry.

        Returns
        -------
        name : str
            Name given to denote this structure. Mainly used for book-keeping.

        """
        return self.name

    def compute_voronoi_tessellation(self):
        """Function to compute the voronoi tessellation of this structure.

        Returns
        -------
        voronoi : VoronoiCellBasedAnalysis
            Tool used to query properties of the tessellation.

        """

        if self.voronoi is None:
            self.voronoi = VoronoiCellBasedAnalysis(radical=False)
            self.voronoi.analyze_structure(self.structure)
        elif not self.voronoi.tessellation_is_converged():
            raise Exception("Tessellation did not converge.")
        return self.voronoi

    def clear_representations(self):
        """Function to clear out the representations used when computing
        attributes.
        """

        self.voronoi = None

    def __str__(self):
        """Function the generate the string representation of this instance.

        Returns
        -------
        output : str
            Correctly formatted output.

        """

        comp = super(CrystalStructureEntry, self).__str__()
        return self.name+":"+comp

    @classmethod
    def import_structures_list(self, dir_path):
        """Function to read a list of crystal structures from a directory.

        Parameters
        ----------
        dir_path : str
            Path to the directory containing the list of vasp files.

        Returns
        -------
        structures_list : array-like
            A list of CrystalStructureEntry's corresponding to the file
            contents.

        """

        structures_list = []
        radii = LookUpData.load_property("CovalentRadius")

        # Thanks to https://stackoverflow.com/questions/3207219/how-do-i-list
        # -all-files-of-a-directory
        only_files = [f for f in listdir(dir_path) if isfile(join(dir_path,
                                                                  f))]
        io = VASP5IO()
        for f in only_files:
            if f.endswith(".vasp"):
                structure = io.parse_file(join(dir_path,f))
                name = f.split(".vasp")[0]
                entry = CrystalStructureEntry(structure, name=name, radii=radii)
                structures_list.append(entry)

        return structures_list

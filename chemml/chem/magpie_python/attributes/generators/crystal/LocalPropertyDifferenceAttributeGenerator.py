import pandas as pd
import numpy as np
from ....data.materials.CrystalStructureEntry import CrystalStructureEntry
from ....data.materials.util.LookUpData import LookUpData

class LocalPropertyDifferenceAttributeGenerator:
    """Class to compute attributes based on the difference in elemental
    properties between neighboring atoms.

    Attributes
    ----------
    elemental_properties : list
        Elemental properties to be associated with this class for the
        generation of features.
    shells : array-like
        Shells to consider. A list of int values.
    attr_name : str
        Property Name.

    Notes
    -----
    For an atom, its "local property difference" is computed by:

    .. math:: \displaystyle\frac{\sum_n f_n * \left|p_{atom} - p_n\right|}{
    \sum_n f_n}

    where :math: `f_n` is the area of the face associated with neighbor
    :math: `n, p_{atom}` is the elemental property of the central atom,
    and :math: `p_n` is the elemental property of the neighbor atom.

    For shells past the 1st nearest neighbor shell, the neighbors are
    identified by finding all of the unique faces on the outside of the
    polyhedron formed by the previous neighbor shell. This list of faces
    will faces corresponding to all of the atoms in the desired shell and the
    total weight for each atom is defined by the total area of the faces
    corresponding to that atom (there may be more than one).

    By default, this class considers the only the 1st nearest neighbor shell.

    This parameter is computed for all elemental properties stored in
    Composition Entry ElementalProperties.

    """

    def __init__(self, shells=None):
        """Function to create instance and initialize fields.

        Parameters
        ----------
        shells : array-like
            Shells to be considered. A list of int values.

        """

        # Elemental properties used to generate attributes.
        self.elemental_properties = []

        # Shells to consider.
        self.shells = [1] if shells is None else list(set(shells))

        # Property Name.
        self.attr_name = "NeighDiff"

        # Property description (used in description output).
        # self.attr_description = "difference between the elemental
        # properties between an atom and neighbors"

    def clear_shells(self):
        """Function to clear the list of shells.

        """

        self.shells = []

    def add_shell(self, shell):
        """Function to add shell to the list used when computing attributes.

        Parameters
        ----------
        shells : int
            Index of nearest-neighbor shell.

        Raises
        ------
        ValueError
            If shell is negative.

        """

        if shell <= 0:
            raise ValueError("Shell index must be > 0.")
        if shell not in self.shells:
            self.shells.append(shell)

    def add_shells(self, shells):
        """Function to add a list of shells to be used when computing
        attributes.

        Parameters
        ----------
        shells : array-like
            Shells to be considered. A list of int values.

        """

        for s in shells:
            self.add_shell(s)

    def add_elemental_property(self, prop):
        """Function to add an elemental property to `self.elemental_properties`
        in order to be used to compute features.

        Parameters
        ----------
        property : str
            Property to be added.

        """

        if prop not in self.elemental_properties:
            self.elemental_properties.append(prop)

    def add_elemental_properties(self, properties):
        """Function to provide a list of elemental properties to be used to
        compute features.

        Parameters
        ----------
        properties : array-like
            Properties to be included. A list of strings containing property
            names.

        """

        for p in properties:
            self.add_elemental_property(p)

    def remove_elemental_property(self, property):
        """Function to remove an elemental property from
        `self.elemental_properties`.

        Parameters
        ----------
        property : str
            Property to be removed.

        """

        if property in self.elemental_properties:
            self.elemental_properties.remove(property)

    def remove_elemental_properties(self, properties):
        """Function to remove a list of elemental properties from the list of
        elemental properties.

        Parameters
        ----------
        properties : array-like
            Properties to be removed. A list of strings containing property
            names.

        """

        for prop in properties:
            self.remove_elemental_property(prop)

    def clear_elemental_properties(self):
        """Function to clear all the elemental properties.

        """

        self.elemental_properties = []

    def generate_features(self, entries):
        """Function to generate features as mentioned in the class description.

        Parameters
        ----------
        entries : array-like
            Crystal structures for which features are to be generated. A list
            of CrystalStructureEntry's.

        Returns
        ----------
        features : DataFrame
            Features for the given entries. Pandas data frame containing the
            names and values of the descriptors.

        Raises
        ------
        ValueError
            If input is not of type list.
            If items in the list are not CrystalStructureEntry instances.

        """

        # Initialize list of feature values for pandas data frame.
        feat_values = []

        # Raise exception if input argument is not of type list of
        # CompositionEntry's.
        if not isinstance(entries, list):
            raise ValueError("Argument should be of type list of "
                             "CrystalStructureEntry's")
        elif (entries and not isinstance(entries[0], CrystalStructureEntry)):
            raise ValueError("Argument should be of type list of "
                             "CrystalStructureEntry's")

        # Insert header names here.
        prefix = ["mean_", "var_", "min_", "max_", "range_"]
        feat_headers =[p+self.attr_name+"_shell{}".format(s)+"_"+prop for s
                       in self.shells for prop in self.elemental_properties
                       for p in prefix]

        # Generate features.
        l_fh = len(feat_headers)
        for entry in entries:
            tmp_list = []
            # Get the Voronoi tessellation.
            try:
                voro = entry.compute_voronoi_tessellation()
            except Exception:
                tmp_list = [np.nan] * l_fh  # If tessellation fails.
                feat_values.append(tmp_list)
                continue

            # Get the elements corresponding to each type.
            st = entry.get_structure()
            elem_index = np.array([LookUpData.element_names.index(
                            st.get_type_name(i)) for i in range(st.n_types())])

            # Loop through each shell.
            for shell in self.shells:
                # Loop through each elemental property.
                for prop in self.elemental_properties:
                    # Get properties for elements in this structure.
                    lookup_table = LookUpData.load_property(prop)
                    prop_values = np.array([lookup_table[elem_index[i]] for i
                                            in range(len(elem_index))])

                    # Compute neighbor differences.
                    neigh_diff = self.get_atom_properties(voro, shell,
                                                          prop_values)
                    mean = np.mean(neigh_diff)
                    tmp_list.append(mean)
                    tmp_list.append(np.mean([abs(x - mean) for x in
                                             neigh_diff]))
                    min_ = np.min(neigh_diff)
                    max_ = np.max(neigh_diff)
                    tmp_list.append(min_)
                    tmp_list.append(max_)
                    tmp_list.append(max_ - min_)

            feat_values.append(tmp_list)

        features = pd.DataFrame(feat_values, columns=feat_headers)
        return features

    def get_atom_properties(self, voro, shell, prop_values):
        """Function to compute the properties of a certain neighbor cell for
        each atom, given the Voronoi tessellation and properties of each atom
        type.

        Parameters
        ----------
        voro : VoronoiCellBasedAnalysis
            Analysis tool.
        shell : int
            Index of shell.
        prop_values : array-like
            Properties of each atom type. A list or NumPy array of float values.

        Returns
        -------
        output : array-like
            Properties of each atom. A list or NumPy array of float values.

        """

        output = voro.neighbor_property_differences(prop_values, shell)
        return output
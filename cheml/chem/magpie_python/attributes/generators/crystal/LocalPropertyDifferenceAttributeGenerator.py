import os
import pandas as pd
import numpy as np
import types
from data.materials.CrystalStructureEntry import CrystalStructureEntry
from data.materials.util.LookUpData import LookUpData

class LocalPropertyDifferenceAttributeGenerator:
    """
    Class to compute attributes based on the difference in elemental
    properties between neighboring atoms. For an atom, its "local property
    difference" is computed by:

    sum_n f_n * |p_atom - p_n| / sum_n f_n

    where f_n is the area of the face associated with neighbor n,
    p_atom the the elemental property of the central atom, and
    p_n is the elemental property of the neighbor atom.

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
        """
        Function to create instance and initialize fields.
        :param shells: List of shells to be considered.
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
        """
        Function to clear the list of shells.
        :return:
        """
        self.shells = []

    def add_shell(self, shell):
        """
        Function to add shell to the list used when computing attributes.
        :param shell: Index of nearest-neighbor shell.
        :return:
        """
        if shell <= 0:
            raise ValueError("Shell index must be > 0.")
        if shell not in self.shells:
            self.shells.append(shell)

    def add_shells(self, shells):
        """
        Function to add a list of shells to be used when computing attributes.
        :param shells: List of shells.
        :return:
        """
        for s in shells:
            self.add_shell(s)

    def add_elemental_property(self, prop):
        """
        Function to add elemental properties to be used when computing
        attributes.
        :param prop: Desired property.
        :return:
        """
        if prop not in self.elemental_properties:
            self.elemental_properties.append(prop)

    def add_elemental_properties(self, properties):
        """
        Function to add a list of elemental properties to be used when
        computing attributes.
        :param properties: List of desired properties.
        :return:
        """
        for p in properties:
            self.add_elemental_property(p)

    def remove_elemental_property(self, prop):
        """
        Function to remove elemental properties to be used when computing
        attributes.
        :param prop: Desired property.
        :return:
        """
        if prop in self.elemental_properties:
            self.elemental_properties.remove(prop)

    def remove_elemental_properties(self, properties):
        """
        Function to remove a list of elemental properties to be used when
        computing attributes.
        :param properties: List of desired properties.
        :return:
        """
        for p in properties:
            self.remove_elemental_property(p)

    def clear_elemental_properties(self):
        """
        Function to clear all the elemental properties.
        :return:
        """
        self.elemental_properties = []

    def generate_features(self, entries, verbose=False):
        """
        Function to generate features as mentioned in the class description.
        :param entries: A list of CrystalStructureEntry's.
        :param verbose: Flag that is mainly used for debugging. Prints out a
        lot of information to the screen.
        :return features: Pandas data frame containing the names and values
        of the descriptors.
        """

        # Initialize list of feature values for pandas data frame.
        feat_values = []

        # Raise exception if input argument is not of type list of
        # CompositionEntry's.
        if (type(entries) is not types.ListType):
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
        if verbose:
            print features.head()
        return features

    def get_atom_properties(self, voro, shell, prop_values):
        """
        Function to compute the properties of a certain neighbor cell for
        each atom, given the Voronoi tessellation and properties of each atom
        type.
        :param voro: Voronoi tessellation.
        :param shell: Index of shell.
        :param prop_values: Properties of each atom type.
        :return: Properties of each atom.
        """
        return voro.neighbor_property_differences(prop_values, shell)
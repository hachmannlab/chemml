import numpy as np
import pandas as pd
import types
from data.materials.CrystalStructureEntry import CrystalStructureEntry
from data.materials.util.LookUpData import LookUpData
from models.regression.crystal.PRDFRegression import PRDFRegression

class PRDFAttributeGenerator:
    """
    Class to compute attributes based on the Pair Radial Distribution
    Function (PRDF). Based on work by Schutt et al.
    http://link.aps.org/doi/10.1103/PhysRevB.89.205118.
    """

    def __init__(self):
        """
        Function to create instance and initialize fields.
        """

        # Cutoff distance for PRDF.
        self.cut_off_distance = 10

        # Number of distance points to evaluate.
        self.n_points = 20

        # List of Elements to use in PRDF.
        self.element_list = []

    def set_cut_off_distance(self, d):
        """
        Function to set the maximum distance to consider when computing the
        PRDF.
        :param d: Desired cutoff distance.
        :return:
        """
        self.cut_off_distance = d

    def set_n_points(self, n_p):
        """
        Function to set the number of points on each PRDF to store.
        :param n_p: Number of evaluation points.
        :return:
        """
        self.n_points = n_p

    def clear_element_list(self):
        """
        Function to clear out the elements in element list.
        :return:
        """
        self.element_list = []

    def set_elements(self, entries):
        """
        Function to set the elements when computing PRDF.
        :param data: A list of CompositionEntry's containing each element to
        be added.
        :return:
        """
        self.clear_element_list()
        for entry in entries:
            for elem in entry.get_element_ids():
                if elem not in self.element_list:
                    self.add_element(id=elem)

    def add_element(self, id=None, name=None):
        """
        Function to add element to list used when computing PRDF.
        :param id: ID of element (Atomic number - 1).
        :param name: Name of the element.
        :return:
        """
        if id is not None:
            self.element_list.append(id)
        elif name is None:
            raise ValueError("Either id or name must be provided to locate "
                             "element to be added.")
        elif name not in LookUpData.element_names:
            raise ValueError("No such element: "+name)
        else:
            self.element_list.append(LookUpData.element_names.index(name))

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
        # CrystalStructureEntry's.
        if (type(entries) is not types.ListType):
            raise ValueError("Argument should be of type list of "
                             "CrystalStructureEntry's")
        elif (entries and not isinstance(entries[0], CrystalStructureEntry)):
            raise ValueError("Argument should be of type list of "
                             "CrystalStructureEntry's")

        # Get names of elements in set.
        self.element_list.sort()
        elem_names = [LookUpData.element_names[i] for i in self.element_list]
        elem_ids = list(self.element_list)

        # Get the step size for the PRDF.
        step_size = self.cut_off_distance / self.n_points

        # Insert header names here.
        feat_headers = ["{:s}_{:s}_R={:3f}".format(elem_a, elem_b, step *
                        step_size) for elem_a in elem_names for elem_b in
                        elem_names for step in range(1, self.n_points + 1)]

        # Initialize PRDF tool.
        tool = PRDFRegression()
        tool.set_cut_off(self.cut_off_distance)
        tool.set_n_bins(self.n_points)

        l_fh = len(feat_headers)
        # Generate features for each entry.
        for entry in entries:
            # Compute the PRDF.
            prdf = tool.compute_representation(entry.get_structure())
            tmp_array = np.zeros(l_fh, dtype=float)
            for pair in prdf:
                # Determine position in output.
                elem_a = elem_ids.index(pair[0])
                elem_b = elem_ids.index(pair[1])

                prdf_pos = (elem_a * len(self.element_list) + elem_b) * \
                           self.n_points
                tmp_array[prdf_pos: (prdf_pos + self.n_points)] = prdf[pair][
                                    0 : self.n_points]

            feat_values.append(tmp_array)

        features = pd.DataFrame(feat_values, columns=feat_headers)
        if verbose:
            print features.head()
        return features
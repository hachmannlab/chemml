# coding=utf-8
import numpy as np
import pandas as pd
import types
from ....data.materials.CrystalStructureEntry import CrystalStructureEntry
from ....data.materials.util.LookUpData import LookUpData
from ....models.regression.crystal.PRDFRegression import PRDFRegression

class PRDFAttributeGenerator:
    """Class to compute attributes based on the Pair Radial Distribution
    Function (PRDF).
    Based on work by Schutt et al. [1].

    Attributes
    ----------
    cut_off_distance : float
        Cutoff distance for PRDF.
    n_points : int
        Number of distance points to evaluate.
    element_list : array-like
        Elements to use in PRDF. A list of int values.

    References
    ----------
    .. [1] K. T. Schütt, H. Glawe, F. Brockherde, A. Sanna, K. R. Müller,
    and E. K. U. Gross, "How to represent crystal structures for machine
    learning: Towards fast prediction of electronic properties," Physical
    Review B, vol. 89, no. 20, May 2014.

    """

    def __init__(self):
        """Function to create instance and initialize fields.

        """

        # Cutoff distance for PRDF.
        self.cut_off_distance = 10

        # Number of distance points to evaluate.
        self.n_points = 20

        # List of Elements to use in PRDF.
        self.element_list = []

    def set_cut_off_distance(self, d):
        """Function to set the maximum distance to consider when computing the
        PRDF.

        Parameters
        ----------
        d : float
            Desired cutoff distance.

        """

        self.cut_off_distance = d

    def set_n_points(self, n_p):
        """Function to set the number of points on each PRDF to store.

        Parameters
        ----------
        n_p : int
            Number of evaluation points.

        """

        self.n_points = n_p

    def clear_element_list(self):
        """Function to clear out the elements in element list.

        """

        self.element_list = []

    def set_elements(self, entries):
        """Function to set the elements when computing PRDF.

        Parameters
        ----------
        data : array-like
            A list of CompositionEntry's containing each element to be added.

        """

        self.clear_element_list()
        for entry in entries:
            for elem in entry.get_element_ids():
                if elem not in self.element_list:
                    self.add_element(id=elem)

    def add_element(self, id=None, name=None):
        """Function to add element to list used when computing PRDF.

        Parameters
        ----------
        id   : int
            ID of element (Atomic number - 1).
        name : str
            Name of the element.

        Raises
        ------
        ValueError
            If both arguments are None.
            If entered element name can not be found in database.

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
        # CrystalStructureEntry's.
        if not isinstance(entries, list):
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
        return features

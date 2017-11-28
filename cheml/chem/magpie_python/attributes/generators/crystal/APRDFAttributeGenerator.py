import types
import pandas as pd
from data.materials.CrystalStructureEntry import CrystalStructureEntry
from data.materials.util.LookUpData import LookUpData
from vassal.analysis.APRDFAnalysis import APRDFAnalysis

class APRDFAttributeGenerator:
    """
    Class to generate attributes based on the Atomic Property Weighted Radial
    Distribution Function (AP-RDF) approach of Fernandez et al.
    http://pubs.acs.org/doi/abs/10.1021/jp404287t

    User can specify the cutoff distance for the AP-RDF, the number of points
    to evaluate it, the smoothing factors for the RDF peaks, and the
    properties used for weighting. The recommended values of these
    parameters have yet to be determined, please contact Logan Ward or the
    authors of this paper if you have questions or ideas for these parameters.
    """

    def __init__(self):
        # Cutoff distance for RDF.
        self.cut_off_distance = 10.0

        # Number of points to evaluate.
        self.num_points = 6

        # Smoothing parameter for AP-RDF.
        self.smooth_parameter = 4.0

        # List of elemental properties to use for weighting.
        self.elemental_properties = []

    def set_smoothing_parameter(self, b):
        """
        Function to set smoothing factor used when computing PRDF.
        :param b: Smoothing factor.
        :return:
        """
        self.smooth_parameter = b

    def set_cut_off_distance(self, d):
        """
        Function to set cut off distance used when computing PRDF.
        :param d: Cut off distance.
        :return:
        """
        self.cut_off_distance = d

    def set_num_points(self, num_points):
        """
        Function to set the number of points at which to evaluate AP-RDF.
        :param num_points: Desired number of windows.
        :return:
        """
        self.num_points = num_points

    def add_elemental_property(self, property_name):
        """
        Function to add elemental property to set of those used for
        generating attributes.
        :param property_name: Name of property to be added.
        :return:
        """
        if property_name not in self.elemental_properties:
            self.elemental_properties.append(property_name)

    def clear_elemental_properties(self):
        """
        Function to clear the list of elemental properties.
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

        # Create tool to compute AP-RDF.
        tool = APRDFAnalysis()
        tool.set_cut_off_distance(self.cut_off_distance)
        tool.set_n_windows(self.num_points)
        tool.set_smoothing_factor(self.smooth_parameter)

        # Insert header names here.
        eval_dist = tool.get_evaluation_distances()
        feat_headers = ["APRDF_{:s}_R={:f}_B={:f}".format(prop, dist,
                        self.smooth_parameter) for prop in
                        self.elemental_properties for dist in eval_dist]

        # Loop through each entry, compute attributes.
        for entry in entries:
            # Get the structure.
            structure = entry.get_structure()

            # Prepare the APRDF tool.
            tool.analyze_structure(structure)

            # Loop through each property.
            for prop in self.elemental_properties:
                prop_lookup = LookUpData.load_property(prop)
                atom_prop = [prop_lookup[LookUpData.element_names.index(
                    structure.get_type_name(t))] for t in range(
                    structure.n_types())]

                # Compute the APRDF.
                ap_rdf = tool.compute_APRDF(atom_prop)

                feat_values.append(ap_rdf)

        features = pd.DataFrame(feat_values, columns=feat_headers)
        if verbose:
            print features.head()
        return features
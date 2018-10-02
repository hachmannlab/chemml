# coding=utf-8
import types
import pandas as pd
from ....data.materials.CrystalStructureEntry import CrystalStructureEntry
from ....data.materials.util.LookUpData import LookUpData
from ....vassal.analysis.APRDFAnalysis import APRDFAnalysis

class APRDFAttributeGenerator:
    """Class to generate attributes based on the Atomic Property Weighted
    Radial Distribution Function (AP-RDF) approach of Fernandez et al. [1].

    User can specify the cutoff distance for the AP-RDF, the number of points
    to evaluate it, the smoothing factors for the RDF peaks, and the
    properties used for weighting. The recommended values of these
    parameters have yet to be determined, please contact Logan Ward or the
    authors of this paper if you have questions or ideas for these parameters.

    Attributes
    ----------
    cut_off_distance : float
        Cutoff distance for RDF.
    num_points : int
        Number of points to evaluate.
    smooth_parameter : float
        Smoothing parameter for AP-RDF.
    elemental_properties : list
        Elemental properties to be associated with this class for the
        generation of features.

    References
    ----------
    .. [1] M. Fernandez, N. R. Trefiak, and T. K. Woo, "Atomic Property
    Weighted Radial Distribution Functions Descriptors of Metal–Organic
    Frameworks for the Prediction of Gas Uptake Capacity," The Journal of
    Physical Chemistry C, vol. 117, no. 27, pp. 14095--14105, Jul. 2013.

    """

    def __init__(self):
        """Function to create instance and initialize fields.

        """

        # Cutoff distance for RDF.
        self.cut_off_distance = 10.0

        # Number of points to evaluate.
        self.num_points = 6

        # Smoothing parameter for AP-RDF.
        self.smooth_parameter = 4.0

        # List of elemental properties to use for weighting.
        self.elemental_properties = []

    def set_smoothing_parameter(self, b):
        """Function to set smoothing factor used when computing PRDF.

        Parameters
        ----------
        b : float
            Smoothing factor.

        """

        self.smooth_parameter = b

    def set_cut_off_distance(self, d):
        """Function to set cut off distance used when computing PRDF.

        Parameters
        ----------
        d : float
            Cut off distance.

        """

        self.cut_off_distance = d

    def set_num_points(self, num_points):
        """Function to set the number of points at which to evaluate AP-RDF.

        Parameters
        ----------
        num_points : int
            Desired number of windows.

        """

        self.num_points = num_points

    def add_elemental_property(self, property_name):
        """Function to add an elemental property to `self.elemental_properties`
        in order to be used to compute features.

        Parameters
        ----------
        property : str
            Property to be added.

        """

        if property_name not in self.elemental_properties:
            self.elemental_properties.append(property_name)

    def add_elemental_properties(self, properties):
        """Function to provide a list of elemental properties to be used to
        compute features.

        Parameters
        ----------
        properties : array-like
            Properties to be included. A list of strings containing property
            names.

        """

        for prop in properties:
            self.add_elemental_property(prop)

    def clear_elemental_properties(self):
        """Function to clear the list of elemental properties.

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
        return features
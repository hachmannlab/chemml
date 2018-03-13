from ....attributes.generators.crystal\
    .LocalPropertyDifferenceAttributeGenerator import \
    LocalPropertyDifferenceAttributeGenerator

class LocalPropertyVarianceAttributeGenerator(
    LocalPropertyDifferenceAttributeGenerator):
    """Class to compute attributes based on the local variance in elemental
    properties around each atom.

    See Also
    --------
    LocalPropertyDifferenceAttributeGenerator : Super class of this class.

    """

    def __init__(self, shells=None):
        """Function to create instance and initialize fields.

        Parameters
        ----------
        shells : array-like
            Shells to be considered. A list of int values.

        """

        LocalPropertyDifferenceAttributeGenerator.__init__(self, shells)
        self.attr_name = "NeighVar"

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

        output = voro.neighbor_property_variances(prop_values, shell)
        return output
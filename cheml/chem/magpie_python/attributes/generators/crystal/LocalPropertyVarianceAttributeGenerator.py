from attributes.generators.crystal.LocalPropertyDifferenceAttributeGenerator \
    import LocalPropertyDifferenceAttributeGenerator

class LocalPropertyVarianceAttributeGenerator(
    LocalPropertyDifferenceAttributeGenerator):
    """
    Class to compute attributes based on the local variance in elemental
    properties around each atom. Similar to
    LocalPropertyDifferenceAttributeGenerator class. See that class
    description for more information.
    """
    def __init__(self, shells=None):
        """
        Function to create instance and initialize fields.
        :param shells: List of shells to be considered.
        """
        LocalPropertyDifferenceAttributeGenerator.__init__(self, shells)
        self.attr_name = "NeighVar"

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
        return voro.neighbor_property_variances(prop_values, shell)
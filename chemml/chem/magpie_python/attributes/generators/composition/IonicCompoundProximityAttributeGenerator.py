import types
import pandas as pd
from ....data.materials.CompositionEntry import CompositionEntry
from ....data.utilities.filters.CompositionDistanceFilter import \
    CompositionDistanceFilter
from ....utility.tools.IonicCompoundFinder import IonicCompoundFinder

class IonicCompoundProximityAttributeGenerator:
    """Class to generate attributes based on the distance of a composition from
    a compositions that can form charge-neutral ionic compounds.

    Attributes
    ----------
    max_formula_unit : int
        Maximum number of atoms per formula unit.

    Notes
    -----
    This generator only computes a single feature: the L_1 distance between the
    composition of an entry and the nearest ionic compound (determined using
    IonicCompoundFinder). For a compound where it is not possible to form an
    ionic compound (e.g., only metallic elements), the entry is assigned
    arbitrarily large distance (equal to the number of elements in the alloy).
    The one adjustable parameter in this calculation is the maximum number of
    atoms per formula unit used when looking for ionic compounds. For binary
    compounds, the maximum conceivable number of elements in a formula unit
    is for a compound with a 9+ and a 5- species, which has 14 atoms in the
    formula unit. Consequently, we recommend using 14 or larger for this
    parameter.

    """

    # Maximum number of atoms per formula unit.
    max_formula_unit = 14

    def set_max_formula_unit(self, size):
        """Function to define the maximum number of atoms per formula unit.

        Parameters
        ----------
        size : int
            Desired size.

        """

        self.max_formula_unit = size

    def generate_features(self, entries):
        """Function to generate features as mentioned in the class description.

        Parameters
        ----------
        entries : array-like
            Compositions for which features are to be generated. A list of
            CompositionEntry's.

        Returns
        ----------
        features : DataFrame
            Features for the given entries. Pandas data frame containing the
            names and values of the descriptors.

        Raises
        ------
        ValueError
            If input is not of type list.
            If items in the list are not CompositionEntry instances.

        """

        # Initialize lists of feature values and headers for pandas data frame.
        feat_values = []
        feat_headers = []

        # Raise exception if input argument is not of type list of
        # CompositionEntry's.
        if not isinstance(entries, list):
            raise ValueError("Argument should be of type list of "
                             "CompositionEntry's")
        elif (entries and not isinstance(entries[0], CompositionEntry)):
            raise ValueError("Argument should be of type list of "
                             "CompositionEntry's")

        # Insert header names here.
        feat_headers.append("IonicCompoundDistance_MaxSize"+str(
            self.max_formula_unit))

        # Get ionic compound finder.
        finder = IonicCompoundFinder()
        finder.set_max_formula_unit_size(self.max_formula_unit)

        cdf = CompositionDistanceFilter()
        for entry in entries:
            elems = entry.get_element_ids()
            # Set the maximum distance to be equal to the number of elements.
            #  The maximum possible L_1 distance for an N-element system is N.
            finder.set_maximum_distance(len(elems))

            # If the material has only 1 element, set feature to 1.0.
            if len(elems) == 1:
                feat_values.append(1.0)
            else:
                # Get the list of all ionic compounds in the system.
                finder.set_nominal_composition(entry)
                ionic_compounds = finder.find_all_compounds()

                # Find the distance to the closest one. If no other
                # compounds, set distance to be the maximum possible.
                if not ionic_compounds:
                    feat_values.append(len(elems))
                else:
                    # print ionic_compounds[0]
                    # print cdf.compute_distance(entry, ionic_compounds[0], 1)
                    feat_values.append(cdf.compute_distance(entry,
                                        ionic_compounds[0], 1))

        features = pd.DataFrame(feat_values, columns=feat_headers)
        return features
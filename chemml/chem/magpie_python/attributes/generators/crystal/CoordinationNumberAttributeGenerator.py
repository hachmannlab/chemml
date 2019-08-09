import types
import pandas as pd
from ....data.materials.CrystalStructureEntry import CrystalStructureEntry

class CoordinationNumberAttributeGenerator:
    """Class to compute attributes based on the coordination number. Uses the
    Voronoi tessellation to define the coordination network.
    """

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

        # Initialize lists of feature values and headers for pandas data frame.
        feat_values = []
        feat_headers = []

        # Raise exception if input argument is not of type list of
        # CrystalStructureEntry's.


        if not isinstance(entries, list):
            raise ValueError("Argument should be of type list of "
                             "CrystalStructureEntry's")
        elif (entries and not isinstance(entries[0], CrystalStructureEntry)):
            raise ValueError("Argument should be of type list of "
                             "CrystalStructureEntry's")

        feat_headers.append("mean_Coordination")
        feat_headers.append("var_Coordination")
        feat_headers.append("min_Coordination")
        feat_headers.append("max_Coordination")

        for entry in entries:
            temp_list = []

            output = entry.compute_voronoi_tessellation()

            mean = output.face_count_average()
            variance = output.face_count_variance()
            minimum = output.face_count_minimum()
            maximum = output.face_count_maximum()

            temp_list.append(mean)
            temp_list.append(variance)
            temp_list.append(minimum)
            temp_list.append(maximum)

            feat_values.append(temp_list)

        features = pd.DataFrame(feat_values, columns=feat_headers)
        return features

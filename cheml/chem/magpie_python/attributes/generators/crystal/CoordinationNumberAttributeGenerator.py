import types
import pandas as pd
from data.materials.CrystalStructureEntry import CrystalStructureEntry

class CoordinationNumberAttributeGenerator:
    """
    Class to compute attributes based on the coordination number. Uses the
    Voronoi tessellation to define the coordination network.

    DEV NOTE (LW 15Jul15): Could benefit from adding a face size cutoff, where
    atoms are only defined as coordinated if the face between them is larger
    than a certain fraction of the surface area of both cells. Otherwise
    faces on the cells that are only present to numerical issues will be
    counted as neighbors. Metallic glass community commonly removes any faces
    smaller than 1% of the total surface area of a cell.
    """

    def generate_features(self, entries, verbose=False):
        """
        Function to generate the charge dependent features as mentioned in
        the class description.
        :param entries: A list of CrystalStructureEntry's.
        :param verbose: Flag that is mainly used for debugging. Prints out a
        lot of information to the screen.
        :return features: Pandas data frame containing the names and values
        of the descriptors.
        """

        # Initialize lists of feature values and headers for pandas data frame.
        feat_values = []
        feat_headers = []

        # Raise exception if input argument is not of type list of
        # CrystalStructureEntry's.


        if (type(entries) is not types.ListType):
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
        if verbose:
            print features.head()
        return features
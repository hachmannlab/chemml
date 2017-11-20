import types
import pandas as pd
from data.materials.CompositionEntry import CompositionEntry
from data.materials.util.LookUpData import LookUpData

class ElementFractionAttributeGenerator:
    """
    Class to set the element fractions as the features of materials.
    """

    def generate_features(self, entries, verbose=False):
        """
        Function to set element fractions as features of materials.
        :param entries: A list of dictionaries containing <Element name,
        fraction> as <key,value> pairs.
        :param verbose: Flag that is mainly used for debugging. Prints out a
        lot of information to the screen.
        :return features: Pandas data frame containing the names and values
        of the descriptors.
        """

        # Initialize lists of feature values and headers for pandas data frame.
        feat_values = []
        feat_headers = []

        # Raise exception if input argument is not of type list of
        # CompositionEntry's.
        if (type(entries) is not types.ListType):
            raise ValueError("Argument should be of type list of "
                             "CompositionEntry's")
        elif (entries and not isinstance(entries[0], CompositionEntry)):
            raise ValueError("Argument should be of type list of "
                             "CompositionEntry's")

        # Insert feature headers here.
        for elem in LookUpData.element_names:
            feat_headers.append("X_"+elem)

        # Insert feature values here.
        for entry in entries:
            tmp_list = []
            elem_ids = entry.get_element_ids()
            elem_fracs = entry.get_element_fractions()
            for elem in range(len(LookUpData.element_names)):
                if elem in elem_ids:
                    idx = elem_ids.index(elem)
                    tmp_list.append(elem_fracs[idx])
                else:
                    tmp_list.append(0.0)
            feat_values.append(tmp_list)

        features = pd.DataFrame(feat_values, columns=feat_headers)
        if verbose:
            print features.head()
        return features
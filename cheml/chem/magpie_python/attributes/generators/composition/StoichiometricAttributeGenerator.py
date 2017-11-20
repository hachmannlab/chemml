import types
import pandas as pd
from data.materials.CompositionEntry import CompositionEntry

class StoichiometricAttributeGenerator:
    """
    Class to set up and generate descriptors based on the stoichiometry of a
    given material. Includes features that are only based on fractions of
    elements, but not what those elements actually are.
    """

    # List of p norms to compute.
    def __init__(self, use_default_norms=True):
        self.p_norms = []
        if use_default_norms:
            self.add_p_norms([2, 3, 5, 7, 10])

    def clear_p_norms(self):
        """
        Clear out the list of p norms to be computed.
        :return:
        """
        del self.p_norms[:]

    def add_p_norm(self, norm):
        """
        Add a p norm to be computed.
        :param norm: desired norm.
        :return:
        """
        if (norm == 0):
            return
        elif (norm == 1):
            raise ValueError("L1 norm is always 1. Useless as attribute.")
        self.p_norms.append(norm)

    def add_p_norms(self, norms):
        """
        Add a list of p norms to be computed.
        :param norms: list of desired norms.
        :return:
        """
        for norm in norms:
            self.add_p_norm(norm)

    def generate_features(self, entries, verbose=False):
        """
        Function to generate the stoichiometric features. Computes the norms
        based on elemental fractions.
        :param entries: A list of CompositionEntry's.
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

        # Issue warning if no p norms are added.
        if (not self.p_norms):
            print "Warning: only L0 norm is computed."

        # Add in feature names.
        feat_headers.append("NComp")
        for p in self.p_norms:
            feat_headers.append("Comp_L"+str(p)+"Norm")

        # Compute features.
        for entry in entries:
            fracs = entry.get_element_fractions()
            tmp_list = []
            n_comp = 0
            for f in fracs:
                if (f > 0):
                    n_comp += 1
            # Number of components.
            tmp_list.append(n_comp)

            # Lp norms.
            for p in self.p_norms:
                tmp = 0.0
                for f in fracs:
                    tmp += f**p
                tmp_list.append(tmp**(1.0/p))
            feat_values.append(tmp_list)

        # features as a pandas data frame.
        features = pd.DataFrame(feat_values, columns=feat_headers)
        if (verbose):
            print features.head()
        return features
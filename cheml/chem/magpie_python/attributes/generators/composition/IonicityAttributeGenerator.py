import math
import types
import pandas as pd
from data.materials.CompositionEntry import CompositionEntry
from data.materials.util.LookUpData import LookUpData
from utility.tools.OxidationStateGuesser import OxidationStateGuesser

class IonicityAttributeGenerator:
    """
    Class to generate the attributes based on the ionicity of a compound.
    Creates attributes based on whether it is possible to form a
    charge-neutral ionic compound, and two features based on a simple measure
    "bond ionicity" I(x,y)
    (see-http://www.wiley.com/WileyCDA/WileyTitle/productCd-EHEP002505.html
    W.D. Callister's text):

    I(x,y) = 1 - exp(-0.25* (chi_x - chi_y)^2)

    Maximum ionic character: Max I(x,y) between any two constituents.
    Mean ionic character: Sum x_i*x_j* I(i,j) where x_i is the fraction of
    element i and chi_x is the electronegativity of element x.
    """

    def generate_features(self, entries, verbose=False):
        """
        Function to generate the three "ionicity" based features as described
        in the description of the class.
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

        # Insert header names here.
        feat_headers.append("CanFormIonic")
        feat_headers.append("MaxIonicChar")
        feat_headers.append("MeanIonicChar")

        # Instantiate and initialize oxidation state guesser with
        # electronegativity and oxidation state values.
        ox_guesser = OxidationStateGuesser()
        en = LookUpData.load_property("Electronegativity")
        ox_guesser.set_electronegativity(en)
        ox_guesser.set_oxidationstates(LookUpData.load_property(
            "OxidationStates"))

        for entry in entries:
            tmp_list = []
            # Can it form an ionic compound?
            tmp_list.append(0 if len(ox_guesser.get_possible_states(entry))
                                 == 0 else 1)
            tmp_en = []

            # Compute mean ionic character.
            mean_ionic = 0.0
            elems = entry.get_element_ids()
            fracs = entry.get_element_fractions()
            for i,elem1 in enumerate(elems):
                tmp_en.append(en[elem1])
                for j,elem2 in enumerate(elems):
                    m = 1 - math.exp(-0.25*(en[elem1] - en[elem2])**2)

                    mean_ionic += fracs[i] * fracs[j] * m

            # Compute max ionic character.
            max_ionic = 1 - math.exp(-0.25 * (max(tmp_en) - min(tmp_en)) ** 2)
            tmp_list.append(max_ionic)
            tmp_list.append(mean_ionic)
            feat_values.append(tmp_list)

        features = pd.DataFrame(feat_values, columns=feat_headers)
        if verbose:
            print features.head()

        return features
import types
import numpy as np
import pandas as pd
from data.materials.CompositionEntry import CompositionEntry
from data.materials.util.LookUpData import LookUpData

class ValenceShellAttributeGenerator:
    """
    Class that generates attributes based on fraction of electrons in valence
    shell of constituent elements. Creates 4 feature: [Composition-weighted
    mean # of electrons in the {s,p,d,f} shells]/[Mean # of Valence Electrons]

    Originally presented by:
    http://journals.aps.org/prb/abstract/10.1103/PhysRevB.89.094104
    Meredig et al. Physical Review B (2015)
    """

    def generate_features(self, entries, lookup_path, verbose=False):
        """
        Function that generates the attributes mentioned in the class
        description above.
        :param entries: A list of CompositionEntry's.
        :param lookup_path: Path to the file containing the property values.
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

        shell = ['s','p','d','f']
        n_valence = np.zeros((len(shell), len(LookUpData.element_names)))

        # Read in the data from tables and insert feature headers here.
        for i in range(len(shell)):
            s = shell[i]
            feat_headers.append("frac_"+s+"Valence")
            n_valence[i] = LookUpData.load_property("N"+s+"Valence",
                                                    lookup_dir=lookup_path)

        for entry in entries:
            elems = entry.get_element_ids()
            fracs = entry.get_element_fractions()
            sum_e = 0.0
            total_e = []
            for i in range(len(shell)):
                tmp_valence = []
                for elem_id in elems:
                    tmp_valence.append(n_valence[i][elem_id])

                # Fraction weighted average # of electrons in this shell.
                x = np.average(tmp_valence, weights=fracs)
                sum_e += x
                total_e.append(x)

            for i in range(len(total_e)):
                total_e[i] /= sum_e

            feat_values.append(total_e)

        features = pd.DataFrame(feat_values, columns=feat_headers)
        if verbose:
            print features.head()
        return features
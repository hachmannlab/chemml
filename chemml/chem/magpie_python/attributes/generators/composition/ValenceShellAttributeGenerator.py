import types
import numpy as np
import pandas as pd
from ....data.materials.CompositionEntry import CompositionEntry
from ....data.materials.util.LookUpData import LookUpData

class ValenceShellAttributeGenerator:
    """Class that generates attributes based on fraction of electrons in
    valence shell of constituent elements.
    Creates 4 features: [Composition-weighted mean # of electrons in the {s,p,
    d,f} shells]/[Mean # of Valence Electrons]
    Originally presented by: Meredig et al. [1].

    References
    ----------
    .. [1] B. Meredig et al., "Combinatorial screening for new materials in
    unconstrained composition space with machine learning," Physical Review
    B, vol. 89, no. 9, Mar. 2014.
    """

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

        shell = ['s','p','d','f']
        n_valence = np.zeros((len(shell), len(LookUpData.element_names)))

        # Read in the data from tables and insert feature headers here.
        for i in range(len(shell)):
            s = shell[i]
            feat_headers.append("frac_"+s+"Valence")
            n_valence[i] = LookUpData.load_property("N"+s+"Valence")

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
        return features

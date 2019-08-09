import types
import numpy as np
import pandas as pd
from ....data.materials.CompositionEntry import CompositionEntry
from ....data.materials.util.LookUpData import LookUpData

class MeredigAttributeGenerator:
    """Class to generate attributes as described by Meredig et al. [1].

    Notes
    -----
    This class is meant to be used in conjunction with
    ElementFractionAttributeGenerator and ValenceShellAttributeGenerator.
    To match the attributes from the Meredig et al. [1] paper, use all three
    attribute generators.

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

        # Insert feature headers here.
        feat_headers.append("mean_AtomicWeight")
        feat_headers.append("mean_Column")
        feat_headers.append("mean_Row")
        feat_headers.append("maxdiff_AtomicNumber")
        feat_headers.append("mean_AtomicNumber")
        feat_headers.append("maxdiff_CovalentRadius")
        feat_headers.append("mean_CovalentRadius")
        feat_headers.append("maxdiff_Electronegativity")
        feat_headers.append("mean_Electronegativity")
        feat_headers.append("mean_NsValence")
        feat_headers.append("mean_NpValence")
        feat_headers.append("mean_NdValence")
        feat_headers.append("mean_NfValence")

        # Load all property tables.
        mass = LookUpData.load_property("AtomicWeight")
        column = LookUpData.load_property("Column")
        row = LookUpData.load_property("Row")
        number = LookUpData.load_property("Number")
        radius = LookUpData.load_property("CovalentRadius")
        en = LookUpData.load_property("Electronegativity")
        s = LookUpData.load_property("NsValence")
        p = LookUpData.load_property("NpValence")
        d = LookUpData.load_property("NdValence")
        f = LookUpData.load_property("NfValence")

        for entry in entries:
            tmp_list = []
            elem_fractions = entry.get_element_fractions()
            elem_ids = entry.get_element_ids()
            tmp_mass = []
            tmp_column = []
            tmp_row = []
            tmp_number = []
            tmp_radius = []
            tmp_en = []
            tmp_s = []
            tmp_p = []
            tmp_d = []
            tmp_f = []
            for elem_id in elem_ids:
                tmp_mass.append(mass[elem_id])
                tmp_column.append(column[elem_id])
                tmp_row.append(row[elem_id])
                tmp_number.append(number[elem_id])
                tmp_radius.append(radius[elem_id])
                tmp_en.append(en[elem_id])
                tmp_s.append(s[elem_id])
                tmp_p.append(p[elem_id])
                tmp_d.append(d[elem_id])
                tmp_f.append(f[elem_id])

            tmp_list.append(np.average(tmp_mass, weights=elem_fractions))
            tmp_list.append(np.average(tmp_column, weights=elem_fractions))
            tmp_list.append(np.average(tmp_row, weights=elem_fractions))
            tmp_list.append(max(tmp_number) - min(tmp_number))
            tmp_list.append(np.average(tmp_number, weights=elem_fractions))
            tmp_list.append(max(tmp_radius) - min(tmp_radius))
            tmp_list.append(np.average(tmp_radius, weights=elem_fractions))
            tmp_list.append(max(tmp_en) - min(tmp_en))
            tmp_list.append(np.average(tmp_en, weights=elem_fractions))
            tmp_list.append(np.average(tmp_s, weights=elem_fractions))
            tmp_list.append(np.average(tmp_p, weights=elem_fractions))
            tmp_list.append(np.average(tmp_d, weights=elem_fractions))
            tmp_list.append(np.average(tmp_f, weights=elem_fractions))

            feat_values.append(tmp_list)

        features = pd.DataFrame(feat_values, columns=feat_headers)
        return features

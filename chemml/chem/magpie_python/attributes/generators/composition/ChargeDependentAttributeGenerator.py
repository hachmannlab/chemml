# coding=utf-8
import sys
from builtins import range
import numpy as np
import pandas as pd
from ....data.materials.CompositionEntry import CompositionEntry
from ....data.materials.util.LookUpData import LookUpData
from ....utility.tools.OxidationStateGuesser import OxidationStateGuesser

class ChargeDependentAttributeGenerator:
    """Class to generate attributes derived from the oxidation states of
    elements in a material.
    Based on work by Deml et al.[1].

    Notes
    -----
    These features are based on the formal charges of materials determined
    using the OxidationStateGuesser. Currently implemented features:
    Statistics of formal charges (min, max, range, mean, variance)
    Cumulative ionization energies/ electron affinities
    Difference in electronegativities between cation and anion.
    For materials that the algorithm fails to find charge states, NaN is set
    for all features.

    References
    ----------
    .. [1] A. M. Deml, R. O’Hayre, C. Wolverton, and V. Stevanović,
    "Predicting density functional theory total energies and enthalpies of
    formation of metal-nonmetal compounds by linear regression," Physical
    Review B, vol. 93, no. 8, Feb. 2016.

    """

    def generate_features(self, entries):
        """
        Function to generate features as mentioned in the class description.

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
        n_features = 8
        feat_headers.append("min_Charge")
        feat_headers.append("max_Charge")
        feat_headers.append("maxdiff_Charge")
        feat_headers.append("mean_Charge")
        feat_headers.append("var_Charge")
        feat_headers.append("CumulativeIonizationEnergy")
        feat_headers.append("CumulativeElectronAffinity")
        feat_headers.append("AnionCationElectronegativityDiff")

        # Load properties here.
        en = LookUpData.load_property("Electronegativity")
        ea = LookUpData.load_property("ElectronAffinity")
        ie = LookUpData.load_property("IonizationEnergies")

        # Instantiate and initialize oxidation state guesser.
        ox_guesser = OxidationStateGuesser()
        ox_guesser.set_electronegativity(en)
        ox_guesser.set_oxidationstates(LookUpData.load_property(
            "OxidationStates"))

        missing_data = {}
        for entry in entries:
            tmp_list = []
            elems = entry.get_element_ids()
            fracs = entry.get_element_fractions()

            # Get possible states with charges.
            possible_states = ox_guesser.get_possible_states(entry)

            # If there are no possible states, set all features to NaN.
            if len(possible_states) == 0:
                for i in range(n_features):
                    tmp_list.append(np.nan)

                feat_values.append(tmp_list)
                continue

            # Check that we have data for all ionization energies.
            any_missing = False
            tmp_charges = possible_states[0]
            for i,elem in enumerate(elems):
                if len(ie[elem]) < tmp_charges[i]:
                    if elem not in missing_data:
                        missing_data[elem] = []
                    if possible_states[0][i] not in missing_data[elem]:
                        missing_data[elem].append(possible_states[0][i])
                    any_missing = True
                    break

            # Compute statistics related to charges.
            min_ = min(tmp_charges)
            max_ = max(tmp_charges)
            max_diff_ = max_ -  min_
            mean_ = np.average([abs(x) for x in tmp_charges], weights=fracs)
            var_ = np.average([abs(abs(x) - mean_) for x in tmp_charges],
                              weights=fracs)

            tmp_list.append(min_)
            tmp_list.append(max_)
            tmp_list.append(max_diff_)
            tmp_list.append(mean_)
            tmp_list.append(var_)

            if any_missing:
                tmp_list.append(np.nan)
                tmp_list.append(np.nan)
                tmp_list.append(np.nan)
                feat_values.append(tmp_list)
                continue

            # Compute features related to ionization/affinity.
            cation_fraction = anion_fraction = cation_ie_sum = anion_ea_sum = \
            mean_cation_en = mean_anion_en = 0.0
            for i,elem in enumerate(elems):
                if tmp_charges[i] < 0:
                    anion_fraction += fracs[i]
                    mean_anion_en +=  en[elem] * fracs[i]
                    anion_ea_sum -= tmp_charges[i] * ea[elem]* fracs[i]
                else:
                    cation_fraction += fracs[i]
                    mean_cation_en += en[elem] * fracs[i]
                    cation_ie_sum += sum(ie[elem][c] * fracs[i] for c in
                                         range(int(tmp_charges[i])))

            mean_anion_en /= anion_fraction
            mean_cation_en /= cation_fraction
            anion_ea_sum /= anion_fraction
            cation_ie_sum /= cation_fraction

            tmp_list.append(cation_ie_sum)
            tmp_list.append(anion_ea_sum)
            tmp_list.append(mean_anion_en - mean_cation_en)

            feat_values.append(tmp_list)

        # Issue warning to user about missing data here if it exists.
        if len(missing_data) > 0:
            sys.stderr.write("WARNING: Missing ionization energy data for:\n")
            for elem in missing_data:
                sys.stderr.write("\t" + LookUpData.element_names[elem] + ":")
                for state in missing_data[elem]:
                    sys.stderr.write(" +" + str(state))
                sys.stderr.write("\n")

        features = pd.DataFrame(feat_values, columns=feat_headers)

        return features

# coding=utf-8
import math
import types
import numpy as np
import pandas as pd
from ....data.materials.CompositionEntry import CompositionEntry
from ....data.materials.util.LookUpData import LookUpData

class YangOmegaAttributeGenerator:
    """Class to compute the attributes :math:`\Omega` and :math:`\delta`
    developed by Yang and Zhang [1].
    These parameters are based on the liquid formation enthalpy and atomic
    sizes of elements respectively and were originally developed to predict
    whether a metal alloy will form a solid solution of bulk metallic glass.

    Notes
    -----
    :math: `\Omega` is derived from the melting temperature, ideal mixing
    entropy, and regular solution solution interaction parameter (
    :math: `\Omega_{i,j}`) predicted by the Miedema model for binary liquids.
    Specifically, it is computed using the relationship:
    .. math:: \Omega = \displaystyle\frac{T_m \Delta S_{mix}} {|\Delta H_{mix}|}
    where :math: `T_m` is the composition-weighted average of the melting
    temperature, :math: `\Delta S_{mix}` is the ideal solution entropy,
    and :math: `\Delta H_{mix}` is the mixing enthalpy. The mixing enthalpy
    is computed using the Miedema mixing enthalpies tabulated by Takeuchi and
    Inoue [2] where:
    .. math:: \Delta H_{mix} = \displaystyle\sum \Omega_{i,j} c_i c_j
    and :math: `\Omega_{i,j} = 4 * \Delta H_{mix}`.
    :math: `\delta` is related to the polydispersity of atomic sizes, and is
    computed using the relationship:
    .. math:: \delta = [\displaystyle\sum c_i (1 - \frac{r_i}{r_{
    average})^2]^0.5
    where :math: `r_i` is the atomic size. Here, we use the atomic radii
    compiled by Miracle et al. [3] rather than those compiled by Kittel,
    as in the original work.

    References
    ----------
    .. [1] X. Yang and Y. Zhang, "Prediction of high-entropy stabilized
    solid-solution in multi-component alloys," Materials Chemistry and
    Physics, vol. 132, no. 2--3, pp. 233--238, Feb. 2012.
    .. [2] A. Takeuchi and A. Inoue, "Classification of Bulk Metallic Glasses
    by Atomic Size Difference, Heat of Mixing and Period of Constituent
    Elements and Its Application to Characterization of the Main Alloying
    Element," MATERIALS TRANSACTIONS, vol. 46, no. 12, pp. 2817--2829, 2005.
    .. [3] D. B. Miracle, D. V. Louzguine-Luzgin, L. V. Louzguina-Luzgina,
    and A. Inoue, "An assessment of binary metallic glasses: correlations
    between structure, glass forming ability and stability," International
    Materials Reviews, vol. 55, no. 4, pp. 218--256, Jul. 2010.

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

        # Insert header names here.
        feat_headers.append("Yang_Omega")
        feat_headers.append("Yang_delta")

        # Load property values here.
        radii = LookUpData.load_property("MiracleRadius")
        meltingT = LookUpData.load_property("MeltingT")
        miedema = LookUpData.load_pair_property("MiedemaLiquidDeltaHf")

        for entry in entries:
            tmp_list = []
            tmp_radii = []
            tmp_meltingT = []

            elem_fracs = entry.get_element_fractions()
            elem_ids = entry.get_element_ids()
            for elem_id in elem_ids:
                tmp_radii.append(radii[elem_id])
                tmp_meltingT.append(meltingT[elem_id])

            # Compute the average melting point.
            averageTm = np.average(tmp_meltingT, weights=elem_fracs)

            # Compute the ideal entropy.
            entropy = 0.0
            for f in elem_fracs:
                entropy += f*math.log(f) if f > 0 else 0.0
            entropy *= 8.314/1000

            # Compute the enthalpy
            enthalpy = 0.0

            for i in range(len(elem_ids)):
                for j in range(i + 1, len(elem_ids)):
                    enthalpy += miedema[max(elem_ids[i], elem_ids[j])][min(
                        elem_ids[i], elem_ids[j])] * elem_fracs[i] * \
                                elem_fracs[j]
            enthalpy *= 4

            # Compute omega
            tmp_list.append(abs(averageTm * entropy / enthalpy))

            # Compute delta
            delta_squared = 0.0
            average_r = np.average(tmp_radii, weights=elem_fracs)
            for i in range(len(elem_ids)):
                delta_squared += elem_fracs[i] * (1 - tmp_radii[i] /
                                                  average_r)**2

            tmp_list.append(math.sqrt(delta_squared))

            feat_values.append(tmp_list)

        features = pd.DataFrame(feat_values, columns=feat_headers)
        return features

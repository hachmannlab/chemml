import types
from math import sqrt, log
import numpy as np
import pandas as pd
from ....data.materials.CompositionEntry import CompositionEntry
from ....data.materials.util.GCLPCalculator import GCLPCalculator

class GCLPAttributeGenerator:
    """Class to compute features based on the T=0K ground state.

    Attributes
    ----------
    GCLPCalculator : GCLPCalculator
        A GCLPCalculator instance.
    count_phases : bool
        Flag to include or exclude the number of phases at equilibrium.

    Notes
    -----
    Features:
    1. Formation energy.
    2. Number of phases in equilibrium.
    3. Distance from closest composition (i.e., ||x_i - x_{i,f}||_2 for each
    component i for phase f).
    4. Average distance from all neighbors.
    5. Quasi-entropy (sum x_i * ln(x_i) where x_i is fraction of phase).
    Certain values of the number of phases in equilibrium and "quasi-entropy"
    are only accessible to systems with larger number of elements. Useful if
    you do not want to consider the number of components in an alloy as a
    predictive variable.

    """

    # Tool used to compute ground states.
    GCLPCalculator = None

    # Whether to include the number of phases at equilibrium.
    count_phases = True

    def set_phases(self, phases, energies):
        """Function to define phases used when computing ground states.

        Parameters
        ----------
        phases : array-like
            Compositions to consider. A list of CompositionEntry's.
        energies : array-like
            Corresponding energies. A list of float values.

        """

        self.GCLPCalculator = GCLPCalculator()
        self.GCLPCalculator.add_phases(phases, energies)

    def set_count_phases(self, count_phases):
        """Function to set variable to count number of phases at equilibrium.
        In some cases, you may want to exclude this as a feature because it is
        tied to the number of components in the compound.

        Parameters
        ----------
        count_phases : bool
            Desired setting.

        """

        self.count_phases = count_phases

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

        # Check if the GCLP calculation has been defined.
        if not self.GCLPCalculator:
            raise ValueError("GCLP calculator has not been setup.")

        feat_headers.append("T0K:Enthalpy")
        if self.count_phases:
            feat_headers.append("T0K:NPhasesEquilibrium")
        feat_headers.append("T0K:ClosestPhaseDistance")
        feat_headers.append("T0K:MeanPhaseDistance")
        if self.count_phases:
            feat_headers.append("T0K:QuasiEntropy")

        for entry in entries:
            tmp_list = []

            # Run GCLP.
            l,r = self.GCLPCalculator.run_GCLP(entry)

            # Compute formation energy.
            tmp_list.append(l)

            # Compute number of phases.
            if self.count_phases:
                tmp_list.append(len(r))

            # Compute distances.
            phase_distances = []
            elements = entry.get_element_ids()
            fractions = entry.get_element_fractions()

            for phase in r:
                dist = 0.0
                for i,elem in enumerate(elements):
                    diff = phase.get_element_fraction(id=elem) - fractions[i]
                    dist += diff * diff
                phase_distances.append(sqrt(dist))

            tmp_list.append(min(phase_distances))
            tmp_list.append(np.mean(phase_distances))

            # Compute quasi-entropy.
            if self.count_phases:
                entropy = 0.0
                for f in r.values():
                    entropy += f * log(f)
                tmp_list.append(entropy)

            feat_values.append(tmp_list)

        features = pd.DataFrame(feat_values, columns=feat_headers)
        return features
import types
import pandas as pd
import numpy as np
from data.materials.CrystalStructureEntry import CrystalStructureEntry

class EffectiveCoordinationNumberAttributeGenerator:
    """
    Compute attributes based on the effective coordination number. The
    effective coordination number can be thought of as a face-size-weighted
    coordination number. It is computed by the formula

    N_eff = 1 / sum[(f_i / SA)^2]

    where f_i is the area of face i and SA is the surface area of the entire
    cell.

    The effective coordination number has major benefit: stability against the
    additional of a very small face. Small perturbations in atomic positions
    can break symmetry in a crystal, and lead to the introduction of small
    faces. The conventional coordination number treats all faces equally,
    so the coordination number changes even when one of these small faces is
    added.

    One approach in the literature is to first apply a screen on small
    faces (e.g., remove any smaller than 1% of the total face area),
    which still runs into problems with discontinuity for larger displacements.

    Our approach is differentiable with respect to the additional of a small
    face (ask Logan if you want the math), and also captures another
    interesting effect small coordination numbers for Voronoi cells with a
    dispersity in face sizes. For example, BCC has 14 faces on its voronoi
    cell. 8 large faces, and 6 small ones. Our effective face size identifies a
    face size of closer to 8, the commonly-accepted value of the BCC
    coordination number, than 14 reported by the conventional measure.
    Additional, for systems with equal-sized faces (e.g., FCC), this measure
    agrees exactly with conventional reports.
    """

    def mean_abs_dev(self, data):
        n = float(len(data))
        mean = sum(data) / n
        diff = [abs(x - mean) for x in data]
        return sum(diff) / n

    def generate_features(self, entries, verbose=False):
        """
        Function to generate features as mentioned in the class description.
        :param entries: A list of CrystalStructureEntry's.
        :param verbose: Flag that is mainly used for debugging. Prints out a
        lot of information to the screen.
        :return features: Pandas data frame containing the names and values
        of the descriptors.
        """

        # Raise exception if input argument is not of type list of
        # CrystalStructureEntry's.

        if (type(entries) is not types.ListType):
            raise ValueError("Argument should be of type list of "
                             "CrystalStructureEntry's")
        elif (entries and not isinstance(entries[0], CrystalStructureEntry)):
            raise ValueError("Argument should be of type list of "
                             "CrystalStructureEntry's")

        # Initialize lists of feature values and headers for pandas data frame.
        feat_headers = []
        feat_values = []

        feat_headers.append("mean_Coordination")
        feat_headers.append("var_Coordination")
        feat_headers.append("min_Coordination")
        feat_headers.append("max_Coordination")

        l_fh = len(feat_headers)
        for entry in entries:
            temp_list = []
            try:
                output = entry.compute_voronoi_tessellation()
            except Exception:
                tmp_list = [np.nan] * l_fh # If tessellation fails.
                feat_values.append(tmp_list)
                continue
            N_eff = output.get_effective_coordination_numbers()

            mean = np.mean(N_eff)
            absdev = self.mean_abs_dev(data=N_eff)
            minimum = np.min(N_eff)
            maximum = np.max(N_eff)

            temp_list.append(mean)
            temp_list.append(absdev)
            temp_list.append(minimum)
            temp_list.append(maximum)

            feat_values.append(temp_list)

        features = pd.DataFrame(feat_values, columns=feat_headers)

        if verbose:
            print features.head()

        return features
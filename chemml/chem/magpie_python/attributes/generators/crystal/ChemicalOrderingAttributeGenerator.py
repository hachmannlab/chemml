import numpy as np
import pandas as pd
from ....data.materials.CrystalStructureEntry import CrystalStructureEntry

class ChemicalOrderingAttributeGenerator:
    """Class to compute attributes based on chemical ordering of structure.
    Determines average Warren-Cowley ordering parameter for the bond network
    defined by the Voronoi tessellation of a structure.

    Attributes
    ----------
    shells : list
        Index of shells to compute features for.
    weighted : bool
        Whether to compute features using weighting or not.

    See Also
    --------
    VoronoiCellBasedAnalysis.get_neighbor_ordering_parameters : Computes
    Warren-Cowley ordering parameters.

    Notes
    -----
    For each atom in the structure, the average Warren-Cowley ordering
    parameter is determined by computing the average magnitude of ordering
    parameter for each type for all atoms in a structure. The ordering
    parameter is 0 for a perfectly-random distribution, so this average
    represents an average degree of "ordering" in the structure. This
    attribute is computed for several nearest-neighbor shells (1st, 2nd,
    and 3rd by default).

    There are two options for computing order parameters: Weighted and
    unweighted. The former is computed by weighing the contribution of each
    neighboring atom by the fraction of surface area corresponding to
    boundaries between that atom and the central atom. The former considers
    all neighbors weighted equally, which means they are very sensitive to
    the introduction of small faces due to numerical problems inherent to the
    Voronoi tessellation. Full details is available in the Vassal
    documentation for VoronoiCellBasedAnalysis.getNeighborOrderingParameters().

    """

    def __init__(self):
        """Function to create instance and initialize fields.

        Will create the WC parameters for the first, second and third
        nearest-neighbor shells by default.

        """

        # Shells to compute the WC attribute for.
        self.shells = [1, 2, 3]

        # Whether to compute weighted WC ordering parameters.
        self.weighted = True

    def set_shells(self, shells):
        """Function to set which nearest-neighbor shells to consider when
        generating features.

        Parameters
        ----------
        shells: list
            Desired shell indices.

        """

        self.shells = list(shells)

    def set_weighted(self, weighted):
        """Function to set whether to consider face sizes when computing
        ordering parameters.

        Parameters
        ----------
        weighted: bool
            Whether to weigh using face sizes.

        """

        self.weighted = weighted

    def generate_features(self, entries):
        """Function to generate features as mentioned in the class description.

        Parameters
        ----------
        entries : array-like
            Crystal structures for which features are to be generated. A list
            of CrystalStructureEntry's.

        Returns
        ----------
        features : DataFrame
            Features for the given entries. Pandas data frame containing the
            names and values of the descriptors.

        Raises
        ------
        ValueError
            If input is not of type list.
            If items in the list are not CrystalStructureEntry instances.

        """

        # Initialize list of feature values for pandas data frame.
        feat_values = []

        # Raise exception if input argument is not of type list of
        # CrystalStructureEntry's.
        if not isinstance(entries, list):
            raise ValueError("Argument should be of type list of "
                             "CrystalStructureEntry's")
        elif (entries and not isinstance(entries[0], CrystalStructureEntry)):
            raise ValueError("Argument should be of type list of "
                             "CrystalStructureEntry's")

        # Insert header names here.
        feat_headers = ["mean_WCMagnitude" + ("" if self.weighted else
                        "_unweighted_") + "_shell" + str(shell) for shell in
                        self.shells]

        # Compute features.
        for entry in entries:
            tmp_list = []
            # Get the Voronoi tessellation.
            try:
                voro = entry.compute_voronoi_tessellation()
            except Exception:
                tmp_list = [np.nan] * len(feat_headers)
                feat_values.append(tmp_list)
                continue

            tmp_list = [voro.warren_cowley_ordering_magnitude(shell=shell,
                        weighted=self.weighted) for shell in self.shells]
            feat_values.append(tmp_list)

        features = pd.DataFrame(feat_values, columns=feat_headers)
        return features

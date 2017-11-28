import pandas as pd
import numpy as np
import types
from data.materials.CrystalStructureEntry import CrystalStructureEntry

class PackingEfficiencyAttributeGenerator:
    """
    Class to compute attributes based on packing efficiency. Packing
    efficiency is determined by finding the largest sphere that would fit
    inside each Voronoi cell and comparing the volume of that sphere to the
    volume of the cell.

    For now, the only attribute computed by this generator is the maximum
    packing efficiency for the entire cell. This is computed by summing the
    total volume of all spheres in all cells, and dividing by the volume of
    the unit cell.
    """
    def generate_features(self, entries, verbose=False):
        """
        Function to generate features as mentioned in the class description.
        :param entries: A list of CrystalStructureEntry's.
        :param verbose: Flag that is mainly used for debugging. Prints out a
        lot of information to the screen.
        :return features: Pandas data frame containing the names and values
        of the descriptors.
        """

        # Initialize list of feature values for pandas data frame.
        feat_values = []

        # Raise exception if input argument is not of type list of
        # CompositionEntry's.
        if (type(entries) is not types.ListType):
            raise ValueError("Argument should be of type list of "
                             "CrystalStructureEntry's")
        elif (entries and not isinstance(entries[0], CrystalStructureEntry)):
            raise ValueError("Argument should be of type list of "
                             "CrystalStructureEntry's")

        # Insert header names here.
        feat_headers = ["MaxPackingEfficiency"]

        # Generate features for each entry.
        for entry in entries:
            tmp_list = []
            # Get the Voronoi tessellation.
            try:
                voro = entry.compute_voronoi_tessellation()
            except Exception:
                tmp_list = [np.nan] # If tessellation fails.
                feat_values.append(tmp_list)
                continue
            tmp_list.append(voro.max_packing_efficiency())
            feat_values.append(tmp_list)

        features = pd.DataFrame(feat_values, columns=feat_headers)
        if verbose:
            print features.head()
        return features
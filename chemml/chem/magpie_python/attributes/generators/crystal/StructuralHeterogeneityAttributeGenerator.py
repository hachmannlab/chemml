import pandas as pd
import numpy as np
import types
from ....data.materials.CrystalStructureEntry import CrystalStructureEntry

class StructuralHeterogeneityAttributeGenerator:
    """Class to compute attributes based on heterogeneity in structure.
    Measures variance in bond lengths (both for a single atom and between
    different atoms) and atomic volumes. Also considers the number of unique
    coordination polyhedron shapes.
    Bond lengths, atomic volumes, and coordination polyhedra are based on the
    Voronoi tessellation of the structure.

    Notes
    -----
    Current attributes:
    1. Mean absolute deviation in average bond length for each atom, normalized
    by mean for all atoms.
    2. Minimum in average bond length, normalized by mean for all atoms.
    3. Maximum in average bond length, normalized by mean for all atoms.
    4. Mean bond length variance between bonds across all atom.
    5. Mean absolute deviation in bond length variance.
    6. Minimum bond length variance.
    7. Maximum bond length variance.
    8. Mean absolute deviation in atomic volume, normalized by mean atomic
    volume.

    Here, bond length variation for a single atom is defined as:
    .. math:: \hat{l} = <l_i - \bar{l}>
    where :math: `l_i` is the distance between an atom and one of its
    neighbors.

    """

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

        # Initialize lists of feature values and headers for pandas data frame.
        feat_values = []
        feat_headers = []

        # Raise exception if input argument is not of type list of
        # CompositionEntry's.
        if not isinstance(entries, list):
            raise ValueError("Argument should be of type list of "
                             "CrystalStructureEntry's")
        elif (entries and not isinstance(entries[0], CrystalStructureEntry)):
            raise ValueError("Argument should be of type list of "
                             "CrystalStructureEntry's")

        # Insert header names here.
        feat_headers.append("var_MeanBondLength")
        feat_headers.append("min_MeanBondLength")
        feat_headers.append("max_MeanBondLength")
        feat_headers.append("mean_BondLengthVariation")
        feat_headers.append("var_BondLengthVariation")
        feat_headers.append("min_BondLengthVariation")
        feat_headers.append("max_BondLengthVariation")
        feat_headers.append("var_CellVolume")

        l_fh = len(feat_headers)
        # Generate features for each entry.
        for entry in entries:
            tmp_list = []
            # Get the Voronoi tessellation.
            try:
                voro = entry.compute_voronoi_tessellation()
            except Exception:
                tmp_list = [np.nan] * l_fh # If tessellation fails.
                feat_values.append(tmp_list)
                continue

            # Bond length features.
            # Variation between cells.
            mean_bond_lengths = voro.mean_bond_lengths()
            mean_bond_lengths /= np.mean(mean_bond_lengths) # Normalize bond
            # lengths.
            m = np.mean(mean_bond_lengths)
            tmp_list.append(np.mean([abs(x - m) for x in mean_bond_lengths]))
            tmp_list.append(np.min(mean_bond_lengths))
            tmp_list.append(np.max(mean_bond_lengths))

            # Variation within a single cell.
            mean_bond_lengths = voro.mean_bond_lengths() # Recompute bond
            # lengths.
            bond_length_variation = voro.bond_length_variance(
                mean_bond_lengths)
            # Normalize bond length variation by mean bond length of each cell.
            bond_length_variation /= mean_bond_lengths
            m = np.mean(bond_length_variation)
            tmp_list.append(m)
            tmp_list.append(np.mean([abs(x - m) for x in
                                     bond_length_variation]))
            tmp_list.append(np.min(bond_length_variation))
            tmp_list.append(np.max(bond_length_variation))

            # Cell volume / shape features.
            tmp_list.append(voro.volume_variance() * entry.get_structure(
            ).n_atoms() / entry.get_structure().volume())

            feat_values.append(tmp_list)

        features = pd.DataFrame(feat_values, columns=feat_headers)
        return features

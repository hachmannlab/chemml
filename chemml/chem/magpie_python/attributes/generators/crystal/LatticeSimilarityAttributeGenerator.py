import pandas as pd
import numpy as np
import types
from ....data.materials.CrystalStructureEntry import CrystalStructureEntry

class LatticeSimilarityAttributeGenerator:
    """Compute similarity of structure to several simple lattices.

    Notes
    -----
    Determined by comparing the shape of each coordination polyhedron in the
    structure (as determined using a Voronoi tessellation) to those in a
    reference lattice.

    Similarity is computed by summing the difference in the number of faces
    with each number of edges between a certain Voronoi cell and that of the
    reference lattice. This difference is then normalized by the number of
    faces in the reference lattice, and averaged over all atoms to produce a
    "similarity index". In this form, structures based on the reference
    lattice have a match of 0, which becomes larger with increase
    dissimilarity.

    For now we consider the BCC, FCC (which has the same coordination
    polyhedron shape as HCP), and SC lattices.

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

        # Raise exception if input argument is not of type list of
        # CrystalStructureEntry's.

        if not isinstance(entries, list):
            raise ValueError("Argument should be of type list of "
                             "CrystalStructureEntry's")
        elif (entries and not isinstance(entries[0], CrystalStructureEntry)):
            raise ValueError("Argument should be of type list of "
                             "CrystalStructureEntry's")

        feat_headers = []
        feat_values = []

        feat_headers.append("dissimilarity_FCC")
        feat_headers.append("dissimilarity_BCC")
        feat_headers.append("dissimilarity_SC")

        l_fh = len(feat_headers)
        for entry in entries:
            temp_list = []
            try:
                output = entry.compute_voronoi_tessellation()
            except Exception:
                tmp_list = [np.nan] * l_fh # If tessellation fails.
                feat_values.append(tmp_list)
                continue

            fcc = output.mean_fcc_dissimilarity()
            bcc = output.mean_bcc_dissimilarity()
            sc = output.mean_sc_dissimilarity()

            temp_list.append(fcc)
            temp_list.append(bcc)
            temp_list.append(sc)

            feat_values.append(temp_list)

        features = pd.DataFrame(feat_values, columns=feat_headers)
        return features
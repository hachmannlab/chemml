import math
import numpy as np
from data.materials.util.LookUpData import LookUpData
from vassal.analysis.PairDistanceAnalysis import PairDistanceAnalysis

class PRDFRegression:
    """
    Class that uses the partial radial distribution function to perform
    regression. Each material is represented by a matrix containing the PRDF
    between each element type. Distance between structures are computed as
    Frobenius norm of the difference between their PRDF matrices. If you use
    this method, please cite Schutt, Glawe, et al. PRB (2015)
    http://link.aps.org/doi/10.1103/PhysRevB.89.205118.
    """

    def __init__(self):
        """
        Function to create instance and initialize fields.
        """

        # Normalization term in kernel function.
        self.sigma = 1

        # Number of bins in PRDF.
        self.n_bins = 25

        # Cutoff distance of PRDF.
        self.cut_off = 7.0

    def set_sigma(self, s):
        """
        Function to set the normalization parameter in the kernel function.
        :param s: Desired normalization parameter.
        :return:
        """
        self.sigma = s

    def set_n_bins(self, n_b):
        """
        Function to set the number of bins used when computing the PRDF.
        :param n_b: Number of bins (>1)
        :return:
        """
        if n_b < 1:
            raise ValueError("# bins must be greater than 1.")
        self.n_bins = n_b

    def set_cut_off(self, d):
        """
        Function to set the cutoff distance used when computing the PRDF.
        :param d: Cutoff distance (>0 Angstrom)
        :return:
        """
        if d <= 0:
            raise ValueError("Distance must be positive.")
        self.cut_off = d

    def compute_similarity(self, structure1, structure2):
        """
        Function to compute similarity between two crystal structures.
        :param structure1: Representation of structure #1.
        :param structure2: Representation of structure #2.
        :return: Similarity between the two structures.
        """
        pairs = set()

        # Compile a complete list of the element pairs for both structures.
        for k in structure1.keys():
            pairs.add(k)
        for k in structure2.keys():
            pairs.add(k)

        # For each pair, compute the squared differences between the two PRDFs.
        # This is equivalent to the Froebius norm.
        diff = 0.0
        for pair in pairs:
            if pair not in structure1:
                # Assume prdf1 == 0.
                diff += sum(structure2[pair] ** 2)
            elif pair not in structure2:
                # Assume prdf2 == 0.
                diff += sum(structure1[pair] ** 2)
            else:
                diff += sum((structure1[pair] - structure2[pair]) ** 2)

        # Compute kernel function to get similarity.
        return math.exp(-1 * diff / self.sigma)

    def compute_representation(self, structure):
        """
        Function to compute the representation of the crystal structure.
        :param structure: Crystal structure.
        :return: Representation of the structure.
        """

        # Get the atomic number of each type.
        n_types = structure.n_types()
        type_z = np.zeros(n_types, dtype=int)
        for i in range(n_types):
            if structure.get_type_name(i) in LookUpData.element_names:
                type_z[i] = LookUpData.element_names.index(
                    structure.get_type_name(i))
            else:
                raise Exception(
                    "No such element: " + structure.get_type_name(i))

        # Compute the PRDF of this structure.
        pda = PairDistanceAnalysis()
        pda.set_cutoff_distance(self.cut_off)

        try:
            pda.analyze_structure(structure)
        except Exception:
            raise RuntimeError("Oops, something went wrong in analyzing "
                               "structure.")

        prdf = pda.compute_PRDF(self.n_bins)
        rep = {}
        for i in range(len(prdf)):
            for j in range(len(prdf[i])):
                rep[(type_z[i], type_z[j])] = prdf[i][j]

        return rep
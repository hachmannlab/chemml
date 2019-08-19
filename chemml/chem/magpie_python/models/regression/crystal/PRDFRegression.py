# coding=utf-8
import math
import numpy as np
from ....data.materials.util.LookUpData import LookUpData
from ....vassal.analysis.PairDistanceAnalysis import PairDistanceAnalysis

class PRDFRegression:
    """Class that uses the partial radial distribution function to perform
    regression.
    Each material is represented by a matrix containing the PRDF
    between each element type. Distance between structures are computed as
    Frobenius norm of the difference between their PRDF matrices. If you use
    this method, please cite Schutt, Glawe, et al. [1].

    Attributes
    ----------
    sigma : float
        Normalization term in kernel function.
    n_bins : int
        Number of distance points to evaluate.
    cut_off_distance : float
        Cutoff distance for PRDF.

    References
    ----------
    .. [1] K. T. Schütt, H. Glawe, F. Brockherde, A. Sanna, K. R. Müller,
    and E. K. U. Gross, "How to represent crystal structures for machine
    learning: Towards fast prediction of electronic properties," Physical
    Review B, vol. 89, no. 20, May 2014.

    """

    def __init__(self):
        """Function to create instance and initialize fields.
        """

        # Normalization term in kernel function.
        self.sigma = 1

        # Number of bins in PRDF.
        self.n_bins = 25

        # Cutoff distance of PRDF.
        self.cut_off = 7.0

    def set_sigma(self, s):
        """Function to set the normalization parameter in the kernel function.

        Parameters
        ----------
        s : float
            Desired normalization parameter.

        """
        self.sigma = s

    def set_n_bins(self, n_b):
        """Function to set the number of bins used when computing the PRDF.

        Parameters
        ----------
        n_b : int
            Number of bins (>1)

        Raises
        ------
        ValueError
            If n_bins is less than 1.

        """
        if n_b < 1:
            raise ValueError("# bins must be greater than 1.")
        self.n_bins = n_b

    def set_cut_off(self, d):
        """Function to set the cutoff distance used when computing the PRDF.

        Parameters
        ----------
        d : float
            Cutoff distance (>0 Angstrom)

        Raises
        ------
        ValueError
            If d is less than or equal to 0.
        """
        if d <= 0:
            raise ValueError("Distance must be positive.")
        self.cut_off = d

    def compute_similarity(self, structure1, structure2):
        """Function to compute similarity between two crystal structures.

        Parameters
        ----------
        structure1 : dict
            Representation of structure #1. Dictionary containing a tuple of
            integers as the key and a list of floats as the values.
        structure2 : dict
            Representation of structure #2. Dictionary containing a tuple of
            integers as the key and a list of floats as the values.

        Returns
        -------
        output : float
            Similarity between the two structures.

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
        """Function to compute the representation of the crystal structure.

        Parameters
        ----------
        structure : CrystalStructureEntry
            Crystal structure.

        Returns
        -------
        output : dict
            Representation of the structure. Dictionary containing a tuple of
            integers as the key and a list of floats as the values.

        Raises
        ------
        Exception
            If element name is invalid.
        RuntimeError
            If something goes wrong while analyzing structure.
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

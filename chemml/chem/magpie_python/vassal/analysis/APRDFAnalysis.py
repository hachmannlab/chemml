# coding=utf-8
import math
from .PairDistanceAnalysis import PairDistanceAnalysis
import numpy as np

class APRDFAnalysis:
    """Class to compute the Atomic Property Weighted Radial Distribution
    Function (AP-RDF).
    Follows the work by Fernandez et al.[1]. Here, we use a scaling factor
    equal to: 1 / #atoms.

    Attributes
    ----------
    structure : Cell
        Link to structure being evaluated.
    cut_off_distance : float
        Cutoff distance used when computing radial distribution function.
    n_windows : int
        Number of points of the RDF to sample.
    B : float
        Smoothing factor in PRDF expression.
    accuracy_factor : float
        Accuracy factor. Determines the number of atoms farther than the
        cutoff radius that are included in the computation of the PRDF. This
        is necessary due to the Gaussian cutoff term in the RDF.

    References
    ----------
    .. [1] M. Fernandez, N. R. Trefiak, and T. K. Woo, "Atomic Property
    Weighted Radial Distribution Functions Descriptors of Metal--Organic
    Frameworks for the Prediction of Gas Uptake Capacity," The Journal of
    Physical Chemistry C, vol. 117, no. 27, pp. 14095--14105, Jul. 2013.

    """
    def __init__(self):
        """Function to create instance and initialize fields.

        Parameters
        ----------
        structure : Cell
            Link to structure being evaluated.
        cut_off_distance : float
            Cutoff distance used when computing radial distribution function.
        n_windows : int
            Number of points of the RDF to sample.
        B : float
            Smoothing factor in PRDF expression.
        accuracy_factor : float
            Accuracy factor. Determines the number of atoms farther than the
            cutoff radius that are included in the computation of the PRDF. This
            is necessary due to the Gaussian cutoff term in the RDF.
        """

        # Link to structure being evaluated.
        self.structure = None

        # Cutoff distance used when computing radial distribution function.
        self.cut_off_distance = 10

        # Number of points of the RDF to sample.
        self.n_windows = 10

        # Smoothing factor in PRDF expression.
        self.B = 2

        # Accuracy factor. Determines the number of atoms farther than the
        # cutoff radius that are included in the computation of the PRDF.
        # This is necessary due to the Gaussian cutoff term in the RDF.
        self.accuracy_factor = 1e-3

    def precompute(self):
        """Function to perform any kind of computations that should be
        performed only once.
        Determine the maximum distance at which atoms contribute
        to the PRDF. This is the maximum distance (r) at which exp(-B * (r -
        R)^2) > AccuracyFactor.

        """
        max_pair_distance = math.sqrt(-1 * math.log(self.accuracy_factor) /
                                      self.B) + self.cut_off_distance

        self.distance_computer = PairDistanceAnalysis()
        self.distance_computer.set_cutoff_distance(max_pair_distance)
        self.distance_computer.analyze_structure(self.structure)

    def analyze_structure(self, s):
        """Function to analyze a specific structure.
        Once this completes, it is possible to retrieve results out of this
        object.

        Parameters
        ----------
        s : Cell
            Structure to be analyzed.

        """
        self.structure = s
        self.precompute()

    def recompute(self):
        """Function to recompute structural information.
        """
        self.precompute()

    def set_smoothing_factor(self, b):
        """Function to set smoothing factor used when computing PRDF.

        Parameters
        ----------
        b : float
            Smoothing factor.

        Raises
        ------
        ValueError
            If B is negative.

        """
        if b <= 0:
            raise ValueError("B must be positive!! Supplied: "+str(b))
        self.B = b

    def set_cut_off_distance(self, d):
        """Function to set cut off distance used when computing PRDF.

        Parameters
        ----------
        d : float
            Cut off distance.

        Raises
        ------
        ValueError
            If d is negative.

        Exception
            If something goes wrong in computing cell.

        """
        if d <= 0:
            raise ValueError("Cut off distance must be positive!! Supplied: "
                             ""+str(d))

        # Set the cutoff distance.
        self.cut_off_distance = d

        # Redo precomputation, if needed.
        if self.structure is not None:
            try:
                self.precompute()
            except Exception:
                raise Exception("Oops, something went wrong. Check "
                                "stacktrace.")

    def set_n_windows(self, n_w):
        """Function to set the number of points at which to evaluate AP-RDF.

        Parameters
        ----------
        n_w : int
            Desired number of windows.

        """
        self.n_windows = n_w

    def compute_APRDF(self, properties):
        """Function to compute the AP-RDF of this structure.

        Parameters
        ----------
        properties : array-like
            Properties of each atom type.

        Returns
        -------
        output : array-like
            AP-RDF at R at n_windows steps between cut off distance /
            n_windows to cut off distance, inclusive.

        """

        # Make sure the number of properties is correct.
        if len(properties) != self.structure.n_types():
            raise ValueError("Incorrect number of properties. Supplied: {}. "
                             "Required: {}".format(len(properties),
                                                   self.structure.n_types()))

        # Get the evaluation distances.
        eval_R = self.get_evaluation_distances()

        # Initialize output.
        ap_rdf = np.zeros(len(eval_R), dtype=float)

        n_atoms = self.structure.n_atoms()
        # Loop through each pair of atoms.
        for i in range(n_atoms):
            for j in range(i, n_atoms):
                # All images of j within cutoff radius of i.
                images = self.distance_computer.find_all_images(i, j)

                # If i != j, the contributions get added twice.
                times_added = 1 if i == j else 2

                # Get types of i and j.
                i_type = self.structure.get_atom(i).get_type()
                j_type = self.structure.get_atom(j).get_type()

                # Add contributes from each pair.
                for image in images:
                    # Evaluate contribution at each distance.
                    for r in range(self.n_windows):
                        ap_rdf[r] += times_added * properties[i_type] * \
                                     properties[j_type] * math.exp(-1 *
                                    self.B * (image[1] - eval_R[r]) ** 2)

        # Scale output by number of atoms.
        ap_rdf *= 1.0 / n_atoms
        return ap_rdf

    def get_evaluation_distances(self):
        """Function to get the distances at which the PRDF should be analyzed.

        Returns
        -------
        output : list
            List of distances.

        """
        step_size = float(self.cut_off_distance) / self.n_windows
        return [(i + 1) * step_size for i in range(self.n_windows)]

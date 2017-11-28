import math
from vassal.analysis.PairDistanceAnalysis import PairDistanceAnalysis
import numpy as np

class APRDFAnalysis:
    """
    Class to compute the Atomic Property Weighted Radial Distribution
    Function (AP-RDF). Follows the work by Fernandez et al.
    http://pubs.acs.org/doi/abs/10.1021/jp404287t.
    Here, we use a scaling factor equal to: 1 / #atoms.
    """
    def __init__(self):
        """
        Function to create instance and initialize fields.
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
        """
        Function to perform any kind of computations that should be performed
        only once. Determine the maximum distance at which atoms contribute
        to the PRDF. This is the maximum distance (r) at which exp(-B * (r -
        R)^2) > AccuracyFactor.
        :return:
        """
        max_pair_distance = math.sqrt(-1 * math.log(self.accuracy_factor) /
                                      self.B) + self.cut_off_distance

        self.distance_computer = PairDistanceAnalysis()
        self.distance_computer.set_cutoff_distance(max_pair_distance)
        self.distance_computer.analyze_structure(self.structure)

    def analyze_structure(self, s):
        """
        Function to analyze a specific structure. Once this completes,
        it is possible to retrieve results out of this object.

        :param s: Structure to be analyzed.
        :return:
        """
        self.structure = s
        self.precompute()

    def recompute(self):
        """
        Function to recompute structural information.
        :return:
        """
        self.precompute()

    def set_smoothing_factor(self, b):
        """
        Function to set smoothing factor used when computing PRDF.
        :param b: Smoothing factor.
        :return:
        """
        if b <= 0:
            raise ValueError("B must be positive!! Supplied: "+str(b))
        self.B = b

    def set_cut_off_distance(self, d):
        """
        Function to set cut off distance used when computing PRDF.
        :param d: Cut off distance.
        :return:
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
        """
        Function to set the number of points at which to evaluate AP-RDF.
        :param n_w: Desired number of windows.
        :return:
        """
        self.n_windows = n_w

    def compute_APRDF(self, properties):
        """
        Function to compute the AP-RDF of this structure.
        :param properties: Properties of each atom type.
        :return: AP-RDF at R at n_windows steps between cut off distance /
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
        """
        Function to get the distances at which the PRDF should be analyzed.
        :return: List of distances.
        """
        step_size = float(self.cut_off_distance) / self.n_windows
        return [(i + 1) * step_size for i in range(self.n_windows)]
from vassal.analysis.PairDistanceAnalysis import PairDistanceAnalysis
from vassal.analysis.voronoi.VoronoiCell import VoronoiCell

class VoronoiTessellationCalculator:
    """
    Class that computes the Voronoi tessellation of a cell. Can either use a
    standard Voronoi tessellation or the radical plane method. The radical
    plane method takes the radii of atoms into account when partitioning the
    cell.
    Citation for Radical Voronoi method:
    http://www.sciencedirect.com/science/article/pii/002230938290093X
    Gellatly and Finney. JNCS (1970).
    """
    @classmethod
    def compute(self, cell, radical):
        """
        Function to run a a tessellation using the Python implementation of the
        voronoi tessellation provided with Vassal.
        Citation for the computation method:
        http://linkinghub.elsevier.com/retrieve/pii/0021999178901109
        Brostow, Dessault, Fox. JCP (1978).
        :param cell: Structure to analyze.
        :param radical: Whether to perform a radical plane tessellation.
        :return: Voronoi cell for each atom.
        """

        # Initialize Voronoi cells.
        output = [VoronoiCell(atom, radical) for atom in cell.get_atoms()]

        # Create tool to find closest images.
        image_finder = PairDistanceAnalysis()
        atom_length_scale = (cell.volume()/ cell.n_atoms()) ** (1/3.0)

        image_finder.analyze_structure(cell)

        # Generate cells.
        for c in output:
            c.compute_cell(image_finder, cutoff=atom_length_scale * 6)

        # Read the.
        return output
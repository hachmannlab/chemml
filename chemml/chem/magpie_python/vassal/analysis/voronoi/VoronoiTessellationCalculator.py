# -*- coding: utf-8 -*-
from ..PairDistanceAnalysis import PairDistanceAnalysis
from ..voronoi.VoronoiCell import VoronoiCell

class VoronoiTessellationCalculator:
    """Class that computes the Voronoi tessellation of a cell.
    Can either use a standard Voronoi tessellation or the radical plane
    method. The radical plane method takes the radii of atoms into account
    when partitioning the cell. Citation for Radical Voronoi method [1].

    References
    ----------
    .. [1] B. J. Gellatly and J. L. Finney, "Characterisation of models of
    multicomponent amorphous metals: The radical alternative to the Voronoi
    polyhedron," Journal of Non-Crystalline Solids, vol. 50, no. 3,
    pp. 313--329, Aug. 1982.

    """
    @classmethod
    def compute(self, cell, radical):
        """Function to run a a tessellation using the Python implementation
        of the voronoi tessellation provided with Vassal.
        Citation for the computation method [1].

        Parameters
        ----------
        cell : Cell
            Structure to analyze.
        radical : bool
            Whether to perform a radical plane tessellation.

        Returns
        -------
        output : array-like
            Voronoi cell for each atom.

        References
        ----------
        .. [1] W. Brostow, J.-P. Dussault, and B. L. Fox, "Construction of
        Voronoi polyhedra," Journal of Computational Physics, vol. 29, no. 1,
        pp. 81--92, Oct. 1978.
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

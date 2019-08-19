from numpy.linalg import norm
import numpy as np

class VoronoiVertex:
    """Class for a vertex in a Voronoi tessellation.

    Attributes
    ----------
    position : array-like
        Position of vertex.
    distance : float
        Distance from cell center.
    next_edge : VoronoiEdge
        Edge that is after this vertex.
    previous_edge : VoronoiEdge
        Edge that is before this vertex.
    """
    def __init__(self, inside_atom=None, position=None, edge1=None, edge2=None):
        """Function to initialize a Voronoi vertex. Can specify either inside
        atom and position or two edges.

        Parameters
        ----------
        inside_atom : Atom
            Atom on "inside" of this face.
        position : array-like
            Position of vertex.
        edge1 : VoronoiEdge
            Edge 1.
        edge2 : VoronoiEdge
            Edge 2.

        Raises
        ------
        ValueError
            Position should be a numpy array!.
        """

        if position is not None and not isinstance(position, np.ndarray):
            raise ValueError("Position should be a numpy array!.")

        in_atom = inside_atom
        pos = position

        # Edge that is before this vertex.
        self.previous_edge = edge1

        # Edge that is after this vertex.
        self.next_edge = edge2

        if in_atom is None and pos is None:
            in_atom = edge1.get_edge_face().get_inside_atom()
            pos = edge1.get_line().intersection(edge2.get_line())

        # Position of vertex.
        self.position = pos

        # Distance from cell center.
        self.distance = norm(self.position - in_atom.get_position_cartesian())

        if inside_atom is None and position is None:
            # Store next and previous edges.
            if edge1.is_ccw(edge2=edge2):
                self.previous_edge = edge1
                self.next_edge = edge2
            else:
                self.previous_edge = edge2
                self.next_edge = edge1

    @classmethod
    def get_centroid(self, points):
        """Function to compute the centroid of a group of vertices.

        Parameters
        ----------
        points : array-like
            Points to be considered.

        Returns
        -------
        center : array-like
            Centroid.

        """
        center = np.zeros(3)
        for p in points:
            center += p.get_position()

        center /= len(points)
        return center

    def distance_from(self, vertex):
        """Function to compute distance between vertices.

        Parameters
        ----------
        vertex : VoronoiVertex
            Other vertex.

        Returns
        -------
        output : float
            Distance between them.

        """
        return norm(self.position - vertex.position)


    def get_distance_from_center(self):
        """Function to get the distance from the center of the cell.

        Returns
        -------
        output : float
            Distance.
        """
        return self.distance

    def get_position(self):
        """Function to get the position of this vertex.

        Parameters
        ----------

        Returns
        -------
        output : array-like
            Position of this vertex.
        """
        return self.position

    def __str__(self):
        """Function to print the position of the vertex.

        Returns
        -------
        output : str
            String containing the position of the vertex.
        """
        return str(self.position)

    def __hash__(self):
        """Function to compute the hash value of the vertex.

        Returns
        -------
        value : int
            Hash value.
        """
        return 3 + hash(self.position)

    def __eq__(self, other):
        """Function to check the instance is equal to another vertex.

        Parameters
        ----------
        other : VoronoiVertex
            Other vertex to check.

        Returns
        -------
        value : bool
            True if equal, else False.
        """
        if isinstance(other, VoronoiVertex):
            return other.previous_edge.__eq__(self.previous_edge) and \
                   other.next_edge.__eq__(self.next_edge)
        return False

    def __cmp__(self, other):
        """Function to compare the instance with another vertex.

        Parameters
        ----------
        other : VoronoiVertex
            Other vertex to compare.

        Returns
        -------
        value : int
            -1 if instance < other, +1 if instance > other, else 0.
        """
        if self.previous_edge.__eq__(other.previous_edge):
            return self.next_edge.__cmp__(other.next_edge)
        else:
            return self.previous_edge.__cmp__(other.previous_edge)

    def get_next_edge(self):
        """Function to get the edge after this vertex.

        Returns
        -------
        next_edge : VoronoiEdge
            Edge that is after this vertex.
        """
        return self.next_edge

    def get_previous_edge(self):
        """Function to get the edge before this vertex.

        Returns
        -------
        previous_edge : VoronoiEdge
            Edge that is before this vertex.

        """
        return self.previous_edge

    def is_on_edge(self, e):
        """Function to check if a vertex is on a certain edge.

        Parameters
        ----------
        e : VoronoiEdge
            Edge to check.

        Returns
        -------
        output : bool
            True if instance is on edge, else False.

        """
        return self.previous_edge.__eq__(e) or self.next_edge.__eq(e)

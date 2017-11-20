import numpy as np
import sys
from numpy.linalg import norm
from vassal.analysis.voronoi.VoronoiVertex import VoronoiVertex
from vassal.data.Cell import Cell

class VoronoiEdge:
    """
    Class describing an edge of a cell in a Voronoi tessellation.
    """
    def __init__(self, edge_face=None, intersecting_face=None):
        """
        Function to initialize an edge instance.
        :param edge_face: Plane defining one face containing this edge.
        :param intersecting_face: Plane defining the other face defining this
        edge.
        """

        # EdgeFace on which this edge exists.
        self.edge_face = edge_face

        # EdgeFace that intersected edge_face to form this edge.
        self.intersecting_face = intersecting_face

        # Line representing this edge.
        self.line = None

        # Direction of edge.
        self.direction = None

        # Point marking the beginning of this edge.
        self.beginning = -float("inf")

        # Edge after this edge.
        self.next_edge = None

        # Point marking the end of this edge.
        self.end = float("inf")

        # Edge before this edge.
        self.previous_edge = None

        # Compute the line.
        self.line = edge_face.get_plane().intersection(other=
            intersecting_face.get_plane())
        if self.line is None:
            raise Exception("Planes are parallel.")

        # Ensure vector is CCW w.r.t edge face.
        cut_direction = -intersecting_face.get_normal()
        self.direction = self.line.get_direction()
        if not self.is_ccw(vec1=edge_face.get_normal(),
                           vec2=self.direction, vec3=cut_direction):
            self.line = self.line.revert()

    def is_ccw(self, vec1=None, vec2=None, vec3=None, edge2=None):
        """
        Function to compute whether, given a face normal, edges with two
        directions are CCW.
        :param vec1: Face normal.
        :param vec2: Direction of edge a.
        :param vec3: Direction of edge b.
        :param edge2: Edge 2.
        :return: Whether they are CCW.
        """
        v0 = vec1
        v1 = vec2
        v2 = vec3
        if edge2 is not None:
            v0 = self.edge_face.get_normal()
            v1 = self.direction
            v2 = edge2.direction

        term1 = v0[0] * v1[1] * v2[2] - v0[0] * v1[2] * v2[1]
        term2 = v0[1] * v1[2] * v2[0] - v0[1] * v1[0] * v2[2]
        term3 = v0[2] * v1[0] * v2[1] - v0[2] * v1[1] * v2[0]

        ccw = term1 + term2 + term3

        # Check if the magnitude of ccw is less than machine epsilon. If so,
        # they are considered to be collinear. Return False.
        if abs(ccw) <= np.finfo(float).eps:
            return False

        return ccw > 0

    @classmethod
    def compute_intersection(self, edge1, edge2, just_join=False):
        """
        Function to compute intersection between two edges or join them.

        Computes the intersection between two vectors. If they intersect,
        within the bounds of each edge, set new boundaries for each edge and
        change the next and previous edge as appropriate.

        Will check orientation of edges to see if edge1 is the edge following
        edge2 (or vis versa), and join them accordingly.
        :param edge1: First edge.
        :param edge2: Second edge.
        :param just_join: Whether to just join them or compute intersection.
        :return: Whether two edges intersect or nothing depending on the input.
        """

        # Determine the point at which the edges intersect.
        point = edge1.line.intersection(edge2.line)
        if point is None:
            if just_join:
                raise Exception("Edges do not intersect.")
            else:
                return False

        # Determine the relationship between edges (using their directions).
        is_forward = edge1.is_ccw(edge2=edge2)

        # Using the direction, check whether intersection is within bounds of
        #  each edge.
        edge1_terminus = edge1.line.get_abscissa(point)
        edge2_terminus = edge2.line.get_abscissa(point)

        if not just_join:
            within_bounds = None
            if is_forward:
                within_bounds = edge1_terminus < edge1.end and edge2_terminus > \
                                                               edge2.beginning
            else:
                within_bounds = edge1_terminus > edge1.beginning and \
                                edge2_terminus < edge2.end

            if not within_bounds:
                return False

        # Now update the edges accordingly.
        if is_forward:
            edge1.end = edge1_terminus
            edge1.next_edge = edge2
            edge2.beginning = edge2_terminus
            edge2.previous_edge = edge1
        else:
            edge1.beginning = edge1_terminus
            edge1.previous_edge = edge2
            edge2.end = edge2_terminus
            edge2.next_edge = edge1

        if not just_join:
            return True

    def __eq__(self, other):
        """
        Function to check if instance is equal to another edge.
        :param other: Other edge.
        :return: True if instance equal to other, else False.
        """
        if isinstance(other, VoronoiEdge):
            return other.edge_face.__eq__(self.edge_face) and \
                   other.intersecting_face.__eq__(self.intersecting_face)
        return False

    def __hash__(self):
        """
        Function to compute the hash value of instance.
        :return: Hash value.
        """
        h = 5
        h = 43 * h + id(self.edge_face)
        h = 43 * h + id(self.intersecting_face)
        return h

    def __cmp__(self, other):
        """
        Function to compare instance and another edge.
        :param other: Other edge.
        :return: -1 if instance < other, +1 if instance > other, else 0.
        """
        if self.edge_face.__eq__(other.edge_face):
            return self.intersecting_face.__cmp__(other.intersecting_face)
        return self.edge_face.__cmp__(other.edge_face)

    def get_line(self):
        """
        Function to get the line defining this edge.
        :return: Line.
        """
        return self.line

    def get_edge_face(self):
        """
        Function to get the face containing this edge.
        :return: Face containing this edge.
        """
        return self.edge_face

    def get_intersecting_face(self):
        """
        Function to get the other face associated with this edge.
        :return: Other face associated with this edge.
        """
        return self.intersecting_face

    def get_next_edge(self):
        """
        Function to get the next edge on this face.
        :return: Next edge (None if no such edge).
        """
        return self.next_edge

    def get_previous_edge(self):
        """
        Function to get the previous edge on this face.
        :return: Previous edge (None if no such edge).
        """
        return self.previous_edge

    def get_length(self):
        """
        Function to get the length of this edge.
        :return: Length.
        """
        return norm((self.beginning - self.end) * self.direction)

    def __str__(self):
        """
        Function to get the string representation of instance.
        :return: String representation of instance.
        """
        output = "("+self.edge_face.__str__()+"," \
                        ""+self.intersecting_face.__str__()+")"
        return output

    def find_next_edge(self, candidates):
        """
        Function to find the edge that is likely to be "next" on a face that
        contains this edge. This is computed by first computing all edges
        that are oriented CCW to this edge. Next, the edge that is closest to
        the origin is found. If multiple edges intersect at the same point,
        the one with the greatest angle between the direction of this edge is
        selected.
        :param candidates: All candidate edges.
        :return: The next edge. None if no suitable candidate is found.
        """

        # Locate the ccw edges.
        ccw_edges = []
        for edge in candidates:
            # print self.intersecting_face.outside_atom.__str__(), \
            #     edge.intersecting_face.outside_atom.__str__()
            if not self.__eq__(edge):
                flag = self.is_ccw(edge2=edge)
                # print edge.intersecting_face.outside_atom, flag
                if flag:
                    ccw_edges.append(edge)

        # Check if any were found.
        if len(ccw_edges) == 0:
            return None
        elif len(ccw_edges) == 1:
            return ccw_edges[0]

        # Find the closest edge(s).
        closest_edges = []
        min_dist = float("inf")
        min_point = np.array([np.inf]*3)
        for edge in ccw_edges:
            other_line = edge.get_line()
            # If this line contains the minimum point, add it to list.
            if other_line.contains(min_point):
                closest_edges.append(edge)
                continue

            intersection = self.line.intersection(other_line)
            if intersection is None:
                # Line is anti parallel.
                continue

            # See if it is the closest.
            x = self.line.get_abscissa(intersection)
            if x < min_dist:
                closest_edges = []
                min_dist = x
                min_point = intersection
                closest_edges.append(edge)

        # If only one edge, return answer.
        if len(closest_edges) == 1:
            return closest_edges[0]

        # Otherwise, find the edge with the largest angle.
        max_angle = 0
        choice = None
        for edge in closest_edges:
            angle = self.angle_between(self.line.get_direction(),
                                       edge.get_line().get_direction())
            if angle > max_angle:
                choice = edge
                max_angle = angle

        return choice

    def get_start_vertex(self):
        """
        Function to get the vertex at the beginning of this vector.
        :return: Starting vertex.
        """
        return VoronoiVertex(edge1=self, edge2=self.previous_edge)

    def get_end_vertex(self):
        """
        Function to get the vertex at the end of this vector.
        :return: End vertex.
        """
        return VoronoiVertex(edge1=self, edge2=self.next_edge)

    def generate_pair(self):
        """
        Function to generate the edge that corresponds to this edge on the
        intersecting face.
        :return: Newly instantiated edge.
        """
        try:
            return VoronoiEdge(self.get_intersecting_face(),
                               self.get_edge_face())
        except Exception:
            raise Exception("Shouldn't be possible.")

    def angle_between(self, v1, v2):
        """
        Documentation obtained from:
        http://commons.apache.org/proper/commons-math/javadocs/api-3.3/org
        /apache/commons/math3/geometry/euclidean/threed/Vector3D.html
        Function to compute the angular separation between two vectors. This
        method computes the angular separation between two vectors using the
        dot product for well separated vectors and the cross product for
        almost aligned vectors. This allows to have a good accuracy in all
        cases, even for vectors very close to each other.
        :param v1: First vector.
        :param v2: Second vector.
        :return: Angular separation between v1 and v2.
        """
        norm_product = norm(v1) * norm(v2)

        if norm_product == 0:
            raise Exception("Norms are zero!")
        dp = np.dot(v1, v2)
        threshold = norm_product * 0.9999
        if dp < -threshold or dp > threshold:
            # The vectors are almost aligned, compute using the sine.
            v3 = np.cross(v1, v2)
            x = np.math.asin(norm(v3) / norm_product)
            if dp >= 0:
                return x
            return np.math.pi - x
        else:
            # The vectors are sufficiently separated to use the cosine.
            return np.math.acos(dp / norm_product)

    def print_properties(self):
        """
        Function to print different properties of the Voronoi Edge instance.
        Mainly used for debugging purposes.
        :return:
        """
        print "Edge face:", self.edge_face.outside_atom.__str__(), \
            "Intersecting face:", self.intersecting_face.outside_atom.__str__()
        print "Line zero:", self.line.zero
        print "Line direction:", self.line.direction
        print "Line tolerance:", self.line.tolerance
        print "Edge direction:", self.direction
        print "Beginning:", self.beginning
        print "End:", self.end
        print
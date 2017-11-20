import numpy as np
from numpy.linalg import norm
from vassal.geometry.Line import Line

class Plane:
    """
    Documentation obtained from: http://commons.apache.org/proper/commons-math
    /apidocs/org/apache/commons/math4/geometry/euclidean/threed/Plane.html
    Class to represent planes in a three dimensional space.
    """

    def __init__(self, normal=None, tolerance=None, p=None, plane=None,
                 p1=None, p2=None, p3=None):
        """
        Function to build a plane normal to a given direction and containing
        the origin. If p is specified, the plane contains the point. If plane
        is specified, makes a copy of the plane.
        :param normal: Normal direction to the plane.
        :param tolerance: Tolerance below which points are considered identical.
        :param p: Point belonging to the plane.
        :param plane: Plane to copy.
        """

        if plane is None and normal is not None and tolerance is not None:
            n = norm(normal)
            if n < 1e-10:
                raise Exception("Norm is zero!")

            # Third vector of the plane frame (plane normal).
            self.w = normal / n

            # Tolerance below which points are considered identical.
            self.tolerance = tolerance

            #  Offset of the origin with respect to the plane.
            self.origin_offset = -np.dot(p, self.w) if p is not None else 0

            # Origin of the plane frame.
            self.origin = -self.origin_offset * self.w

            # First vector of the plane frame (in plane).
            self.u = self.orthogonal(self.w)

            # Second vector of the plane frame (in plane).
            self.v = np.cross(self.w, self.u)
        elif plane is not None:
            #  Offset of the origin with respect to the plane.
            self.origin_offset = plane.origin_offset

            # Origin of the plane frame.
            self.origin = plane.origin

            # First vector of the plane frame (in plane).
            self.u = plane.u

            # Second vector of the plane frame (in plane).
            self.v = plane.v

            # Third vector of the plane frame (plane normal).
            self.w = plane.w

            # Tolerance below which points are considered identical.
            self.tolerance = plane.tolerance
        elif p1 is not None and p2 is not None and p3 is not None and \
                tolerance is not None:
            v1 = np.array(p1, dtype=float)
            v2 = np.array(p2, dtype=float)
            v3 = np.array(p3, dtype=float)
            n_vec = np.cross(v2 - v1, v3 - v1)
            self.__init__(p=v1, normal=n_vec, tolerance=tolerance)

    def orthogonal(self, w):
        """
        Documentation obtained from:
        http://commons.apache.org/proper/commons-math/javadocs/api-3.3/org
        /apache/commons/math3/geometry/euclidean/threed/Vector3D.html
        Function to compute a vector orthogonal to a given vector.
        There are an infinite number of normalized vectors orthogonal to the
        instance. This method picks up one of them almost arbitrarily. It is
        useful when one needs to compute a reference frame with one of the
        axes in a predefined direction. The following example shows how to
        build a frame having the k axis aligned with the known vector u :
        Vector3D k = u.normalize();
        Vector3D i = k.orthogonal();
        Vector3D j = Vector3D.crossProduct(k, i);
        :param w: Given vector.
        :return: Normalized orthogonal vector.
        """

        threshold = 0.6 * norm(w)
        if threshold == 0:
            raise Exception("Norm is zero!")
        x = w[0]
        y = w[1]
        z = w[2]
        if abs(x) <= threshold:
            inverse = 1 / np.math.sqrt(y ** 2 + z ** 2)
            return np.array([0, inverse * z, -inverse * y])
        elif abs(y) <= threshold:
            inverse = 1 / np.math.sqrt(x ** 2 + z ** 2)
            return np.array([-inverse * z, 0, inverse * x])
        inverse = 1 / np.math.sqrt(x ** 2 + y ** 2)
        return np.array([inverse * y, -inverse * x, 0])

    def get_normal(self):
        """
        Function to get the direction normal to the plane.
        :return: Direction normal to the plane.
        """
        return self.w

    def get_origin(self):
        """
        Function to get the origin of the plane frame.
        :return: Origin of the plane frame.
        """
        return self.origin

    def project(self, p):
        """
        Function to transform a point into a projection.
        :param p: Desired point to transform.
        :return: Transformation.
        """
        p2 = np.array([np.dot(p, self.u), np.dot(p, self.v)])
        return self.u * p2[0] + self.v * p2[1] - self.origin_offset * self.w

    def get_point_at(self, in_plane, offset):
        """
        Function to get one point from 3D-space.
        :param in_plane: Desired in-plane coordinates for the point in the
        plane.
        :param offset: Desired offset for the point.
        :return: One point in the 3D-space, with given coordinates and offset.
        """
        return self.u * in_plane[0] + self.v * in_plane[1] - \
               self.origin_offset * self.w

    def intersection(self, l=None, other=None):
        """
        Function to compute the intersection with another line or plane.
        :param l: Line intersecting the instance.
        :param other: Other plane.
        :return: Point or line of intersection depending on input.
        """
        if l is not None:
            dir = l.get_direction()
            dot = np.dot(self.w, dir)
            if dot < 1e-10:
                return None
            p = l.point_at(0.0)
            k = - (self.origin_offset + np.dot(self.w, p)) / dot
            return p + k * dir
        else:
            dir = np.cross(self.w, other.w)
            if norm(dir) < self.tolerance:
                return None
            p = self.intersection_3_planes(self, other, Plane(normal=dir,
                tolerance=self.tolerance))
            return Line(p1=p, p2=p + dir, tolerance=self.tolerance)

    @classmethod
    def intersection_3_planes(self, p1, p2, p3):
        """
        Function to the compute the intersection of three planes, None if
        some planes are parallel.
        :param p1: First plane.
        :param p2: Second plane.
        :param p3: Third plane.
        :return: Point of intersection of three planes.
        """

        # Coefficients of the three planes linear equations.
        a1 = p1.w[0]
        b1 = p1.w[1]
        c1 = p1.w[2]
        d1 = p1.origin_offset

        a2 = p2.w[0]
        b2 = p2.w[1]
        c2 = p2.w[2]
        d2 = p2.origin_offset

        a3 = p3.w[0]
        b3 = p3.w[1]
        c3 = p3.w[2]
        d3 = p3.origin_offset

        # Direct Cramer resolution of the linear system.
        # This is still feasible for a 3x3 system.
        a23 = b2 * c3 - b3 * c2
        b23 = c2 * a3 - c3 * a2
        c23 = a2 * b3 - a3 * b2

        determinant = a1 * a23 + b1 * b23 + c1 * c23
        if abs(determinant) < 1e-10:
            return None

        r = 1.0 / determinant

        return np.array([
        (-a23 * d1 - (c1 * b3 - c3 * b1) * d2 - (c2 * b1 - c1 * b2) * d3) * r,
        (-b23 * d1 - (c3 * a1 - c1 * a3) * d2 - (c1 * a2 - c2 * a1) * d3) * r,
        (-c23 * d1 - (b1 * a3 - b3 * a1) * d2 - (b2 * a1 - b1 * a2) * d3) * r
        ])

    def contains(self, p):
        """
        Function to check if the instance contains a point.
        :param p: Point to check.
        :return: True if p belongs to plane, else False.
        """
        return abs(self.get_offset(point=p)) < self.tolerance

    def get_offset(self, point=None, plane=None):
        """
        Function to get the offset (oriented distance) of a parallel plane or a
        point. This method should be called only for parallel planes
        otherwise the result is not meaningful. The offset is 0 if both
        planes are the same, it is positive if the plane is on the plus side
        of the instance and negative if it is on the minus side, according to
        its natural orientation.
        :param point: Point to check.
        :param plane: Plane to check.
        :return: Offset of the plane or the point depending on input.
        """
        if plane is None:
            return np.dot(point, self.w) + self.origin_offset
        else:
            return self.origin_offset + (-plane.origin_offset if
                    self.same_orientation_as(plane) else plane.origin_offset)

    def same_orientation_as(self, other):
        """
        Function to check if the instance has the same orientation as another
        plane.
        :param other: Other plane to check against the instance.
        :return: True if the instance and the other plane have the same
        orientation, else False.
        """
        return np.dot(self.w, other.w) > 0.0
from __future__ import print_function
import numpy as np
from numpy.linalg import norm
import sys
import warnings

class Line:
    """Class to represent lines in a three dimensional space.
    Documentation obtained from http://commons.apache.org/proper/commons-math
    /javadocs/api-3.3/org/apache/commons/math3/geometry/euclidean/threed/Line
    .html
    Each oriented line is intrinsically associated with an abscissa which is
    a coordinate on the line. The point at abscissa 0 is the orthogonal
    projection of the origin on the line, another equivalent way to express
    this is to say that it is the point of the line which is closest to the
    origin. Abscissa increases in the line direction.

    Attributes
    ----------
    direction : array-like
        Line direction.
    zero : array-like
        Line point closest to the origin.
    tolerance : float
        Tolerance below which points are considered identical.

    """

    def __init__(self, p1=None, p2=None, tolerance=None, l=None):
        """Constructor to create instance of the Line object.

        Parameters
        ----------
        p1 : array-like
            First point belonging to the Line (this can be any point).
        p2 : array-like
            Second point belonging to the Line (this can be any point,
            different from p1).
        tolerance : float
            Tolerance below which points are considered identical.
        l : Line
            Line to copy.
        """

        if l is None:
            p1_arr = np.array(p1, dtype=float)
            p2_arr = np.array(p2, dtype=float)

            delta = p2_arr - p1_arr
            norm2 = delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2
            norm1 = np.math.sqrt(norm2)
            if norm2 == 0:
                raise Exception("Norm is zero!")

            # Line direction.
            self.direction = delta / norm1

            # Line point closest to the origin.
            self.zero = p1_arr - np.dot(p1_arr, delta) / norm2

            # Tolerance below which points are considered identical.
            self.tolerance = 1e-10 if tolerance is None else tolerance
        else:
            # Tolerance below which points are considered identical.
            self.tolerance = l.tolerance

            # Line direction.
            self.direction = l.direction

            # Line point closest to the origin.
            self.zero = l.zero

    def set_tolerance(self, tol):
        """Function to set the tolerance.

        Parameters
        ----------
        tol : float
            Desired tolerance.

        """
        self.tolerance = tol

    def get_tolerance(self):
        """Function to get the tolerance.

        Returns
        -------
        output : float
            Tolerance.

        """
        return self.tolerance

    def revert(self):
        """Function to revert the direction of the current line.

        Returns
        -------
        output : Line
            A new Line instance with the direction reversed.
        """
        reverted = Line(l=self)
        reverted.direction *= -1.0
        return reverted

    def get_direction(self):
        """Function to get the direction of the line.

        Returns
        -------
        output : array-like
            Direction as a numpy array.
        """
        return self.direction

    def get_origin(self):
        """Get the line point closest to the origin.

        Returns
        -------
        output : array-like
            Point as a numpy array.
        """
        return self.zero

    def get_abscissa(self, p):
        """Function to get the abscissa of a point with respect to a line.

        The abscissa is 0 if the projection of the point and the projection of
        the frame origin on the line are the same point.

        Parameters
        ----------
        p : array-like
            Desired point.

        Returns
        -------
        output : float
            Abscissa of the point.

        """
        return np.dot(p - self.zero, self.direction)

    def point_at(self, abscissa):
        """Function to get one point from the line.

        Parameters
        ----------
        abscissa : float
            Desired abscissa for the point/

        Returns
        -------
        output : array-like
            One point belonging to the line, at specified abscissa.

        """
        return self.zero + abscissa * self.direction

    def contains(self, p):
        """Function to check if the instance contains a point.

        Parameters
        ----------
        p : array-like
            Point to check.

        Returns
        -------
        output : bool
            True if p belongs to the line, else False.

        """
        return self.distance(p=p) < self.tolerance

    def distance(self, p=None, l=None):
        """Function to compute the distance between the instance and a point or
        the shortest distance between the instance and another line.

        Parameters
        ----------
        p : array-like
            Point to compute distance between. (Default value = None)
        l : Line
            Line to compute distance between. (Default value = None)

        Returns
        -------
        output : float
            Distance.

        """
        if l is None:
            d = p - self.zero
            n = np.zeros(3)
            # try:
            #     n = d - np.dot(d, self.direction) * self.direction
            # except RuntimeWarning:
            #     print(d, self.direction)
            # return norm(n)
            with warnings.catch_warnings(record=True) as w:
                # Cause all warnings to always be triggered.
                warnings.simplefilter("always")
                n = d - np.dot(d, self.direction) * self.direction
                # print(n, norm(n))
                if len(w) > 0 and issubclass(w[-1].category, RuntimeWarning):
                    # Todo: check w/ Ram if this is what he meant to do when catch a warning: n = np.zeros(3)
                    # n = np.zeros(3)
                    # print(d, self.direction)
                    pass
                return norm(n)
        else:
            normal = np.cross(self.direction, l.direction)
            n = norm(normal)
            if n < sys.float_info.min:
                # Lines are parallel.
                return self.distance(p=l.zero)
            offset = np.dot(l.zero - self.zero, normal) / n
            return np.abs(offset)

    def distance_sq(self, p):
        """Function to compute the distance squared between the instance and a
        point.

        Parameters
        ----------
        p : array-like
            Point to compute distance between.

        Returns
        -------
        output : float
            Distance.

        """
        d = p - self.zero
        n = d - np.dot(d, self.direction) * self.direction
        return n[0] ** 2 + n[1] ** 2 + n[2] ** 2

    def closest_point(self, l):
        """Function to the compute the point of the instance closest to another
        line.

        Parameters
        ----------
        l : Line
            Line to check against the instance.

        Returns
        -------
        output : array-like
            Point of the instance closest to another line.

        """
        cos = np.dot(self.direction, l.direction)
        n = 1 - cos ** 2
        if n < sys.float_info.epsilon:
            # Lines are parallel.
            return self.zero

        d0 = l.zero - self.zero
        a = np.dot(d0, self.direction)
        b = np.dot(d0, l.direction)
        return self.zero + self.direction * ( a - b * cos) / n

    def intersection(self, l):
        """Function to compute the intersection point of the instance and
        another line.

        Parameters
        ----------
        l : Line
            Other line.

        Returns
        -------
        output : array-like
            Intersection point of the instance and the other line,
            else None.

        """
        closest = self.closest_point(l)
        return closest if l.contains(closest) else None

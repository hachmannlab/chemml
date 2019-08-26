import numpy as np

class Atom:
    """Class that represents a single atom.

    Attributes
    ----------
    position : array-like
        Position in fractional coordinates.
    position_cartesian : array-like
        Position in cartesian coordinates.
    type : int
        Type of atom.
    cell : Cell
        Cell associated with this atom.
    id : int
        ID number of this atom.
    radius : float
        Radius of atom.
    """

    def __init__(self, position, type):
        """Constructor to create a new instance of the object.

        Attributes
        ----------
        position : array-like
            Position in fractional coordinates.
        type : int
            Type of atom.
        """

        # Position in fractional coordinates.
        self.position = None

        # Position in Cartesian coordinates.
        self.position_cartesian = None

        # Type of atom.
        self.type = None

        # Cell associated with this atom.
        self.cell = None

        # ID number of this atom.
        self.id = None

        # Radius of atom.
        self.radius = 1.0

        self.position = np.array(position, dtype=float)
        self.type = type

    def __copy__(self):
        """
        Function to override the copy() method.

        Returns
        -------
        output : Atom
            A new instance with the appropriate properties set.
        """
        try:
            x = type(self)(self.__class__)
        except TypeError:
            cls = self.__class__
            x = cls.__new__(cls)
        x.__dict__.update(self.__dict__)
        x.position = self.position.copy()
        x.position_cartesian = self.position_cartesian.copy()
        return x

    def __eq__(self, other):
        """Function to override the check for equality with another object of
        the same class.

        Parameters
        ----------
        other : Atom
            Other object.

        Returns
        -------
        output : bool
            True if equal, else False.
        """

        if isinstance(other, Atom):
            # Compare type.
            my_name = self.get_type_name()
            your_name = other.get_type_name()

            if not (my_name == "NA" and your_name == "NA"):
                # If type has name, compare names.
                if my_name != your_name:
                    return False
            else:
                # Otherwise, compare index.
                if self.type != other.type:
                    return False

            # Compare positions: Equal if they are closer than 0.001 Cartesian.
            my_pos = self.get_position_cartesian()
            other_pos = other.get_position_cartesian()

            dist = 0.0
            for i in range(3):
                dist += (my_pos[i] - other_pos[i])**2
                if dist > 1e-6:
                    return False

            return True
        return False

    def set_id(self, id):
        """Function to set the ID number of this atom.

        Parameters
        ----------
        id : int
            Desired id.

        """

        self.id = id

    def set_type(self, type):
        """Function to set the type of atom.

        Parameters
        ----------
        type : int
            Index of type.

        """

        self.type = type
        if self.cell is not None:
            while self.cell.n_types() <= self.type:
                self.cell.add_type(None)

    def set_cell(self, cell):
        """Function to define the cell in which this atom is situated.

        Parameters
        ----------
        cell : Cell
            Cell.

        """

        self.cell = cell
        self.update_cartesian_coordinates()

    def update_cartesian_coordinates(self):
        """Function to recompute the cartesian coordinates of this atom.

        """

        self.position_cartesian = self.cell.convert_fractional_to_cartesian(
            self.position)

    def get_cell(self):
        """Function to get the cell that contains this atom.

        Returns
        -------
        output : Cell
            Link to the cell.

        """

        return self.cell

    def get_position(self):
        """Function to get the fractional coordinates of this atom.

        Returns
        -------
        output : array-like
            Fractional coordinates.
        """

        return self.position

    def get_position_cartesian(self):
        """Function to get the cartesian coordinates of this atom.

        Returns
        -------
        output : array-like
            Cartesian coordinates.
        """

        return self.position_cartesian

    def set_radius(self, radius):
        """Function to set the radius of the atom.

        Parameters
        ----------
        radius : float
            Desired radius.

        """

        self.radius = radius

    def get_radius(self):
        """Function to get the radius of this atom.

        Returns
        -------
        output : float
            Radius.
        """

        return self.radius

    def get_id(self):
        """Function to get the ID number of this atom.

        Returns
        -------
        output : int
            ID number.
        """

        return self.id

    def get_type(self):
        """Function to get the type of this atom.

        Returns
        -------
        output : int
            Type of atom.
        """

        return self.type

    def get_type_name(self):
        """Function to get the name of this atom type.

        Returns
        -------
        output : str
            Name of the type of atom.
        """

        return self.cell.get_type_name(self.type)

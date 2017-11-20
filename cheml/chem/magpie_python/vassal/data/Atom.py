import numpy as np

class Atom:
    """
    Class that represents a single atom.
    """

    def __init__(self, position, type):
        """
        Constructor to create a new instance of the object.
        :param position: Position in fractional coordinates.
        :param type: Atom type.
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
        :return: A new instance with the appropriate properties set.
        """
        pos = self.position.copy()
        t = self.type
        x = Atom(pos, t)
        x.position_cartesian = self.position_cartesian.copy()
        x.id = self.id
        return x

    def __eq__(self, other):
        """
        Function to override the check for equality with another object of
        the same class.
        :param other: Other object.
        :return: True if equal, else False.
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
        """
        Function to set the ID number of this atom.
        :param id: Desired id.
        :return:
        """

        self.id = id

    def set_type(self, type):
        """
        Function to set the type of atom.
        :param type: Index of type.
        :return:
        """

        self.type = type
        if self.cell is not None:
            while self.cell.n_types() <= self.type:
                self.cell.add_type(None)

    def set_cell(self, cell):
        """
        Function to define the cell in which this atom is situated.
        :param cell: Cell.
        :return:
        """

        self.cell = cell
        self.update_cartesian_coordinates()

    def update_cartesian_coordinates(self):
        """
        Function to recompute the cartesian coordinates of this atom.
        :return:
        """

        self.position_cartesian = self.cell.convert_fractional_to_cartesian(
            self.position)

    def get_cell(self):
        """
        Function to get the cell that contains this atom.
        :return: Link to the cell.
        """

        return self.cell

    def get_position(self):
        """
        Function to get the fractional coordinates of this atom.
        :return: Fractional coordinates.
        """

        return self.position

    def get_position_cartesian(self):
        """
        Function to get the fractional coordinates of this atom.
        :return: Fractional coordinates.
        """

        return self.position_cartesian

    def set_radius(self, radius):
        """
        Function to set the radius of the atom.
        :param radius: Desired radius.
        :return:
        """

        self.radius = radius

    def get_radius(self):
        """
        Function to get the radius of this atom.
        :return: Radius.
        """

        return self.radius

    def get_id(self):
        """
        Function to get the ID number of this atom.
        :return: ID number.
        """

        return self.id

    def get_type(self):
        """
        Function to get the type of this atom.
        :return: Type of atom.
        """

        return self.type

    def get_type_name(self):
        """
        Function to get the name of this atom type.
        :return: Name of this atom type.
        """

        return self.cell.get_type_name(self.type)
#py2 and py3
from six import iteritems

from itertools import combinations as comb
import numpy as np
from ....utility.EqualSumCombinations import EqualSumCombinations
from ....data.materials.CompositionEntry import CompositionEntry
from ....data.materials.util.LookUpData import LookUpData

class PhaseDiagramCompositionEntryGenerator:
    """Class to generate composition entries at many points in many phase
    diagrams. Has two different ways of selecting compositions within phase
    diagrams.
    1. Even Spacing: Compositions are selected to be evenly-spaced throughout
    the phase diagram (e.g. A0.2B0.2C0.6, A0.4B0.2C0.4 etc.). This method is
    most appropriate for alloy systems.
    2. Simple Fractions: Compositions with the smallest denominator are
    selected (e.g. ABC, A2C, B2C, etc.). This method is most appropriate for
    phase diagrams that represent ordered crystalline compounds.

    Attributes
    ----------
    e_ids : array-like
        A list of int values denoting the list of elements to use.
    min_order : int
        Minimum number of constituents.
    max_order : int
        Maximum number of constituents.
    even_spacing : bool
        Whether to use even spacing or small integers.
    size : int
        Either number of stops in each direction or max denominator.

    """

    # List of elements to use (id is Z-1).
    e_ids = None

    # Minimum number of constituents.
    min_order = 1

    # Maximum number of constituents.
    max_order = 1

    # Whether to use even spacing or small integers.
    even_spacing = True

    # Either number of stops in each direction or max denominator.
    size = 3

    def set_elements_by_index(self, indices):
        """Function to define the list of elements to be included in the phase
        diagrams.

        Parameters
        ----------
        indices : array-like
            List of elements by index (Z-1).

        Raises
        ------
        ValueError
            If an element index is out of the allowed range.

        """
        self.e_ids = []
        for i in indices:
            if i < len(LookUpData.element_names):
                self.e_ids.append(i)
            else:
                raise ValueError("Index out of range: "+str(i))

    def set_elements_by_name(self, names):
        """Function to define the list of elements to be included in the phase
        diagrams.

        Parameters
        ----------
        names : array-like
            List of element names.

        Raises
        ------
        ValueError
            If an element name is not valid.

        """

        self.e_ids = []
        for name in names:
            if name in LookUpData.element_names:
                self.e_ids.append(LookUpData.element_names.index(name))
            else:
                raise ValueError("Element: "+name+" invalid!")

    def set_order(self, min_, max_):
        """Function to define the order of generated phase diagrams.

        Parameters
        ----------
        min_ : int
            Minimum number of constituents.
        max_ : int
            Maximum number of constituents.

        Raises
        ------
        ValueError
            If min_ or max_ is less than 1.

        """
        if min_ < 1 or max_ < 1:
            raise ValueError("Orders must be greater than 1.")
        self.min_order = min_
        self.max_order = max_

    def set_even_spacing(self, es):
        """Function to define whether to use evenly-spaced compositions (or
        low-denominator).

        Parameters
        ----------
        es : bool
            Boolean indicating the same.

        """
        self.even_spacing = es

    def set_size(self, size):
        """Function to define the number of points per binary diagram (in using
        even spacing) or the maximum denominator (for low-denominator).

        Parameters
        ----------
        size : int
            Desired size parameter.

        Raises
        ------
        ValueError
            If input is less than 2.

        """
        if size < 2:
            raise ValueError("Size must be greater than 1.")
        self.size = size

    def generate_alloy_compositions(self):
        """Function to generate evenly-spaced compositions. Generates
        compositions for all diagrams up to the user-specified Minimum order.
        For example: If the user wants ternary diagrams with a minimum
        spacing of 0.25 this code will generate the following map:
        1 -> ([1.0])
        2 -> ([0.25, 0.75], [0.5, 0.5], [0.75, 0.25])
        3 -> ([0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5])

        Returns
        -------
        output : dict
            A dictionary containing <order, possible compositions> as <key,
            value> pairs. Here, order is the number of elements and possible
            compositions is a list of numpy arrays containing the fractions
            of elements.

        """

        output = {}

        # Add in diagrams of greater order.
        for order in range(self.min_order, self.max_order+1):
            if order == 1:
                tmp_list = []
                tmp_list.append(np.array([1.0]))
                output[order] = tmp_list
                continue

            tmp_list = []
            es = EqualSumCombinations(self.size-1, order)
            for compI in es.get_combinations(self.size - 1, order):
                if 0 in compI:
                    # Don't add compositions from a lower-order diagram.
                    continue
                comp = np.zeros(order)
                for i in range(order):
                    comp[i] = compI[i]/ float(self.size - 1.0)
                tmp_list.append(comp)
            output[order] = tmp_list
        return output

    def generate_crystal_compositions(self):
        """Function to generate compositions with small denominators. Generates
        compositions for all diagrams up to the user-specified Minimum order.
        For example: If the user wants ternary diagrams with a maximum
        denominator of 3 this code will generate the following map:
        1 -> ([1])
        2 -> ([1/3, 2/3], [1/2, 1/2], [2/3, 1/3])
        3 -> ([1/3, 1/3, 1/3])

        Returns
        -------
        output : dict
            A dictionary containing <order, possible compositions> as <key,
            value> pairs. Here, order is the number of elements and possible
            compositions is a list of numpy arrays containing the fractions
            of elements.

        """

        output = {}

        # Add in diagrams of greater order.
        for order in range(self.min_order, self.max_order+1):
            if order == 1:
                tmp_list = []
                tmp_list.append(np.array([1.0]))
                output[order] = tmp_list
                continue

            tmp_list = []
            reduced_examples = []
            for d in range(order, self.size+1):
                es = EqualSumCombinations(d, order)
                for compI in es.get_combinations(d, order):
                    if 0 in compI:
                        # Don't add compositions from a lower-order diagram.
                        continue
                    comp = np.zeros(order)
                    red_comp = np.zeros(order)
                    for i in range(order):
                        comp[i] = float(compI[i])
                        red_comp[i] = comp[i] / d

                    # Check if this composition is already represented.
                    was_found = False
                    for ex in reduced_examples:
                        if np.array_equal(red_comp, ex):
                            was_found = True
                            break

                    if not was_found:
                        tmp_list.append(comp)
                        reduced_examples.append(red_comp)
            output[order] = tmp_list
        return output

    def generate_entries(self):
        """Function to generate the list of entries corresponding to the list of
        compositions, element names specified by the user and the mapping of
        number of elements to compositions.

        Returns
        -------
        entries: array-like
            A list of CompositionEntry's.

        """
        entries = []

        # Get the correct composition mapping.
        compositions = self.generate_alloy_compositions() if \
            self.even_spacing else self.generate_crystal_compositions()

        for (order,list_of_fractions) in iteritems(compositions):

            # Generate all possible combinations of elements of a given order.
            compounds = [list(i) for i in comb(self.e_ids, order)]
            # print order, len(compounds)*len(list_of_fractions)
            for frac in list_of_fractions:
                for compound in compounds:
                    entry = CompositionEntry(element_ids=compound,
                                             fractions=list(frac))
                    entries.append(entry)
        return entries

import re
from itertools import izip
from data.materials.util.LookUpData import LookUpData

class CompositionEntry(object):
    """
    Class that defines a composition entry object. Mainly used to store ids,
    names and fractions of elements belonging to a single compound.
    """

    # Names of each element.
    lp_element_names = LookUpData.element_names

    # Rank of each element (used in display order).
    lp_sorting_order = LookUpData.sorting_order

    # Element ids present in composition.
    element_ids = []

    # Element names present in composition.
    element_names = []

    # Fraction of each element.
    fractions = []

    # Number of atoms in cell (used to convert when printing).
    number_in_cell = -float("inf")

    def __init__(self, composition=None, element_ids=None,
                 element_names=None, fractions=None):
        """
        Make a new instance by paring the composition of an object, provided
        by a string or list of element ids/names and fractions.

        First splits using the regex [A-Z][^A-Z] to separate each component
        of the alloy. This assumes that capitalization is used properly. Then
        splits each component into the alphabetical part (assumed to be
        element name) and the numeric part (amount of element). If no numeric
        part is present, 1 is assumed. An element can appear more than once.

        :param composition: String containing the chemical formula of a
        material.
        :param element_ids: List of integers denoting the element ids.
        :param element_names: List of strings denoting the element names.
        :param fractions: List of floats denoting the element fractions.
        """

        # Parse composition if passed.
        if composition:
            comp_map = self.parse_composition(composition)

            # Crash if nothing is read.
            if not comp_map:
                raise RuntimeError("No composition was read.")

            # Store values.
            self.element_ids = comp_map.keys()
            self.element_names = [self.lp_element_names[i] for i in
                                  self.element_ids]
            self.fractions = comp_map.values()
            self.sort_and_normalize()
        elif fractions and (element_names or element_ids):
            self.set_composition(fractions, element_ids=element_ids,
                                 element_names=element_names)

    def parse_composition(self, composition):
        """
        Function to parse a string containing the composition. Supports
        parentheses and addition compounds (ex: Na_2CO_3-10H_2O). Note,
        will not properly parse addition compounds inside parentheses (ex:
        Na_2(CO_3 - 10H_2O)_1).
        :param composition: String describing a composition.
        :return: Dictionary containing element ids and fractions as keys and
        values respectively.
        """

        # Check for a guest structure (ex: Al2O3-2H20).
        start_guest = -1 if "-" not in composition else composition.index("-")
        pos = -1 if chr(183) not in composition else composition.index(chr(183))

        if pos != -1 and (pos < start_guest or start_guest == -1):
            start_guest = pos

        # If this composition doesn't contain a guest either.
        if start_guest == -1:

            # Check for ({['s
            start_paren = -1 if '(' not in composition else composition.index(
                '(')
            pos = -1 if '{' not in composition else composition.index(
                '{')
            if pos != -1 and (pos < start_paren or start_paren == -1):
                start_paren = pos
            pos = -1 if '[' not in composition else composition.index(
                '[')
            if pos != -1 and (pos < start_paren or start_paren == -1):
                start_paren = pos

            # If none found, parse as normal.
            if start_paren == -1:
                return self.parse_element_amounts(composition)

            # First, get the part of string before the parens.
            first_part = composition[:start_paren]

            # Next, find the portion inside of the parens.
            paren_type = composition[start_paren]
            pos = start_paren
            paren_balance = 1
            while paren_balance > 0:
                # Increment position.
                pos += 1

                # Check whether we are past the end of the string.
                if pos > len(composition):
                    raise ValueError("Missing close paren. Start: "
                                     ""+composition[start_paren:start_paren+3])

                # Get the character at the current position.
                cur_char = composition[pos]

                if paren_type == '(':
                    if cur_char == '(':
                        paren_balance += 1
                    elif cur_char == ')':
                        paren_balance -= 1
                elif paren_type == '{':
                    if cur_char == '{':
                        paren_balance += 1
                    elif cur_char == '}':
                        paren_balance -= 1
                elif paren_type == '[':
                    if cur_char == '[':
                        paren_balance += 1
                    elif cur_char == ']':
                        paren_balance -= 1
                else:
                    raise ValueError("Unrecognized paren type: "+paren_type)

            # Get the multiplier of the composition inside the parens.
            end_paren = pos
            pos += 1
            mult = ""
            while pos < len(composition) and (composition[pos].isdigit() or
                                                  composition[pos] == "."):
                mult += composition[pos]
                pos += 1

            paren_mult = 1.0 if not mult else float(mult)

            # Get the portion insides of those parens, and end portion.
            inside_paren = composition[start_paren+1:end_paren]
            end_portion = composition[pos:]

            total_comp = self.parse_element_amounts(first_part)
            add_comp = self.parse_composition(inside_paren)

            self.combine_compositions(total_comp, add_comp, paren_mult)

            add_comp = self.parse_composition(end_portion)

            self.combine_compositions(total_comp, add_comp, 1.0)
            return total_comp

        else:
            # Find how many guests.
            end_host = start_guest
            pos = start_guest + 1
            mult = ""
            while pos < len(composition) and (composition[pos].isdigit() or
                                                  composition[pos] == "."):
                mult += composition[pos]
                pos += 1
            start_guest = pos
            guest_mult = 1.0 if not mult else float(mult)

            # Split compound.
            host_part = composition[:end_host]
            guest_part = composition[start_guest:]

            # Compute them separately.
            host_comp = self.parse_composition(host_part)
            guest_comp = self.parse_composition(guest_part)
            self.combine_compositions(host_comp, guest_comp, guest_mult)
            return host_comp

    def set_composition(self, amounts, element_ids=None, element_names=None,
                        to_sort=True):
        """
        Function to set the composition of this entry. Checks to make sure
        all elements have positive amounts.
        :param amounts: List of amounts for each element.
        :param element_ids: List of element ids.
        :param element_names: List of element names.
        :return:
        """

        if element_ids is None and element_names is None:
            raise ValueError("Must include a list of element names or ids "
                             "along with amounts!")
        if element_names is None:
            if len(amounts) != len(element_ids):
                raise ValueError("Array lengths of amounts and element_ids "
                                 "must be equal!")
            self.element_ids = element_ids
        else:
            if len(amounts) != len(element_names):
                raise ValueError("Array lengths of amounts and element_names "
                                 "must be equal!")
            self.element_names = element_names
            self.element_ids = [self.lp_element_names.index(elem) for elem in
                                element_names]
        self.fractions = amounts
        self.sort_and_normalize(to_sort)

    def __copy__(self):
        x = CompositionEntry()
        x.element_ids = list(self.element_ids)
        x.fractions = list(self.fractions)
        return x

    def parse_element_amounts(self, composition):
        """
        Function to compute fractions of element given a string of elements
        and amounts.
        :param composition: Composition as a string.
        :return: Dictionary containing element ids and fractions as keys and
        values respectively.

        """
        # Create temporary entry.
        tmp_entry = {}

        # Add up all the constituents.
        comp_iter = re.compile(r"[A-Z][^A-Z]*").finditer(composition)
        elem_pattern = re.compile(r"[A-Z][a-z]?")
        fraction_pattern = re.compile(r"[.0-9]+")
        for comp in comp_iter:
            component = comp.group()

            # Get the element information.
            elem_matcher = elem_pattern.match(component)
            if not elem_matcher:
                raise ValueError("Something has gone horribly wrong!")
            element = elem_matcher.group()
            if element == "D" or element == "T":
                element = "H"
            if element not in self.lp_element_names:
                raise ValueError("Element "+element+" not recognized")
            element_id = self.lp_element_names.index(element)

            # Get the amount of this element.
            fraction_matcher = fraction_pattern.search(component)
            fraction = 1.0
            if fraction_matcher:
                try:
                    fraction = float(fraction_matcher.group())
                except ValueError:
                    raise ValueError("Element amount "+
                        fraction_matcher.group()+" not a valid number.")

            # Skip if fraction is zero.
            if fraction == 0:
                continue

            if element_id in tmp_entry:
                tmp_entry[element_id] += fraction
            else:
                tmp_entry[element_id] = fraction

        return tmp_entry


    def get_element_names(self):
        """
        Function to get the element names in the composition.
        :return: List of element names.
        """
        return self.element_names

    def get_element_ids(self):
        """
        Function to get the element ids in the composition.
        :return: List of element ids.
        """
        return self.element_ids

    def get_element_fractions(self):
        """
        Function to get the element fractions in the composition.
        :return: List of element fractions.
        """
        return self.fractions

    def get_element_fraction(self, name=None, id=None):
        """
        Function to get the element fraction given either the name or the id
        of the element.
        :param name: Name of the element.
        :param id: Id of the element.
        :return: Elemental fraction.
        """

        e_id = id
        if e_id is None:
            for i in range(len(self.lp_element_names)):
                if self.lp_element_names[i].lower() == name.lower():
                    e_id = i

        for i in range(len(self.element_ids)):
            if self.element_ids[i] == e_id:
                return self.fractions[i]
        return 0.0

    def __cmp__(self, other):
        """
        Function to compare two composition entries.
        :param other: Other composition entry to compare.
        :return: -1 if self < other , 1 if self > other or 0 if self = other.
        """
        if isinstance(other, CompositionEntry):
            # If this has more elements, this is greater.
            if len(self.element_ids) != len(other.element_ids):
                return -1 if len(self.element_ids) < len(other.element_ids) \
                    else 1
            # Check which has greater element fractions.
            for i in range(len(other.element_ids)):
                if self.element_ids[i] != other.element_ids[i]:
                    return -1 if self.element_ids[i] < other.element_ids[i] \
                        else 1
                elif self.fractions[i] != other.fractions[i]:
                    return -1 if self.fractions[i] < other.fractions[i] else 1
            # We have concluded that they are equal.
        return 0

    def __hash__(self):
        """
        Function to compute the hashcode of a given entry. Computes the
        hashcode of the list of element ids and fractions separately. Then
        does the logical XOR operation between the two hashcodes and 1 and
        returns the result.
        :return: Hashcode of the entry.
        """

        h1 = h2 = 0
        for e,f in izip(self.element_ids, self.fractions):
            h1 += 31*h1 + hash(e)
            h2 += 31*h2 + hash(f)

        return h1 ^ h2 ^ 1

    def __eq__(self, other):
        """
        Function to compare the equality between two composition entries.
        :param other: Other composition entry to compare.
        :return: True if they are equal and False otherwise.
        """

        if isinstance(other, CompositionEntry):
            if len(self.element_ids) != len(other.element_ids):
                return False
            return self.element_ids == other.element_ids and \
                   self.element_names == other.element_names and \
                   self.fractions == other.fractions
        return False

    def sort_and_normalize(self, to_sort=True):
        """
        Function to sort the element ids based on their electronegativity
        order and normalizes the fractions. Makes sure the entry is in a
        proper format. Must be run from constructor.
        :return:
        """

        # Sort elements based on the electronegativity order.
        if to_sort:
            tmp_tuple = zip(self.element_ids, self.fractions)
            tmp_tuple.sort(key=lambda (x,y): self.lp_sorting_order[x])
        else:
            tmp_tuple = zip(self.element_ids, self.fractions)

        # Normalize the fractions.
        self.number_in_cell = sum(self.fractions)

        self.element_ids = []
        self.element_names = []
        self.fractions = []

        for e_id, f in tmp_tuple:
            if f > 0.0:
                f_ = float(f) / self.number_in_cell
                if e_id not in self.element_ids:
                    self.element_ids.append(e_id)
                    self.element_names.append(self.lp_element_names[e_id])
                    self.fractions.append(f_)
                else:
                    idx = self.element_ids.index(e_id)
                    self.fractions[idx] += f_

    def combine_compositions(self, total_comp, add_comp, multiplier):
        """
        Function to add one composition entry to another.
        :param total_comp: Composition entry to be added to.
        :param add_comp: Composition entry to add.
        :param multiplier: Multiplier for the added part.
        :return: Combined compositions.
        """

        for e,f in add_comp.iteritems():
            # If total_comp contains this element.
            if e in total_comp:
                total_comp[e] += multiplier * f
            else:
                total_comp[e] = multiplier * f

    @classmethod
    def print_number(self, fraction, n_in_formula_unit):
        """
        Function to print out the number of atoms in a formula unit for each
        element given its fraction.
        :param fraction: List of element fractions to be printed.
        :param n_in_formula_unit: Number of atoms in a formula unit.
        :return: Result formatted as a string.
        """

        output = []
        for f in fraction:
            expanded = f * n_in_formula_unit
            if abs(expanded - round(expanded)) < 0.0001:
                output.append("" if int(expanded) == 1 else "{0:d}".format(
                    int(expanded)))
            else:
                output.append("{0:1.3f}".format(expanded))
        return output

    def __str__(self):
        """
        Function to print the composition entry in the proper format.
        :return: Correctly formatted output.
        """
        if not self.element_names:
            return "Elem_list not defined"
        else:
            output = ""
            numbers = self.print_number(self.fractions, self.number_in_cell)
            for i in range(len(self.element_names)):
                output += self.element_names[i]
                output += "" if not numbers else numbers[i]

            return output

    @classmethod
    def import_composition_list(self, file_path):
        composition_list = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                words = line.strip()
                entry = CompositionEntry(composition=words)
                composition_list.append(entry)

        return composition_list

    @classmethod
    def import_values_list(self, file_path):
        property_list = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                property_list.append(float(line.strip()))

        return property_list
import types
import numpy as np
from ....data.materials.CompositionEntry import CompositionEntry

class CompositionDistanceFilter:
    """Class to filter compositions based on distance from a target composition.
    Filters any composition where the maximum change in any element is less
    than a certain value.

    Attributes
    ----------
    target_composition : CompositionEntry
        Composition of the target.
    threshold : float
        Threshold to be used.


    """
    def __init__(self):
        """Initialize field variables here.
        """

        # Target composition.
        self.target_composition = None

        # Threshold distance.
        self.threshold = 0.0

    def set_target_composition(self, entry):
        """Function to define the target composition.

        Parameters
        ----------
        entry : CompositionEntry
            Desired target composition.

        """
        self.target_composition = entry

    def set_distance_threshold(self, distance):
        """Function to define the threshold composition distance. Here distance
        is defined as the maximum change in the fraction of any element.

        Parameters
        ----------
        distance : float
            Target threshold in %


        """
        self.threshold = distance / 100.0

    @classmethod
    def compute_distance(self, entry_1, entry_2, p):
        """Function to compute the distance between two entries. Distance is
        defined as the L_p norm of the distance between element fractions.

        Parameters
        ----------
        entry_1 : CompositionEntry
            Composition of entry 1.
        entry_2 : CompositionEntry
            Composition of entry 2.
        p : int
            Desired norm.

        Returns
        -------
        output : float
            Distance between two entries.

        """

        # Get the set of common elements with fractions greater than zero to
        # consider.
        elements = set()
        for e1 in entry_1.get_element_ids():
            elements.add(e1)

        for e2 in entry_2.get_element_ids():
            elements.add(e2)

        # Compute differences
        dist = 0.0
        for e in elements:
            diff = entry_1.get_element_fraction(id=e) - \
                   entry_2.get_element_fraction(id=e)
            if p == 0:
                if diff != 0:
                    dist += 1
            elif p == -1:
                dist = max(dist, diff)
            else:
                dist += abs(diff)**p

        # Compute distance
        if p == 0 or p == -1:
            return dist

        return dist**(1.0 / p)

    def label(self, entries):
        """Function to compute labels of composition entries indicating whether
        or not they are within the threshold of the target composition.

        Parameters
        ----------
        entries : array-like
            A list of CompositionEntry's.

        Returns
        -------
        label : array-like
            A numpy array containing True if the entry is within bounds of
            the target composition and False otherwise.

        Raises
        ------
        ValueError
            If input is not of type list.
            If items in the list are not CompositionEntry instances.
        """

        # Raise exception if input argument is not of type list of
        # CompositionEntry's.
        if not isinstance(entries, list):
            raise ValueError("Argument should be of type list of "
                             "CompositionEntry's.")
        elif (entries and not isinstance(entries[0], CompositionEntry)):
            raise ValueError("Argument should be of type list of "
                             "CompositionEntry's.")

        target_elements = self.target_composition.get_element_ids()
        target_fractions = self.target_composition.get_element_fractions()

        label = np.array([False]*len(entries))
        for i,entry in enumerate(entries):
            within_bounds = True
            for e in range(len(target_elements)):
                v1 = entry.get_element_fraction(id=target_elements[e])
                if abs(v1 - target_fractions[e]) > self.threshold:
                    within_bounds = False
                    break
            label[i] = within_bounds

        return label
import itertools
import numpy as np
from ...data.materials.CompositionEntry import CompositionEntry

class OxidationStateGuesser:
    """Class to predict the likely oxidation states of a material, given its
    input composition.

    Attributes
    ----------
    electronegativity : array-like
        A list of electronegativity values (float).
    oxidation_states : array-like
        A 2-D numpy array containing the property values for all the elements.

    """
    electronegativity = np.zeros(0)
    oxidationstates = np.zeros(0, dtype=object)

    def set_electronegativity(self, values):
        """Function to set the electronegativity values.

        Parameters
        ----------
        values : array-like
            Numpy array containing electronegativity values for all the
            elements.

        Returns
        -------

        """
        self.electronegativity = values

    def set_oxidationstates(self, values):
        """Function to set the oxidation states values.

        Parameters
        ----------
        values : array-like
            2-D numpy array containing oxidation states values for all the
            elements.

        Returns
        -------

        """
        self.oxidationstates = values

    def get_possible_states(self, entry):
        """Function to compute all the possible oxidation states of a material,
        given its input composition.
        The function works by finding all
        combinations of non-zero oxidation states for each element, computing
        which are the most reasonable, and finding which of those have minimum
        value of
        sum_{i,j} (chi_i - chi_j)*(c_i - c_j) for i < j
        where chi_i is the electronegativity and c_i is the oxidation. This
        biases the selection towards the more electronegative elements being
        more negatively charged.

        Parameters
        ----------
        entry : CompositionEntry
            A CompositionEntry object.

        Returns
        -------
        output : array-like
            A numpy array containing the list of possible oxidation states
            arranged in the order mentioned above.

        Raises
        ------
        ValueError
            If input is empty.
            If input is not a CompositionEntry object.
            If electronegativity or oxidationstates haven't been set.

        """

        # Make sure entry is not empty.
        if not entry:
            raise ValueError("Input argument cannot be empty. Please pass a "
                             "valid argument.")


        # Make sure entry is of type CompositionEntry.
        if not isinstance(entry, CompositionEntry):
            raise ValueError("Entry must be of type CompositionEntry.")

        # Make sure electronegativity and oxidation states are not empty.
        if not self.electronegativity.size or not self.oxidationstates.size:
            raise ValueError("Electronegativity or OxidationStates values are "
                             "not initialized. Set them and try again.")

        # Initialize list of possible states.
        possible_states = []

        # Get element ids and fractions.
        elem_ids = entry.get_element_ids()
        elem_fracs = entry.get_element_fractions()
        if len(elem_ids) == 1:
            return np.asarray([])

        # List of all states.
        states = []
        for id in elem_ids:
            states.append(self.oxidationstates[id])

        # Generate all combinations of those charge states, only store the
        # ones that are charge balanced.
        for state in itertools.product(*states):
            charge = np.dot(state, elem_fracs)
            # If charge is balanced, add state to the list of possible states.
            if abs(charge) < 1E-6:
                possible_states.append(list(state))

        if len(possible_states) < 2:
            return np.asarray(possible_states)

        # Compute the summation mentioned in the function description.
        rankVal = np.zeros(len(possible_states))
        for s in range(len(possible_states)):
            state = possible_states[s]
            tmp_val = 0.0
            for i in range(len(state)):
                for j in range(i+1,len(state)):
                    tmp_val += (self.electronegativity[elem_ids[i]] -
                                self.electronegativity[elem_ids[j]]) * (
                        state[i] - state[j])
            rankVal[s] = tmp_val

        # Order them based on electronegativity rank.
        output = [ps for i, ps in sorted(zip(rankVal, possible_states))]
        return np.asarray(output)

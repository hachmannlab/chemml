# -*- coding: utf-8 -*-
# py2 and py3
from builtins import zip as izip
# from itertools import izip
import numpy as np
import pandas as pd
from scipy.linalg import lu
from scipy.optimize import linprog
from .LookUpData import LookUpData
from ..CompositionEntry import CompositionEntry

class GCLPCalculator:
    """Class that uses Grand-Canonical Linear Programming (GCLP) method to
    compute phase equilibria.
    How to use this class:
    1. Load in phase equilibria data with `self.add_phase`, `self.set_mu` or
    `self.add_phases`.
    2. Compute equilibrium with `self.run_GCLP`.

    Attributes
    ----------
    lp_element_names : array-like
        List of element names.
    phases : dict
        Dictionary containing CompositionEntry as the key and energy (float)
        as the value. Phases to consider for equilibria and their energy.
        Only contains the lowest-energy phase at each entry.

    References
    ----------
    .. [1] A. R. Akbarzadeh, V. Ozoliņš, and C. Wolverton, “First-Principles
    Determination of Multicomponent Hydride Phase Diagrams: Application to
    the Li-Mg-N-H System,” Advanced Materials, vol. 19, no. 20,
    pp. 3233--3239, Sep. 2007.

    """
    lp_element_names = LookUpData.element_names

    def __init__(self):
        """Function to create instance and initialize fields.
        Initialize a GCLP calculator. Sets the chemical potential of each
        element to be 0.
        """
        self.phases = {}
        # Add for each element.
        for elem in self.lp_element_names:
            entry = CompositionEntry(element_names=[elem], fractions=[1.0])
            self.phases[entry] = 0.0

    def set_mu(self, elem, mu):
        """Function to set the chemical potential of an element.

        Parameters
        ----------
        elem : str
            Abbreviation of element.

        mu : float
            Desired chemical potential.

        """
        entry = CompositionEntry(composition=elem)
        if len(entry.get_element_ids()) != 1:
            raise ValueError("Not an element "+elem)
        self.phases[entry] = mu

    def add_phases(self, entries, energies):
        """Function to set many phase energies.

        Parameters
        ----------
        entries : array-like
            A list of CompositionEntry's.

        energies : array-like
            A list of energy values as floats.

        """
        if isinstance(energies,pd.DataFrame):
            if energies.shape[1] == 1:
                energies = energies.values
            else:
                msg = "a dataframe with one column is required"
                raise Exception(msg)
        if isinstance(entries,pd.DataFrame):
            if entries.shape[1] == 1:
                entries = entries.values
            else:
                msg = "a dataframe with one column is required"
                raise Exception(msg)

        for entry,energy in izip(entries, energies):
            # if has measurement
            self.add_phase(entry, energy)

    def add_phase(self, entry, energy):
        """Function to set one phase.

        Parameters
        ----------
        entry : CompositionEntry
            Composition of phase.

        energy : float
            Energy of phase.


        """
        if entry not in self.phases:
            # Add if there is no current entry at this composition.
            self.phases[entry] = float(energy)
        elif self.phases[entry] > energy:
            # If there is a phase, update only if new energy is lower than
            # current.
            self.phases[entry] = float(energy)

    def get_num_phases(self):
        """
        Function to get the number of phases being considered for GCLP.

        Returns
        -------
        output : int
            The number of phases being considered.

        """
        return len(self.phases)

    def run_GCLP(self, composition):
        """Function to compute the ground state phase equilibria for a
        certain composition.

        Parameters
        ----------
        composition : CompositionEntry
            Composition to be considered.

        Returns
        -------
        ground_state_energy : float
            Ground state energy.
        equilibrium : dict
            Dictionary containing the phase composition (CompositionEntry) as
            key and the fractions (float) as values.

        Raises
        ------
        TypeError
            If composition is not CompositionEntry.

        """
        if not isinstance(composition, CompositionEntry):
            raise TypeError("Composition should be of type CompositionEntry!")

        cur_elements = composition.get_element_ids()
        cur_fractions = composition.get_element_fractions()

        # List of composition entries.
        components = []

        # List of energies.
        energies = []

        # Get the current possible phases (i.e., those that contain
        # exclusively the elements in the current compound.
        for entry in self.phases:
            this_elements = entry.get_element_ids()
            # Check whether this entry is in the target phase diagram.
            if set(this_elements) <= set(cur_elements):
                components.append(entry)
                energies.append(self.phases[entry])

        # Set up constraints.
        # Type #1: Mass conservation.
        l_components = len(components)
        l_composition = len(cur_elements)
        a_eq = np.ones((l_composition + 1, l_components))
        b_eq = np.ones(l_composition + 1)
        for i in range(l_composition):
            b_eq[i] = cur_fractions[i]
            for j in range(l_components):
                a_eq[i][j] = components[j].get_element_fraction(
                    id=cur_elements[i])

        # Type #2: Normalization.
        # Taken care of when we initialized a_eq and b_eq to ones.

        # Perform LU decomposition to check if there are any linearly
        # dependent rows in the matrix a_eq. For some reason, linprog can't
        # handle linearly dependent matrices.
        _, u = lu(a_eq, permute_l=True)

        mask = np.all(abs(u) < 1e-14, axis=1)
        indices = [i for i in range(len(mask)) if mask[i]]
        if indices:
            a_eq = np.delete(a_eq, indices, axis=0)
            b_eq = np.delete(b_eq, indices)

        c = np.array(energies)

        # Call LP solver and store result.
        res = linprog(c=c, A_eq=a_eq, b_eq=b_eq)
        equilibrium = {}
        equilibrium_fractions = res.x
        for i in range(l_components):
            if equilibrium_fractions[i] > 1e-6:
                equilibrium[components[i]] = equilibrium_fractions[i]

        # Add zero to avoid returning -0.0 values.
        ground_state_energy = res.fun + 0

        return ground_state_energy, equilibrium

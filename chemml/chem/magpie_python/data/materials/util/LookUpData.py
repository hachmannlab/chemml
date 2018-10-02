from __future__ import print_function
from builtins import range

import numpy as np
import sys
import os

class LookUpData:
    """Class to look up properties of elements stored in files.

    Attributes
    ----------
    element_ids : dict
        Dictionary containing element names (str) as the keys and index (int)
        in the periodic table as the values. Index is computed as Z - 1,
        where Z is the atomic number.
    element_names : array-like
        A list of abbreviated element names in the periodic table.
    sorting_order : array-like
        A list of electronegativity-based indices (int) used for printing the
        composition in a pretty format.
    element_order : dict
        Dictionary containing element names (str) as the keys and the sorting
        order (int) as values.
    all_properties : array-like
        A list of properties for which property-tables exist in
        magpie_python/lookup-data/.
    this_file_path : str
        Path to this file.
    abs_path : str
        Absolute path to the lookup-data directory.
    pair_abs_path : str
        Absolute path to the pair properties lookup directory.
    """

    # Element indices of the periodic table.
    element_ids = {"H": 0, "He": 1, "Li": 2, "Be": 3, "B": 4, "C": 5,
                   "N": 6, "O": 7, "F": 8, "Ne": 9, "Na": 10, "Mg": 11,
                   "Al": 12, "Si": 13, "P": 14, "S": 15, "Cl": 16,
                   "Ar": 17, "K": 18, "Ca": 19, "Sc": 20, "Ti": 21,
                   "V": 22, "Cr": 23, "Mn": 24, "Fe": 25, "Co": 26,
                   "Ni": 27, "Cu": 28, "Zn": 29, "Ga": 30, "Ge": 31,
                   "As": 32, "Se": 33, "Br": 34, "Kr": 35, "Rb": 36,
                   "Sr": 37, "Y": 38, "Zr": 39, "Nb": 40, "Mo": 41,
                   "Tc": 42, "Ru": 43, "Rh": 44, "Pd": 45, "Ag": 46,
                   "Cd": 47, "In": 48, "Sn": 49, "Sb": 50, "Te": 51,
                   "I": 52, "Xe": 53, "Cs": 54, "Ba": 55, "La": 56,
                   "Ce": 57, "Pr": 58, "Nd": 59, "Pm": 60, "Sm": 61,
                   "Eu": 62, "Gd": 63, "Tb": 64, "Dy": 65, "Ho": 66,
                   "Er": 67, "Tm": 68, "Yb": 69, "Lu": 70, "Hf": 71,
                   "Ta": 72, "W": 73, "Re": 74, "Os": 75, "Ir": 76,
                   "Pt": 77, "Au": 78, "Hg": 79, "Tl": 80, "Pb": 81,
                   "Bi": 82, "Po": 83, "At": 84, "Rn": 85, "Fr": 86,
                   "Ra": 87, "Ac": 88, "Th": 89, "Pa": 90, "U": 91,
                   "Np": 92, "Pu": 93, "Am": 94, "Cm": 95, "Bk": 96,
                   "Cf": 97, "Es": 98, "Fm": 99, "Md": 100, "No": 101,
                   "Lr": 102, "Rf": 103, "Db": 104, "Sg": 105,
                   "Bh": 106, "Hs": 107, "Mt": 108, "Ds": 109,
                   "Rg": 110, "Cn": 111}

    element_names = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                     "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
                     "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
                     "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
                     "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
                     "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba",
                     "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
                     "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
                     "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
                     "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",
                     "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md",
                     "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
                     "Rg", "Cn"]

    sorting_order = [91, 1, 26, 62, 85, 102, 109, 111, 112, 2, 24, 53, 64,
                     74, 90, 104, 110, 3, 20, 27, 55, 60, 66, 68, 61, 72, 73,
                     78, 75, 67, 71, 83, 89, 103, 107, 108, 21, 25, 36, 54,
                     63, 88, 76, 92, 97, 93, 79, 69, 70, 80, 86, 87, 106,
                     105, 19, 22, 28, 30, 31, 32, 4, 33, 5, 34, 35, 37, 38,
                     39, 40, 6, 41, 43, 58, 100, 77, 94, 95, 98, 101, 81, 65,
                     99, 84, 82, 96, 7, 18, 23, 29, 44, 59, 57, 56, 42, 45,
                     46, 47, 48, 49, 50, 51, 52, 8, 9, 10, 11, 12, 13, 14,
                     15, 16, 17]

    element_order = {"H": 91, "He": 1, "Li": 26, "Be": 62, "B": 85, "C": 102,
                     "N": 109, "O": 111, "F": 112, "Ne": 2, "Na": 24,
                     "Mg": 53, "Al": 64, "Si": 74, "P": 90, "S": 104,
                     "Cl": 110, "Ar": 3, "K": 20, "Ca": 27, "Sc": 55,
                     "Ti": 60, "V": 66, "Cr": 68, "Mn": 61, "Fe": 72,
                     "Co": 73, "Ni": 78, "Cu": 75, "Zn": 67, "Ga": 71,
                     "Ge": 83, "As": 89, "Se": 103, "Br": 107, "Kr": 108,
                     "Rb": 21, "Sr": 25, "Y": 36, "Zr": 54, "Nb": 63,
                     "Mo": 88, "Tc": 76, "Ru": 92, "Rh": 97, "Pd": 93,
                     "Ag": 79, "Cd": 69, "In": 70, "Sn": 80, "Sb": 86,
                     "Te": 87, "I": 106, "Xe": 105, "Cs": 19, "Ba": 22,
                     "La": 28, "Ce": 30, "Pr": 31, "Nd": 32, "Pm": 4,
                     "Sm": 33, "Eu": 5, "Gd": 34, "Tb": 35, "Dy": 37,
                     "Ho": 38, "Er": 39, "Tm": 40, "Yb": 6, "Lu": 41,
                     "Hf": 43, "Ta": 58, "W": 100, "Re": 77, "Os": 94,
                     "Ir": 95, "Pt": 98, "Au": 101, "Hg": 81, "Tl": 65,
                     "Pb": 99, "Bi": 84, "Po": 82, "At": 96, "Rn": 7,
                     "Fr": 18, "Ra": 23, "Ac": 29, "Th": 44, "Pa": 59,
                     "U": 57, "Np": 56, "Pu": 42, "Am": 45, "Cm": 46,
                     "Bk": 47, "Cf": 48, "Es": 49, "Fm": 50, "Md": 51,
                     "No": 52, "Lr": 8, "Rf": 9, "Db": 10, "Sg": 11,
                     "Bh": 12, "Hs": 13, "Mt": 14, "Ds": 15, "Rg": 16,
                     "Cn": 17}

    # List of all the properties (except Abbreviation, IonizationEnergies,
    # OxidationStates) present in the directory: lookup-data/
    all_properties = ["AtomicVolume", "AtomicWeight",
                      "BoilingTemp", "BoilingT", "BulkModulus", "Column",
                      "CovalentRadius", "Density", "DipolePolarizability",
                      "ElectronAffinity", "Electronegativity",
                      "FirstIonizationEnergy", "FusionEnthalpy", "GSbandgap",
                      "GSenergy_pa", "GSestBCClatcnt", "GSestFCClatcnt",
                      "GSmagmom", "GSvolume_pa", "HeatCapacityMass",
                      "HeatCapacityMolar", "HeatFusion", "HHIp", "HHIr",
                      "ICSDVolume", "IsAlkali", "IsDBlock", "IsFBlock",
                      "IsMetalloid", "IsMetal", "IsNonmetal", "MeltingT",
                      "MendeleevNumber", "MiracleRadius", "NdUnfilled",
                      "NdValence", "NfUnfilled", "NfValence", "NpUnfilled",
                      "NpValence", "NsUnfilled", "NsValence", "Number",
                      "NUnfilled", "NValance", "n_ws^third", "phi",
                      "Polarizability", "Row", "ShearModulus",
                      "SpaceGroupNumber", "ZungerPP-r_d", "ZungerPP-r_pi",
                      "ZungerPP-r_p", "ZungerPP-r_sigma", "ZungerPP-r_s"]

    this_file_path = os.path.dirname(__file__)
    abs_path = os.path.join(this_file_path, "../../../lookup-data/")
    pair_abs_path = abs_path+"pair/"

    @classmethod
    def load_property(self, property):
        """Function to load a specific property from the directory containing
        all the lookup tables.

        Parameters
        ----------
        property : str
            Property whose values need to be loaded.

        Returns
        -------
        values : array-like
            A numpy array containing the property values for all the elements.

        Raises
        ------
        IOError
            If property table doesn't exist.

        """

        # IonizationEnergies and OxidationStates are 2-D arrays. So treat
        # them differently.
        if property == "IonizationEnergies" or property == "OxidationStates":
            return self.load_special_property(property)

        # Initialize the numpy array.
        values = np.zeros(len(self.element_ids), dtype=np.float)
        values.fill(np.nan)

        # Property file name.
        file = self.abs_path + property + ".table"
        try:
            prop_file = open(file, 'r')
        except IOError:
            raise IOError("File {} doesn't exist!!! Please make sure you " \
                  "specify the correct file name".format(file))

        else:
            for i in range(values.size):
                line = prop_file.readline().strip()
                try:
                    values[i] = float(line)
                except ValueError:
                    # Line is a string, which implies that data is missing.
                    # So we let the value of NaN stay.
                    continue
            prop_file.close()
        return values

    @classmethod
    def load_pair_property(self, property):
        """Function to load property of a binary system.

        Parameters
        ----------
        property : str
            Property whose values need to be loaded.

        Returns
        -------
        values : array-like
            A 2-D numpy array containing the property values for all the
            elements.

        Raises
        ------
        IOError
            If property table doesn't exist.

        """

        # Initialize the 2-D numpy array.
        values = np.zeros(len(self.element_ids), dtype=object)
        for i in range(len(self.element_ids)):
            values[i] = np.zeros(i, dtype=float)

        # Property file name.
        file = self.pair_abs_path + property + ".table"

        try:
            prop_file = open(file, 'r')
        except IOError:
            raise IOError("File {} doesn't exist!!! Please make sure you " \
                  "specify the correct file name".format(file))
        else:
            for line in prop_file.readlines():
                words = line.strip().split()
                if len(words) < 3:
                    continue
                else:
                    if words[0] not in self.element_ids or words[1] not in \
                            self.element_ids:
                        continue
                    elemA = self.element_ids[words[0]]
                    elemB = self.element_ids[words[1]]
                    if words[2].endswith("\n"):
                        print(line)
                        sys.exit(1)
                    values[max(elemA,elemB)][min(elemA,elemB)] = float(words[2])
            prop_file.close()
        return values

    @classmethod
    def load_pair_properties(self, properties):
        """Function to load multiple pair property values from the directory
        containing all the lookup tables.

        Parameters
        ----------
        properties : array-like
            A list of pair properties whose values need to be loaded.

        Returns
        -------
        values: dict
            A dictionary containing property name (str) as the key and a
            numpy array (float) containing the pair property values.

        """

        # Initialize the dictionary.
        values = {}

        for prop in properties:
            values[prop] = self.load_pair_property(prop)
        return values

    @classmethod
    def load_properties(self, properties):
        """Function to load multiple property values from the directory
        containing all the lookup tables.

        Parameters
        ----------
        properties : array-like
            A list of properties whose values need to be loaded.

        Returns
        -------
        values: dict
            A dictionary containing property name (str) as the key and a
            numpy array (float) containing the pair property values.

        """

        # Initialize the dictionary.
        values = {}

        for prop in properties:
            values[prop] = self.load_property(prop)
        return values

    @classmethod
    def load_special_property(self, property):
        """Function to load the special property files related to
        IonizationEnergies and OxidationStates.

        Parameters
        ----------
        property : str
            Property whose values need to be loaded.

        Returns
        -------
        values : array-like
            A 2-D numpy array containing the property values for all the
            elements.

        Raises
        ------
        IOError
            If property table doesn't exist.

        """
        # Property file name.
        file = self.abs_path + property + ".table"

        # Initialize the list.
        tmp_values = []

        try:
            prop_file = open(file, 'r')
        except IOError:
            print ("File {} doesn't exist!!! Please make sure you " \
                  "specify the correct file name".format(file))
            sys.exit(1)
        else:
            for line in prop_file.readlines():
                words = line.strip().split()
                tmp_list = []
                for word in words:
                    tmp_list.append(float(word))
                tmp_values.append(tmp_list)
            prop_file.close()

        values = np.zeros(len(tmp_values), dtype=object)
        for i in range(len(tmp_values)):
            values[i] = np.asarray(tmp_values[i], dtype=float)
        return values
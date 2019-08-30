from __future__ import print_function
import pkg_resources
import os
import pandas as pd

from chemml.chem import Molecule


def load_cep_homo():
    """Load and return a small sample of HOMO energies of organic photovoltaic candidates from CEP database (regression).
    Clean Energy Project (CEP) database is available at: https://cepdb.molecularspace.org
    The unit of HOMO (highest occupied molecular orbitals) energies is electron Volt (eV).
    The photovaltaic candidates are provided using SMILES representation.

    =================   ==============
    rows                           500
    Columns                          2
    headers             smiles,homo_eV
    molecules rep.              SMILES
    Features                         0
    Returns               2 dataframes
    =================   ==============

    Returns
    -------
    smiles : pandas dataframe
        The SMILES representation of molecules, shape: (500,1)

    homo : pandas dataframe
        The HOMO energies of the molecules (eV), shape: (500,1)

    Examples
    --------
    >>> from chemml.datasets import load_cep_homo
    >>> smi, homo  = load_cep_homo()
    >>> print(list(smi.columns))
    ['smiles']
    >>> print(homo.shape)
    (500, 1)
    """
    DATA_PATH = pkg_resources.resource_filename('chemml', os.path.join('datasets','data','cep_homo.csv'))
    df = pd.read_csv(DATA_PATH)

    smi = pd.DataFrame(df['smiles'], columns=['smiles'])
    homo = pd.DataFrame(df['homo_eV'], columns=['homo_eV'])

    return smi, homo


def load_organic_density():
    """Load and return 500 small organic molecules with their density and molecular descriptors.

    =================   ======================
    rows                                   500
    Columns                                202
    last twoo headers     smiles,density_Kg/m3
    molecules rep.                      SMILES
    Features                               200
    Returns                       3 dataframes
    =================   ======================

    Returns
    -------
    smiles : pandas dataframe
        The SMILES representation of molecules, shape: (500,1)

    density : pandas dataframe
        The density of molecules (Kg/m3), shape: (500,1)

    features : pandas dataframe
        The molecular descriptors of molecules, shape: (500,200)

    Examples
    --------
    >>> from chemml.datasets import load_organic_density
    >>> smi, density, features = load_organic_density()
    >>> print(list(smi.columns))
    ['smiles']
    >>> print(features.shape)
    (500, 200)
    """
    DATA_PATH = pkg_resources.resource_filename('chemml', os.path.join('datasets','data','moldescriptor_density_smiles.csv'))
    df = pd.read_csv(DATA_PATH)

    smi = pd.DataFrame(df['smiles'], columns=['smiles'])
    density = pd.DataFrame(df['density_Kg/m3'], columns=['density_Kg/m3'])
    features = df.drop(['smiles', 'density_Kg/m3'],1)

    return smi, density, features


def load_xyz_polarizability():
    """Load and return xyz files and polarizability (Bohr^3).
    The xyz coordinates of small organic molecules are optimized with BP86/def2svp level of theory.
    Polarizability of the molecules are also calcualted in the same level of thoery.

    =================   ======================
    rows                                    50
    Columns                                  1
    header                      polarizability
    molecules rep.                         xyz
    Features                                 0
    Returns             1 dataframe and 1 dict
    =================   ======================

    Returns
    -------
    molecules : list
        The list of molecule objects with xyz coordinates.

    pol : pandas dataframe
        The polarizability of each molecule as a column of dataframe.

    Examples
    --------
    >>> from chemml.datasets import load_xyz_polarizability
    >>> molecules, polarizabilities = load_xyz_polarizability()
    >>> print(len(molecules))
    50
    >>> print(polarizabilities.shape)
    (50, 1)
    """
    DATA_PATH = pkg_resources.resource_filename('chemml', os.path.join('datasets','data','organic_xyz'))
    # from chemml.initialization import XYZreader
    # reader = XYZreader(path_pattern=['[1-9]_opt.xyz', '[1-9][0-9]_opt.xyz'],
    #                    path_root=DATA_PATH,
    #                    reader='manual',
    #                    skip_lines=[2, 0])
    # molecules = reader.read()
    molecules = []
    for i in range(1,51):
        molecule = Molecule(os.path.join(DATA_PATH,"%i_opt.xyz"%i), "xyz")
        molecule.to_smiles()
        molecule.pybel_molecule = None
        molecules.append(molecule)

    df = pd.read_csv(os.path.join(DATA_PATH,'pol.csv'))
    return molecules, df


def load_comp_energy():
    """Load and return composition entries and formation energies (eV).
    From Magpie https://bitbucket.org/wolverton/magpie

    =================   ======================
    rows                                   630
    header                    formation_energy
    molecules rep.                 composition
    Features                                 0
    Returns             1 dataframe and 1 list
    =================   ======================

    Returns
    -------
    entries : list
        The list of composition entries from CompositionEntry class.

    energy : pandas dataframe
        The formation energy for each composition.

    Examples
    --------
    >>> from chemml.datasets import load_comp_energy
    >>> entries, df = load_comp_energy()
    >>> print(df.shape)
    (630, 1)
    """
    DATA_PATH = pkg_resources.resource_filename('chemml', os.path.join('datasets', 'data', 'magpie_python_test', 'small_set_comp.txt'))
    TARGET_PATH = pkg_resources.resource_filename('chemml', os.path.join('datasets', 'data', 'magpie_python_test', 'small_set_delta_e.txt'))
    from chemml.chem.magpie_python import CompositionEntry
    entries = CompositionEntry.import_composition_list(DATA_PATH)
    df = pd.read_csv(TARGET_PATH,header=None)
    df.columns = ['formation_energy']
    return entries, df


def load_crystal_structures():
    """Load and return crystal structure entries.
    From Magpie https://bitbucket.org/wolverton/magpie

    =================   ======================
    length                                  18
    header                    formation_energy
    molecules rep.                 composition
    Features                                 0
    Returns                             1 list
    =================   ======================

    Returns
    -------
    entries : list
        The list of crystal structure entries from CrystalStructureEntry class.

    Examples
    --------
    >>> from chemml.datasets import load_crystal_structures
    >>> entries = load_crystal_structures()
    >>> print(len(entries))
    18
    """
    DATA_PATH = pkg_resources.resource_filename('chemml', os.path.join('datasets', 'data', 'magpie_python_test'))
    from chemml.chem.magpie_python import CrystalStructureEntry
    entries = CrystalStructureEntry.import_structures_list(DATA_PATH)
    return entries


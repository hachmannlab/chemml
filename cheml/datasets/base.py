import pkg_resources
import pandas as pd


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
    data : two dataframes
        first dataframe for SMILES representation of molecules, shape: (500,1)
        second dataframe for HOMO energies of the molecules (eV), shape: (500,1)

    Examples
    --------
    Let's say you want to know the header and shape of the dataframe.
    >>> from cheml.datasets import load_cep_homo
    >>> smi, homo  = load_cep_homo()
    >>> print list(smi.columns)
    ['smiles']
    >>> print homo.shape
    (500, 1)
    """
    DATA_PATH = pkg_resources.resource_filename('cheml', 'datasets/data/cep_homo.csv')
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
    data : three dataframes
        first dataframe for SMILES representation of molecules, shape: (500,1)
        second dataframe for the density of molecules (Kg/m3), shape: (500,1)
        third dataframe for the molecular descriptors of molecules, shape: (500,1)

    Examples
    --------
    Let's say you want to know the header and shape of the dataframe.
    >>> from cheml.datasets import load_organic_density
    >>> smi, density, features = load_organic_density()
    >>> print list(smi.columns)
    ['smiles']
    >>> print features.shape
    (500, 200)
    """
    DATA_PATH = pkg_resources.resource_filename('cheml', 'datasets/data/moldescriptor_density_smiles.csv')
    df = pd.read_csv(DATA_PATH)

    smi = pd.DataFrame(df['smiles'], columns=['smiles'])
    density = pd.DataFrame(df['density_Kg/m3'], columns=['density_Kg/m3'])
    features = df.drop(['smiles', 'density_Kg/m3'],1)

    return smi, density, features


def load_xyz_polarizability():
    """Load and return xyz files and polarizability (Bohr^3)
    The xyz coordinates of small organic molecules are optimized with BP86/def2svp level of theory.
    Polarizability of the molecules are also calcualted in the same level of thoery.

    Returns
    -------
    dictionary
        the xyz coordinates and atomic numbers of each atom of the molecule will be in a numpy array.

    """
    DATA_PATH = pkg_resources.resource_filename('cheml', 'datasets/data/xyz')



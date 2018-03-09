import pkg_resources
import pandas as pd


def load_cep_homo(return_X_y=False):
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
    =================   ==============

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns X (molecules) and y (property) separately.
        Both X and y are in pandas dataframe format.

    Returns
    -------
    data : pandas dataframe
        one or two dataframes (depends on parameter return_X_y value)

    Examples
    --------
    Let's say you want to know the header and shape of the dataframe.
    >>> from cheml.datasets import load_cep_homo
    >>> df = load_cep_homo()
    >>> list(df.columns)
    ['smiles', 'homo_eV']
    >>> df.shape
    (500, 2)
    """
    DATA_PATH = pkg_resources.resource_filename('cheml', 'datasets/data/cep_homo.csv')
    df = pd.read_csv(DATA_PATH)

    if return_X_y:
        return pd.DataFrame(df['smiles'],columns=['smiles']), pd.DataFrame(df['homo_eV'],columns=['homo_eV'])

    return df

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



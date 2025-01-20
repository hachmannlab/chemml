'''
FOSS Molecular Descriptor Generators
This file contains implementations of classes for generating molecular descriptors using free and open-source (FOSS) libraries. Currently, it includes:

    Mordred Descriptors: A wrapper class for the Mordred descriptor library, which is an open-source alternative to commercial descriptor generators like Dragon.
    RDKit Descriptors: A class for generating molecular descriptors using the RDKit library, which provides a comprehensive set of cheminformatics functionalities.

The file is designed to be extensible, allowing for easy addition of other FOSS molecular descriptor packages. Each descriptor generator is implemented as a separate class, following a similar structure:

    Initialization with customizable parameters
    A represent method that takes a list of molecules (as SMILES or molecular objects) and returns a pandas DataFrame of calculated descriptors

Key features of the current implementation:

    Flexible input handling (SMILES strings or molecular objects)
    Options for data cleaning (dropping NA values, removing highly correlated descriptors)
    Easy-to-use interface consistent across different descriptor packages

To add a new descriptor package:

    Create a new class for the descriptor package
    Implement an __init__ method with relevant parameters
    Implement a represent method that calculates descriptors and returns a DataFrame
    Ensure the new class follows a similar interface to existing classes for consistency
    Update __init__.py with the new class name 
'''

import pandas as pd
import numpy as np
from mordred import Calculator, descriptors
from rdkit.Chem import Descriptors, MolFromSmiles

from chemml.chem import Molecule

class RDKDesc(object):
    """
    A class for generating molecular descriptors using RDKit.

    This class provides functionality to calculate a wide range of molecular descriptors
    for chemical compounds using the RDKit library.

    Attributes:
        descriptor_list (list): A list of available descriptor names.

    Methods:
        represent(mol_list, output_directory='./', dropna=True, remove_corr=False):
            Generates molecular descriptors for a list of molecules.

    Examples:
        >>> from rdkit import Chem
        >>> rdkit_desc = RDKitDescriptors()
        >>> smiles_list = ['CC', 'CCO', 'CCCO']
        >>> df = rdkit_desc.represent(smiles_list)
    """

    def __init__(self):
        self.descriptor_list = [x[0] for x in Descriptors._descList]

    def represent(self, mol_list, output_directory='./', dropna=True, remove_corr=False):
        """
        Generate RDKit molecular descriptors for a list of molecules.

        This method calculates RDKit descriptors for the provided molecules and returns them as a pandas DataFrame.

        Parameters:
        -----------
        mol_list : list or str
            Input molecules. Can be one of the following:
            - List of SMILES strings
            - Single SMILES string
        output_directory : str, optional
            Directory to save generated descriptors. Default is './'.
        dropna : bool, optional
            If True, drop columns with NaN values. Default is True.
        remove_corr : bool, optional
            If True, remove highly correlated descriptors (correlation > 0.95). Default is False.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame containing the calculated RDKit descriptors. Each row represents a molecule,
            and each column represents a descriptor. The 'SMILES' column is added to identify the molecules.

        Raises:
        -------
        ValueError
            If the input SMILES strings are not in a valid format.
        """
        if isinstance(mol_list, list):
            if isinstance(mol_list[0], str):
                try:
                    smi_list = mol_list
                    mol_list = [MolFromSmiles(m) for m in mol_list]
                except Exception as e:
                    print(e,' SMILES not in valid format')
            if isinstance(mol_list[0], Molecule):
                try:
                    smi_list = [m.smiles for m in mol_list]
                    mol_list = [m.rdkit_molecule for m in mol_list]
                except Exception as e:
                    print(str(e)+' Make sure input is a list of ChemML Molecule objects')
        else:
            if isinstance(mol_list, Molecule):

                smi_list = [mol_list.smiles]
                mol_list = [mol_list.rdkit_molecule]

            if isinstance(mol_list, str):
                smi_list = [mol_list]
                try:
                    mol_list = [MolFromSmiles(mol_list)]
                except Exception as e:
                    print(e,' SMILES not in valid format')

        desc_data = []
        for mol in mol_list:
            mol_desc = {}
            for desc_name in self.descriptor_list:
                mol_desc[desc_name] = getattr(Descriptors, desc_name)(mol)
            desc_data.append(mol_desc)

        df = pd.DataFrame(desc_data)
        df['SMILES'] = smi_list

        if dropna:
            df.dropna(axis=1, inplace=True)

        if remove_corr:
            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            df = df.drop(columns=to_drop)

        return df

class Mordred(object):
    '''
    A wrapper class for generating Mordred molecular descriptors.

    This class provides an interface to create Mordred molecular descriptors, which are an open-source
    alternative to Dragon descriptors. It allows for the calculation of various molecular properties
    and features based on chemical structures.

    Attributes:
    ----------
        calc (Calculator): A Mordred Calculator object for descriptor generation.

    Parameters:
    ----------
        ignore_3D (bool): If True, ignore 3D descriptor generation. Default is True.
        selected_descriptors (bool): If True, generate only selected descriptors. Default is False.

    Methods:
    -------
        represent(mol_list, output_directory='./', dropna=True, quiet=True, remove_corr=False):
            Generates molecular descriptors for a list of molecules.

    Notes:
    -----
        - Requires installation of Mordred descriptors as described in:
          https://github.com/mordred-descriptor/mordred
        - By default, all available descriptors are generated.

    Examples:
    --------
        >>> import pandas as pd
        >>> from chemml.chem import Mordred
        >>> mord = Mordred()
        >>> df = mord.represent(mol_list)

    '''
    def __init__(self, ignore_3D=True, selected_descriptors=False):
        try:
            if selected_descriptors:
                # TODO: Construct section for selected descriptor generation
                # Default is to generate all descriptors available.
                pass
            else:
                self.calc = Calculator(descriptors, ignore_3D=ignore_3D)
        except ModuleNotFoundError as m:
            print(m,': Are you sure Mordred is installed in the environment?')

    def represent(self, mol_list, output_directory='./', dropna=True, quiet=True, remove_corr=False):
        '''
        Generate Mordred molecular descriptors for a list of molecules.

        This method calculates Mordred descriptors for the provided molecules and returns them as a pandas DataFrame.
        It can handle input in the form of SMILES strings or ChemML Molecule objects.

        Parameters:
        ----------
        mol_list : list, str, or Molecule
            Input molecules. Can be one of the following:
            - List of SMILES strings
            - List of ChemML Molecule objects
            - Single SMILES string
            - Single ChemML Molecule object
        output_directory : str, optional
            Directory to save generated descriptors. Default is './'.
        dropna : bool, optional
            If True, drop rows with NaN values. Default is True.
        quiet : bool, optional
            If True, suppress Mordred's output messages. Default is True.
        remove_corr : bool, optional
            If True, remove highly correlated descriptors (correlation > 0.95). Default is False.

        Returns:
        -------
        pandas.DataFrame
            A DataFrame containing the calculated Mordred descriptors. Each row represents a molecule,
            and each column represents a descriptor. The 'SMILES' column is added to identify the molecules.

        Raises:
        ------
        Exception
            If the input SMILES strings are not in a valid format or if there's an issue with ChemML Molecule objects.

        Notes:
        ------
        - The method handles various input formats flexibly, converting them to RDKit molecule objects internally.
        - Infinite values in descriptors are replaced with NaN.
        - If remove_corr is True, highly correlated descriptors (correlation > 0.95) are removed to reduce redundancy.

        Examples:
        --------
        >>> mord = Mordred()
        >>> smiles_list = ['CC', 'CCO', 'CCCO']
        >>> df = mord.represent(smiles_list, dropna=True, remove_corr=True)
        >>> print(df.shape)
        '''
        if isinstance(mol_list, list):
            if isinstance(mol_list[0], str):
                try:
                    smi_list = mol_list
                    mol_list = [MolFromSmiles(m) for m in mol_list]
                except Exception as e:
                    print(e,' SMILES not in valid format')
            if isinstance(mol_list[0], Molecule):
                try:
                    smi_list = [m.smiles for m in mol_list]
                    mol_list = [m.rdkit_molecule for m in mol_list]
                except Exception as e:
                    print(str(e)+' Make sure input is a list of ChemML Molecule objects')
        else:
            if isinstance(mol_list, Molecule):

                smi_list = [mol_list.smiles]
                mol_list = [mol_list.rdkit_molecule]

            if isinstance(mol_list, str):
                smi_list = [mol_list]
                try:
                    mol_list = [MolFromSmiles(mol_list)]
                except Exception as e:
                    print(e,' SMILES not in valid format')
        
        pand = self.calc.pandas(mol_list, quiet=quiet)
        pand = pand.select_dtypes([np.number]).replace([np.inf, -np.inf], np.nan)

        if dropna:
            pand.dropna(inplace=True)
        
        if remove_corr:
            # Generate matrix of correlation values
            corr_matrix = pand.corr().abs()
            # Keep only upper triangle of values, since the correlation matrix is mirrored around the diagonal
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            # Find columns that are highly correlated
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            pand = pand.drop(columns=to_drop)


        pand['SMILES'] = smi_list
        return pand



class PadelDesc:
    """
    A class for generating molecular descriptors using PaDEL-Descriptor via PaDELPy.

    This class provides functionality to calculate a wide range of molecular descriptors
    for chemical compounds using the PaDEL-Descriptor software through its Python wrapper.

    Methods:
        represent(mol_list, output_directory='./', dropna=True, remove_corr=False):
            Generates molecular descriptors for a list of molecules.

    Examples:
        >>> padel_desc = PadelDesc()
        >>> smiles_list = ['CC', 'CCO', 'CCCO']
        >>> df = padel_desc.represent(smiles_list)
    """

    def __init__(self):
        pass

    def represent(self, mol_list, output_directory='./', dropna=True, remove_corr=False):
        """
        Generate PaDEL molecular descriptors for a list of molecules.

        This method calculates PaDEL descriptors for the provided molecules and returns them as a pandas DataFrame.

        Parameters:
        -----------
        mol_list : list or str
            Input molecules. Can be one of the following:
            - List of SMILES strings
            - Single SMILES string
        output_directory : str, optional
            Directory to save generated descriptors. Default is './'.
        dropna : bool, optional
            If True, drop columns with NaN values. Default is True.
        remove_corr : bool, optional
            If True, remove highly correlated descriptors (correlation > 0.95). Default is False.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame containing the calculated PaDEL descriptors. Each row represents a molecule,
            and each column represents a descriptor. The 'SMILES' column is added to identify the molecules.

        Raises:
        -------
        ValueError
            If the input SMILES strings are not in a valid format.
        """
        from padelpy import from_smiles

        if isinstance(mol_list, list):
            if isinstance(mol_list[0], str):
                try:
                    smi_list = mol_list
                    mol_list = [MolFromSmiles(m) for m in mol_list]
                except Exception as e:
                    print(e,' SMILES not in valid format')
            if isinstance(mol_list[0], Molecule):
                try:
                    smi_list = [m.smiles for m in mol_list]
                    mol_list = [m.rdkit_molecule for m in mol_list]
                except Exception as e:
                    print(str(e)+' Make sure input is a list of ChemML Molecule objects')
        else:
            if isinstance(mol_list, Molecule):

                smi_list = [mol_list.smiles]
                mol_list = [mol_list.rdkit_molecule]

            if isinstance(mol_list, str):
                smi_list = [mol_list]
                try:
                    mol_list = [MolFromSmiles(mol_list)]
                except Exception as e:
                    print(e,' SMILES not in valid format')

        # Calculate descriptors
        descriptors = from_smiles(smi_list)
        
        # Convert to DataFrame
        df = pd.DataFrame(descriptors)
        df['SMILES'] = smi_list

        if dropna:
            df.dropna(axis=1, inplace=True)

        if remove_corr:
            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            df = df.drop(columns=to_drop)

        return df
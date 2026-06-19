'''
FOSS Molecular Descriptor Generators
This file contains implementations of classes for generating molecular descriptors using free and open-source (FOSS) libraries. Currently, it includes:

    Mordred Descriptors: A wrapper class for the Mordred descriptor library, which is an open-source alternative to commercial descriptor generators like Dragon.
    RDKit Descriptors: A class for generating molecular descriptors using the RDKit library, which provides a comprehensive set of cheminformatics functionalities.
    PaDEL Descriptors: A class for generating molecular descriptors using the PaDEL-Descriptor software via its Python wrapper, PaDELPy.

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

import warnings
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from rdkit.Chem import Descriptors, MolFromSmiles
from tqdm import tqdm
from chemml.chem import Molecule

def normalize_input(mol_list, quiet=True, force_molecule=False):
        """
        Normalize input to a consistent format for further processing.
        
        Parameters:
        -----------
        mol_list : list, str, or Molecule
            Input molecules. Can be one of the following:
            - List of SMILES strings
            - Single SMILES string
            - List of ChemML Molecule objects
            - Single ChemML Molecule object
        quiet : bool, optional; default True
            If True, suppress progress bars and warnings.
        force_molecule : bool, optional, default False
            If True, create ChemML Molecule objects from mol_list.
        
        Returns:
        --------
        tuple of lists
            (smiles_list, rdkit_mol_list, molecule_obj_list) if force_molecule is True, otherwise (smiles_list, rdkit_mol_list)
        
        Raises:
        -------
        ValueError
            If the input is not in a valid format or if there are issues with SMILES parsing or Molecule object creation.
        """
        if isinstance(mol_list, list):
            items = mol_list
        else:
            items = [mol_list]

        if len(items) == 0:
            raise ValueError('The input molecule list is empty.')

        smiles_out = []
        rdkit_out = []
        molecule_out = []

        if isinstance(items[0], Molecule):
            iterator = tqdm(items, desc='Normalizing Molecule input', disable=quiet)
            for mol_obj in iterator:
                if not isinstance(mol_obj, Molecule):
                    raise ValueError('Mixed input types are not supported in the same list.')
                if mol_obj.smiles is None:
                    mol_obj.to_smiles()

                smiles_out.append(mol_obj.smiles)
                rdkit_out.append(mol_obj.rdkit_molecule)
                molecule_out.append(mol_obj)
        elif isinstance(items[0], str):
            iterator = tqdm(items, desc='Converting SMILES input', disable=quiet)
            for smi in iterator:
                if not isinstance(smi, str):
                    raise ValueError('Mixed input types are not supported in the same list.')

                if force_molecule:
                    try:
                        mol_obj = Molecule(smi, 'smiles')
                        smiles_out.append(mol_obj.smiles)
                        rdkit_out.append(mol_obj.rdkit_molecule)
                        molecule_out.append(mol_obj)
                    except Exception as exc:
                        warnings.warn(f'Skipping invalid SMILES {smi} due to: {str(exc)}')
                else:
                    mol = MolFromSmiles(smi)
                    if mol is None:
                        warnings.warn(f'Skipping invalid SMILES {smi} due to RDKit parsing failure.')
                        continue
                    smiles_out.append(smi)
                    rdkit_out.append(mol)
        else:
            raise ValueError('Input must be a SMILES string, list of SMILES, Molecule, or list of Molecule objects.')

        if force_molecule:
            return smiles_out, rdkit_out, molecule_out
        return smiles_out, rdkit_out



class RDKDesc(object):
    """
    A class for generating molecular descriptors using RDKit.

    This class provides functionality to calculate a wide range of molecular descriptors
    for chemical compounds using the RDKit library.

    Attributes:
        descriptor_list (list): A list of available descriptor names.

    Methods:
        represent(mol_list, dropna=True, remove_corr=False):
            Generates molecular descriptors for a list of molecules.

    Examples:
        >>> from chemml.chem import RDKDesc
        >>> rdkit_desc = RDKDesc()
        >>> smiles_list = ['CC', 'CCO', 'CCCO']
        >>> df = rdkit_desc.represent(smiles_list)
    """

    def __init__(self):
        self.descriptor_list = [x[0] for x in Descriptors._descList]

    def represent(self, mol_list, dropna=True, remove_corr=False, n_jobs=1):
        """
        Generate RDKit molecular descriptors for a list of molecules.

        This method calculates RDKit descriptors for the provided molecules and returns them as a pandas DataFrame.

        Parameters:
        -----------
        mol_list : list or str
            Input molecules. Can be one of the following:
            - List of SMILES strings
            - Single SMILES string
            - List of ChemML Molecule objects
            - Single ChemML Molecule object

        dropna : bool, optional
            If True, drop columns with NaN values. Default is True.
        remove_corr : bool, optional
            If True, remove highly correlated descriptors (correlation > 0.95). Default is False.
            Warning: Only use this option if you have >100 molecules, as correlation calculations can be unreliable with small datasets.
        n_jobs : int, optional
            Number of parallel jobs to run for descriptor calculation. Default is 1 (no parallelization).
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
        smi_list, mol_list = normalize_input(mol_list)

        desc_data = []
        if n_jobs == 1:
            for mol in tqdm(mol_list, desc='Calculating RDKit descriptors'):
                mol_desc = {}
                for desc_name in self.descriptor_list:
                    mol_desc[desc_name] = getattr(Descriptors, desc_name)(mol)
                desc_data.append(mol_desc)
        else:
            
            def calculate_descriptors(mol, descriptor_list):
                mol_desc = {}
                for desc_name in descriptor_list:
                    mol_desc[desc_name] = getattr(Descriptors, desc_name)(mol)
                return mol_desc
            
            desc_data = Parallel(n_jobs=n_jobs)(
            delayed(calculate_descriptors)(mol, self.descriptor_list) for mol in tqdm(mol_list, desc='Calculating RDKit descriptors')
            )

        df = pd.DataFrame(desc_data)
        df['SMILES'] = smi_list

        if dropna:
            df.dropna(axis=1, inplace=True)

        if remove_corr and len(mol_list) > 100:
            corr_matrix = df.drop(columns=['SMILES']).corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            df = df.drop(columns=to_drop)
        elif len(mol_list) <= 100 and remove_corr:
            warnings.warn('Correlation calculations can be unreliable with small datasets. Use this option only if you have >100 molecules.')
        
        
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
        selected_descriptors (bool): If True, generate only selected descriptors. Default is False. Currently, this option is not implemented and all descriptors are generated by default.

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
        from mordred import Calculator, descriptors
        self.ignore_3D = ignore_3D
        try:
            if selected_descriptors:
                # TODO: Construct section for selected descriptor generation
                # Default is to generate all descriptors available.
                pass
            else:
                self.calc = Calculator(descriptors, ignore_3D=ignore_3D)
        except ModuleNotFoundError as m:
            print(m,': Are you sure Mordred is installed in the environment?')

    

    def _optimize_3d_serial(self, molecule_list, optimizer='MMFF', quiet=True, **kwargs):
        if optimizer not in ['MMFF', 'UFF']:
            raise ValueError("The optimizer must be either 'MMFF' or 'UFF' for RDKit 3D optimization.")
        force_optimize = kwargs.pop('force_optimize', False)
        optimized_smis = []
        optimized_mols = []

        iterator = tqdm(molecule_list, desc='Optimizing 3D geometries', disable=quiet)
        for mol_obj in iterator:
            if mol_obj.xyz is not None and not force_optimize:
                iterator.desc = '3D geometry already exists, loading geometry'
                optimized_smis.append(mol_obj.smiles)
                optimized_mols.append(mol_obj.rdkit_molecule)
                continue
            try:
                # Uses ChemML's internal RDKit optimizer
                mol_obj.to_xyz(optimizer=optimizer, **kwargs)
                optimized_smis.append(mol_obj.smiles)
                optimized_mols.append(mol_obj.rdkit_molecule)
            except Exception as exc:
                warnings.warn('Skipping molecule %s due to 3D optimization failure: %s' % (mol_obj.smiles, str(exc)))

        return optimized_smis, optimized_mols

    def represent(self, mol_list, quiet=True, remove_corr=False, optimizer='MMFF', **kwargs):
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
            - List of RDKit Mol objects
            - Single SMILES string
            - Single ChemML Molecule object
            - Single RDKit Mol object
        quiet : bool, optional
            If True, suppress Mordred's output messages. Default is True.
        remove_corr : bool, optional
            If True, remove highly correlated descriptors (correlation > 0.95). Default is False.
            Warning: Only use this option if you have >100 molecules, as correlation calculations can be unreliable with small datasets.

        Returns:
        -------
        pandas.DataFrame
            A DataFrame containing the calculated Mordred descriptors. Each row represents a molecule,
            and each column represents a descriptor. The 'SMILES' column is added to identify the molecules.
            We automatically drop columns with all NaN values, but row imputation is left to the user.

        Raises:
        ------
        Exception
            If the input SMILES strings are not in a valid format or if there's an issue with ChemML Molecule objects.

        Examples:
        --------
        >>> mord = Mordred()
        >>> smiles_list = ['CC', 'CCO', 'CCCO']
        >>> df = mord.represent(smiles_list, remove_corr=True)
        >>> print(df.shape)
        '''

        if self.ignore_3D:
            smi_list, mol_list = normalize_input(mol_list, quiet=quiet, force_molecule=False)

        else:
            smi_list, _, mol_list = normalize_input(mol_list, quiet=quiet, force_molecule=True)
            force_optimize = kwargs.pop('force_optimize', False)
            smi_list, mol_list  = self._optimize_3d_serial(mol_list, optimizer=optimizer, quiet=quiet, force_optimize=force_optimize)
        
        pand = self.calc.pandas(mol_list, quiet=quiet)
        pand = pand.select_dtypes([np.number]).replace([np.inf, -np.inf], np.nan)
        pand = pand.dropna(axis=1, how='all')

        pand['SMILES'] = smi_list

        if remove_corr and len(mol_list) > 1000:
            # Generate matrix of correlation values
            corr_matrix = pand.drop(columns=['SMILES']).corr().abs()
            # Keep only upper triangle of values, since the correlation matrix is mirrored around the diagonal
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            # Find columns that are highly correlated
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            pand = pand.drop(columns=to_drop)
        elif len(mol_list) <= 1000 and remove_corr:
            warnings.warn('Correlation calculations can be unreliable with small datasets. Use this option only if you have >1000 molecules.')
            
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

    def represent(self, mol_list, dropna=True, remove_corr=False):
        """
        Generate PaDEL molecular descriptors for a list of molecules.

        This method calculates PaDEL descriptors for the provided molecules and returns them as a pandas DataFrame.
        Note: Requires installation of PaDEL-Descriptor and its Python wrapper, PaDELPy.

        Parameters:
        -----------
        mol_list : list or str
            Input molecules. Can be one of the following:
            - List of SMILES strings
            - Single SMILES string
            - List of ChemML Molecule objects
            - Single ChemML Molecule object
            - List of RDKit Mol objects
            - Single RDKit Mol object
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

        smi_list, _ = normalize_input(mol_list, quiet=True, force_molecule=False)

        # Calculate descriptors
        descriptors = from_smiles(smi_list)
        
        # Convert to DataFrame
        df = pd.DataFrame(descriptors)
        df['SMILES'] = smi_list

        if dropna:
            df.dropna(axis=1, inplace=True)

        if remove_corr:
            corr_matrix = df.drop(columns=['SMILES']).corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            df = df.drop(columns=to_drop)

        return df
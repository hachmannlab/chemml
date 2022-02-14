from builtins import range
import warnings
import os
import time
import pandas as pd
from lxml import objectify, etree
import numpy as np
from chemml.utils import std_datetime_str
from chemml.utils import bool_formatter
from chemml.utils import tot_exec_time_str

class Dragon(object):
    """
    An interface to Dragon 6 and 7 chemoinformatics software. Dragon is a commercial software and
    you should provide

    Parameters
    ----------
    version : int, optional (default=7)
        The version of available Dragon on the user's system. (available versions: 6 or 7)

    Weights : list, optional (default=["Mass","VdWVolume","Electronegativity","Polarizability","Ionization","I-State"])
        A list of weights to be used

    blocks : list, optional (default = list(range(1,31)))
        A list of integers as descriptor blocks' id. There are totally 29 and 30 blocks available in
        version 6 and 7, respectively.
        This module is not atimed to cherry pick descriptors in each block.
        For doing so, please use Script Wizard in Drgon GUI.

    external : boolean, optional (default=False)
        If True, include external variables at the end of each saved file.

    Notes
    -----
        The documentation for the rest of parameters can be found in the following links:
            - http://www.talete.mi.it/help/dragon_help/index.html
            - https://chm.kode-solutions.net/products_dragon_tutorial.php


    Examples
    --------
        >>> import pandas as pd
        >>> from chemml.chem import Dragon
        >>> drg = Dragon()
        >>> df = drg.represent(mol_list, output_directory='./', dropna=False)

    """

    def __init__(self,
                 CheckUpdates=True,
                 SaveLayout=True,
                 PreserveTemporaryProjects=True,
                 ShowWorksheet=False,
                 Decimal_Separator=".",
                 Missing_String="NaN",
                 DefaultMolFormat="1",
                 HelpBrowser="/usr/bin/xdg-open",
                 RejectUnusualValence=False,
                 Add2DHydrogens=False,
                 MaxSRforAllCircuit="19",
                 MaxSR="35",
                 MaxSRDetour="30",
                 MaxAtomWalkPath="2000",
                 LogPathWalk=True,
                 LogEdge=True,
                 Weights=("Mass", "VdWVolume", "Electronegativity", "Polarizability", "Ionization",
                          "I-State"),
                 SaveOnlyData=False,
                 SaveLabelsOnSeparateFile=False,
                 SaveFormatBlock="%b-%n.txt",
                 SaveFormatSubBlock="%b-%s-%n-%m.txt",
                 SaveExcludeMisVal=False,
                 SaveExcludeAllMisVal=False,
                 SaveExcludeConst=False,
                 SaveExcludeNearConst=False,
                 SaveExcludeStdDev=False,
                 SaveStdDevThreshold="0.0001",
                 SaveExcludeCorrelated=False,
                 SaveCorrThreshold="0.95",
                 SaveExclusionOptionsToVariables=False,
                 SaveExcludeMisMolecules=False,
                 SaveExcludeRejectedMolecules=False,
                 blocks=list(range(1, 31)),
                 SaveStdOut=False,
                 SaveProject=False,
                 SaveProjectFile="Dragon_project.drp",
                 SaveFile=True,
                 SaveType="singlefile",
                 SaveFilePath="Dragon_descriptors.txt",
                 logMode="file",
                 logFile="Dragon_log.txt",
                 external=False,
                 fileName=None,
                 delimiter=",",
                 consecutiveDelimiter=False,
                 MissingValue="NaN",
                 RejectDisconnectedStrucuture=False,
                 RetainBiggestFragment=False,
                 DisconnectedCalculationOption="0",
                 RoundCoordinates=True,
                 RoundWeights=True,
                 RoundDescriptorValues=True,
                 knimemode=False):
        
        self.version = 7
        self.CheckUpdates = CheckUpdates
        self.PreserveTemporaryProjects = PreserveTemporaryProjects
        self.SaveLayout = SaveLayout
        self.ShowWorksheet = ShowWorksheet
        self.Decimal_Separator = Decimal_Separator
        self.Missing_String = Missing_String
        self.DefaultMolFormat = DefaultMolFormat
        self.HelpBrowser = HelpBrowser
        self.RejectUnusualValence = RejectUnusualValence
        self.Add2DHydrogens = Add2DHydrogens
        self.MaxSRforAllCircuit = MaxSRforAllCircuit
        self.MaxSR = MaxSR
        self.MaxSRDetour = MaxSRDetour
        self.MaxAtomWalkPath = MaxAtomWalkPath
        self.LogPathWalk = LogPathWalk
        self.LogEdge = LogEdge
        self.Weights = Weights
        self.SaveOnlyData = SaveOnlyData
        self.SaveLabelsOnSeparateFile = SaveLabelsOnSeparateFile
        self.SaveFormatBlock = SaveFormatBlock
        self.SaveFormatSubBlock = SaveFormatSubBlock
        self.SaveExcludeMisVal = SaveExcludeMisVal
        self.SaveExcludeAllMisVal = SaveExcludeAllMisVal
        self.SaveExcludeConst = SaveExcludeConst
        self.SaveExcludeNearConst = SaveExcludeNearConst
        self.SaveExcludeStdDev = SaveExcludeStdDev
        self.SaveStdDevThreshold = SaveStdDevThreshold
        self.SaveExcludeCorrelated = SaveExcludeCorrelated
        self.SaveCorrThreshold = SaveCorrThreshold
        self.SaveExclusionOptionsToVariables = SaveExclusionOptionsToVariables
        self.SaveExcludeMisMolecules = SaveExcludeMisMolecules
        self.SaveExcludeRejectedMolecules = SaveExcludeRejectedMolecules
        self.blocks = blocks
        self.SaveStdOut = SaveStdOut
        self.SaveProject = SaveProject
        self.SaveProjectFile = SaveProjectFile
        self.SaveFile = SaveFile
        self.SaveType = SaveType
        self.SaveFilePath = SaveFilePath
        self.logMode = logMode
        self.logFile = logFile
        self.external = external
        self.fileName = fileName
        self.delimiter = delimiter
        self.consecutiveDelimiter = consecutiveDelimiter
        self.MissingValue = MissingValue
        self.RejectDisconnectedStrucuture = RejectDisconnectedStrucuture
        self.RetainBiggestFragment = RetainBiggestFragment
        self.DisconnectedCalculationOption = DisconnectedCalculationOption
        self.RoundCoordinates = RoundCoordinates
        self.RoundWeights = RoundWeights
        self.RoundDescriptorValues = RoundDescriptorValues
        self.knimemode = knimemode

    def _script_wizard(self, input_mol_list, input_mol_type):
        """
        The script_wizard is designed to build a Dragon script file. The name and
        the functionality of this function is the same as available Script wizard
        in the Dragon Graphic User Interface.
        Note: All reported nodes are mandatory, except the <EXTERNAL> tag
        Note: Script for version 7 doesn't support fingerprints block

        Parameters
        ----------
        input_mol_list : list
            if SMILES, list contains only 1 element -- path to one SMILES file
            if mol2, list of '.mol2' file paths

        input_mol_type: str
            Input molecule type: either 'mol2' or 'smiles'
        
        output_directory : string, optional (default = './')
            the path to the working directory to store output files.

        """
        
        self.dragon = objectify.Element(
            "DRAGON",
            version="%i.0.0" % self.version,
            description="Dragon7 - FP1 - MD5270",
            script_version="1",
            generation_date=std_datetime_str('date').replace('-', '/'))

        OPTIONS = objectify.SubElement(self.dragon, "OPTIONS")
        OPTIONS.append(
            objectify.Element("CheckUpdates", value=bool_formatter(self.CheckUpdates)))
        OPTIONS.append(
            objectify.Element(
                "PreserveTemporaryProjects",
                value=bool_formatter(self.PreserveTemporaryProjects)))
        OPTIONS.append(objectify.Element("SaveLayout", value=bool_formatter(self.SaveLayout)))
        #                 OPTIONS.append(objectify.Element("ShowWorksheet", value = bool_formatter(self.ShowWorksheet)))
        OPTIONS.append(objectify.Element("Decimal_Separator", value=self.Decimal_Separator))
        if self.Missing_String == "NaN": self.Missing_String = "na"
        OPTIONS.append(objectify.Element("Missing_String", value=self.Missing_String))
        OPTIONS.append(objectify.Element("DefaultMolFormat", value=self.DefaultMolFormat))
        #                 OPTIONS.append(objectify.Element("HelpBrowser", value = self.HelpBrowser))
        OPTIONS.append(
            objectify.Element(
                "RejectDisconnectedStrucuture",
                value=bool_formatter(self.RejectDisconnectedStrucuture)))
        OPTIONS.append(
            objectify.Element(
                "RetainBiggestFragment", value=bool_formatter(self.RetainBiggestFragment)))
        OPTIONS.append(
            objectify.Element(
                "RejectUnusualValence", value=bool_formatter(self.RejectUnusualValence)))
        OPTIONS.append(
            objectify.Element("Add2DHydrogens", value=bool_formatter(self.Add2DHydrogens)))
        OPTIONS.append(
            objectify.Element(
                "DisconnectedCalculationOption", value=self.DisconnectedCalculationOption))
        OPTIONS.append(objectify.Element("MaxSRforAllCircuit", value=self.MaxSRforAllCircuit))
        #                 OPTIONS.appendm(objectify.Element("MaxSR", value = self.MaxSR))
        OPTIONS.append(objectify.Element("MaxSRDetour", value=self.MaxSRDetour))
        OPTIONS.append(objectify.Element("MaxAtomWalkPath", value=self.MaxAtomWalkPath))
        OPTIONS.append(
            objectify.Element("RoundCoordinates", value=bool_formatter(self.RoundCoordinates)))
        OPTIONS.append(
            objectify.Element("RoundWeights", value=bool_formatter(self.RoundWeights)))
        OPTIONS.append(objectify.Element("LogPathWalk", value=bool_formatter(self.LogPathWalk)))
        OPTIONS.append(objectify.Element("LogEdge", value=bool_formatter(self.LogEdge)))
        Weights = objectify.SubElement(OPTIONS, "Weights")
        for weight in self.Weights:
            if weight not in [
                    "Mass", "VdWVolume", "Electronegativity", "Polarizability", "Ionization",
                    "I-State"
            ]:
                msg = "'%s' is not a valid weight type." % weight
                raise ValueError(msg)
            Weights.append(objectify.Element('weight', name=weight))
        OPTIONS.append(
            objectify.Element("SaveOnlyData", value=bool_formatter(self.SaveOnlyData)))
        OPTIONS.append(
            objectify.Element(
                "SaveLabelsOnSeparateFile", value=bool_formatter(self.SaveLabelsOnSeparateFile)))
        OPTIONS.append(objectify.Element("SaveFormatBlock", value=self.SaveFormatBlock))
        OPTIONS.append(objectify.Element("SaveFormatSubBlock", value=self.SaveFormatSubBlock))
        OPTIONS.append(
            objectify.Element("SaveExcludeMisVal", value=bool_formatter(self.SaveExcludeMisVal)))
        OPTIONS.append(
            objectify.Element(
                "SaveExcludeAllMisVal", value=bool_formatter(self.SaveExcludeAllMisVal)))
        OPTIONS.append(
            objectify.Element("SaveExcludeConst", value=bool_formatter(self.SaveExcludeConst)))
        OPTIONS.append(
            objectify.Element(
                "SaveExcludeNearConst", value=bool_formatter(self.SaveExcludeNearConst)))
        OPTIONS.append(
            objectify.Element("SaveExcludeStdDev", value=bool_formatter(self.SaveExcludeStdDev)))
        OPTIONS.append(objectify.Element("SaveStdDevThreshold", value=self.SaveStdDevThreshold))
        OPTIONS.append(
            objectify.Element(
                "SaveExcludeCorrelated", value=bool_formatter(self.SaveExcludeCorrelated)))
        OPTIONS.append(objectify.Element("SaveCorrThreshold", value=self.SaveCorrThreshold))
        OPTIONS.append(
            objectify.Element(
                "SaveExclusionOptionsToVariables",
                value=bool_formatter(self.SaveExclusionOptionsToVariables)))
        OPTIONS.append(
            objectify.Element(
                "SaveExcludeMisMolecules", value=bool_formatter(self.SaveExcludeMisMolecules)))
        OPTIONS.append(
            objectify.Element(
                "SaveExcludeRejectedMolecules",
                value=bool_formatter(self.SaveExcludeRejectedMolecules)))
        OPTIONS.append(
            objectify.Element(
                "RoundDescriptorValues", value=bool_formatter(self.RoundDescriptorValues)))

        DESCRIPTORS = objectify.SubElement(self.dragon, "DESCRIPTORS")
        for i in self.blocks:
            if i < 1 or i > 30:
                msg = "block id must be in range 1 to 30."
                raise ValueError(msg)
            DESCRIPTORS.append(objectify.Element('block', id="%i" % i, SelectAll="true"))

        MOLFILES = objectify.SubElement(self.dragon, "MOLFILES")
        MOLFILES.append(objectify.Element("molInput", value="file"))
        # if self.molInput == "stdin":
        #     if self.molInputFormat not in [
        #             'SYBYL', 'MDL', 'HYPERCHEM', 'SMILES', 'CML', 'MACROMODEL'
        #     ]:
        #         msg = "'%s' is not a valid molInputFormat. Formats:['SYBYL','MDL','HYPERCHEM','SMILES','CML','MACROMODEL']" % self.molInputFormat
        #         raise ValueError(msg)
        #     MOLFILES.append(objectify.Element("molInputFormat", value=self.molInputFormat))
        
        if input_mol_type == 'mol2':
            for mol2 in input_mol_list:
                MOLFILES.append(objectify.Element("molFile", value=mol2))
        
        else:
            MOLFILES.append(objectify.Element("molFile", value=input_mol_list[0]))
        
        # if isinstance(self.molFile, dict):
            # for f in range(1, len(self.molFile) + 1):
                # if os.path.exists(self.molFile[f]['file']):
                    # MOLFILES.append(objectify.Element("molFile", value=self.molFile[f]['file']))
                # else:
                    # msg = "file not found at "+ self.molFile[f]['file']
                    # raise FileNotFoundError(msg)
        # elif isinstance(self.molFile, str):
            # MOLFILES.append(objectify.Element("molFile", value=self.molFile))
        # else:
            # msg = 'Variable molFile can be either a string or a list'
            # raise ValueError(msg)
    
        
        OUTPUT = objectify.SubElement(self.dragon, "OUTPUT")
        OUTPUT.append(objectify.Element("knimemode", value=bool_formatter(self.knimemode)))
        OUTPUT.append(objectify.Element("SaveStdOut", value=bool_formatter(self.SaveStdOut)))
        OUTPUT.append(objectify.Element("SaveProject", value=bool_formatter(self.SaveProject)))
        if self.SaveProject:
            OUTPUT.append(objectify.Element("SaveProjectFile", value=self.SaveProjectFile))
        OUTPUT.append(objectify.Element("SaveFile", value=bool_formatter(self.SaveFile)))
        if self.SaveFile:
            OUTPUT.append(objectify.Element(
                "SaveType", value=self.SaveType))  # value = "[singlefile/block/subblock]"
            OUTPUT.append(
                objectify.Element(
                    "SaveFilePath", value=self.output_directory + self.SaveFilePath)
            )  #Specifies the file name for saving results as a plan text file(s), if the "singlefile" option is set; if "block" or "subblock" are set, specifies the path in which results files will be saved.
        OUTPUT.append(objectify.Element("logMode",
                                        value=self.logMode))  # value = [none/stderr/file]
        if self.logMode == "file":
            OUTPUT.append(
                objectify.Element("logFile", value=self.output_directory + self.logFile))

        if self.external:
            EXTERNAL = objectify.SubElement(self.dragon, "EXTERNAL")
            EXTERNAL.append(objectify.Element("fileName", value=self.fileName))
            EXTERNAL.append(objectify.Element("delimiter", value=self.delimiter))
            EXTERNAL.append(
                objectify.Element(
                    "consecutiveDelimiter", value=bool_formatter(self.consecutiveDelimiter)))
            EXTERNAL.append(objectify.Element("MissingValue", value=self.MissingValue))
        
        # objectify.deannotate(self.dragon)
        etree.cleanup_namespaces(self.dragon)
        self.drs_name = 'Dragon_script.drs'
        with open(os.path.join(self.output_directory, self.drs_name), 'w') as outfile:
            outfile.write(etree.tostring(self.dragon, pretty_print=True).decode())
        
        self.data_path = self.dragon.OUTPUT.SaveFilePath.attrib['value']
        return self


    def _run(self):
        t0 = time.time()
        print("running Dragon%i ..." % self.version)
        os_ret = os.system('nohup dragon%sshell -s %s' %
                           (self.version, os.path.join(self.output_directory, self.drs_name)))
        if os_ret != 0:
            msg = "Oops, dragon%ishell command didn't work! Are you sure Dragon%i software is installed on your machine?" % (
                self.version, self.version)
            raise ImportError(msg)

        # execution time
        tmp_str = tot_exec_time_str(t0)
        print("... Dragon job completed in %s"%tmp_str)

        # print subprocess.check_output(['nohup dragon%sshell -s %s'%(self.version,self.drs)])


    def represent(self, mol_list, output_directory='./', dropna=True):
        """
        Parameters
        ----------
        mol_list: list
            list of chemml.chem.Molecule objects
            
        output_directory: str
            output directory to save dragon scripts
        
        dropna: bool
            Drops all columns with any NaN value.
        
        Returns
        -------
        """
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        
        if not isinstance(mol_list, list) or not mol_list:
            raise ValueError("mol_list variable should be a list of chemml.chem.Molecule objects")
        
        # check chemml.chem.molecule.Molecule type
        from chemml.chem import Molecule
        check_mols = [isinstance(mol, Molecule) for mol in mol_list]
        
        if False in check_mols:
            raise ValueError("mol_list variable should be a uniform list of valid chemml.chem.Molecule objects")
            
        else:
            if mol_list[0].creator[0].lower() == 'smiles':
                smiles_mol_list = [i.creator[1] + '\n' for i in mol_list]
                input_mol_list = [output_directory + 'molecules.smi']
                with open(input_mol_list[0], 'w') as f: f.writelines(smiles_mol_list)
                input_mol_type = 'smiles'


            if mol_list[0].creator[0].lower() == 'xyz':
                from chemml.utils import ConvertFile
                xyz_mol_list = [i.creator[1] for i in mol_list]
                input_mol_list = ConvertFile(xyz_mol_list, 'xyz', 'mol2')
                input_mol_type = 'mol2'
            
            if mol_list[0].creator[0].lower() == 'mol2':
                input_mol_list = [i.creator[1] for i in mol_list]
                input_mol_type = 'mol2'
                
        
        # call run and convert_to_csv
        self._script_wizard(input_mol_list=input_mol_list, input_mol_type=input_mol_type)
        self._run()
        
        # df = self._convert_to_csv(remove=True)
        df = pd.read_csv(self.data_path, sep=None, engine='python')
        
        # remove original tab delimited file
        os.remove(self.data_path)
        self.data_path = None

        df = df.drop(['No.','NAME'], axis=1)
        df = df.loc[:, (df != df.iloc[0]).any()]
        df = df.replace(to_replace='nan',value=np.nan)
        df = df.replace(to_replace='na',value=np.nan)
        if dropna: df.dropna(axis=1,inplace=True)
        
        return df
        
    
    
    
    

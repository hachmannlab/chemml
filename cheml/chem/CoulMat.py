import pandas as pd
import numpy as np
from lxml import objectify, etree
import subprocess
import warnings
import os 
from ..utils.utilities import std_datetime_str

__all__ = [
    'CoulombMatrix',
]


class CoulombMatrix(object):
    """ The implementation of coulomb matrix by Matthias Rupp et al 2012, PRL.
    
    Parameters
    ----------
    version: int, optional (default=7)
        The version of available Dragon on the user's machine
    
    Weights: list, optional (default=["Mass","VdWVolume","Electronegativity","Polarizability","Ionization","I-State"])
        A list of weights to be used

    blocks: list, optional (default=False)
        A list of descriptor blocks' id. For all of them parameter SelectAll="true" is given. 
        To select descriptors one by one based on descriptor names, use Script Wizard in Drgon GUI.
                
    external: boolean, optional (default=False)
        If True, include external variables at the end of each saved file.

    Returns
    -------
    Dragon Script and descriptors.
    """
    def __init__(self, version = 6,CheckUpdates = True,SaveLayout = True,
                ShowWorksheet = False,Decimal_Separator = ".",Missing_String = "NaN",
                DefaultMolFormat = "1",HelpBrowser = "/usr/bin/xdg-open",RejectUnusualValence = False,
                Add2DHydrogens = False,MaxSRforAllCircuit = "19",MaxSR = "35",
                MaxSRDetour = "30",MaxAtomWalkPath = "2000",LogPathWalk = True,
                LogEdge = True,Weights = ["Mass","VdWVolume","Electronegativity","Polarizability","Ionization","I-State"],
                SaveOnlyData = False,SaveLabelsOnSeparateFile = False,SaveFormatBlock = "%b-%n.txt",
                SaveFormatSubBlock = "%b-%s-%n-%m.txt",SaveExcludeMisVal = False,SaveExcludeAllMisVal = False,
                SaveExcludeConst = False,SaveExcludeNearConst = False,SaveExcludeStdDev = False,
                SaveStdDevThreshold = "0.0001",SaveExcludeCorrelated = False,SaveCorrThreshold = "0.95",
                SaveExclusionOptionsToVariables = False,SaveExcludeMisMolecules = False,
                SaveExcludeRejectedMolecules = False,blocks = range(1,30),molInput = "stdin",
                molInputFormat = "SMILES",molFile = None,SaveStdOut = False,SaveProject = False,
                SaveProjectFile = "Dragon_project.drp",SaveFile = True,SaveType = "singlefile",
                SaveFilePath = "Dragon_descriptors.txt",logMode = "file",logFile = "Dragon_log.txt",
                external = False,fileName = None,delimiter = ",",consecutiveDelimiter = False,MissingValue = "NaN"):
        self.version = version
        self.CheckUpdates = CheckUpdates

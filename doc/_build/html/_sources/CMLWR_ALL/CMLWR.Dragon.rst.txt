.. _Dragon:

Dragon
=======

:task:
    | Prepare Data

:subtask:
    | feature representation

:host:
    | cheml

:function:
    | Dragon

:input tokens (receivers):
    | ``molfile`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   input DataFrame

:output tokens (senders):
    | ``df`` : pandas DataFrame, shape(n_samples, n_features), requied
    |   output DataFrame

:required parameters:
    | script  ( required for the block: string - 'new' or a path to the script file)
    | output_directory  (  required for the block: string - '')
    |
    .. note:: The documentation for this function can be found here_

    .. _here: :py:func:`cheml.chem.Dragon`

:required packages:
    | :py:mod:`cheml`, version: 1.3.1
    | Pandas_, version: 0.20.2\n\n    .. _Pandas: http://pandas.pydata.org
    | Dragon_, version: 6 or 7\n\n    .. _Dragon: http://www.talete.mi.it/products/dragon_description.htm

:input file view:
    | ``## Prepare Data``
    |   ``<< host = cheml    << function = Dragon``
    |   ``<< version  =  6``
    |   ``<< CheckUpdates  =  True``
    |   ``<< SaveLayout  =  True``
    |   ``<< PreserveTemporaryProjects  =  True``
    |   ``<< ShowWorksheet  =  False``
    |   ``<< Decimal_Separator  =  "."``
    |   ``<< Missing_String  =  "NaN"``
    |   ``<< DefaultMolFormat  =  "1"``
    |   ``<< HelpBrowser  =  "/usr/bin/xdg-open"``
    |   ``<< RejectUnusualValence  =  False``
    |   ``<< Add2DHydrogens  =  False``
    |   ``<< MaxSRforAllCircuit  =  "19"``
    |   ``<< MaxSR  =  "35"``
    |   ``<< MaxSRDetour  =  "30"``
    |   ``<< MaxAtomWalkPath  =  "2000"``
    |   ``<< LogPathWalk  =  True``
    |   ``<< LogEdge  =  True``
    |   ``<< Weights  =  ["Mass"-"VdWVolume"-"Electronegativity"-"Polarizability"-"Ionization"-"I-State"]``
    |   ``<< SaveOnlyData  =  False``
    |   ``<< SaveLabelsOnSeparateFile  =  False``
    |   ``<< SaveFormatBlock  =  "%b-%n.txt"``
    |   ``<< SaveFormatSubBlock  =  "%b-%s-%n-%m.txt"``
    |   ``<< SaveExcludeMisVal  =  False``
    |   ``<< SaveExcludeAllMisVal  =  False``
    |   ``<< SaveExcludeConst  =  False``
    |   ``<< SaveExcludeNearConst  =  False``
    |   ``<< SaveExcludeStdDev  =  False``
    |   ``<< SaveStdDevThreshold  =  "0.0001"``
    |   ``<< SaveExcludeCorrelated  =  False``
    |   ``<< SaveCorrThreshold  =  "0.95"``
    |   ``<< SaveExclusionOptionsToVariables  =  False``
    |   ``<< SaveExcludeMisMolecules  =  False``
    |   ``<< SaveExcludeRejectedMolecules  =  False``
    |   ``<< blocks  =  range(1-30)``
    |   ``<< molInput  =  "stdin"``
    |   ``<< molInputFormat  =  "SMILES"``
    |   ``<< molFile  =  None``
    |   ``<< SaveStdOut  =  False``
    |   ``<< SaveProject  =  False``
    |   ``<< SaveProjectFile  =  "Dragon_project.drp"``
    |   ``<< SaveFile  =  True``
    |   ``<< SaveType  =  "singlefile"``
    |   ``<< SaveFilePath  =  "Dragon_descriptors.txt"``
    |   ``<< logMode  =  "file"``
    |   ``<< logFile  =  "Dragon_log.txt"``
    |   ``<< external  =  False``
    |   ``<< fileName  =  None``
    |   ``<< delimiter  =  "-"``
    |   ``<< consecutiveDelimiter  =  False``
    |   ``<< MissingValue  =  "NaN"``
    |   ``<< RejectDisconnectedStrucuture  =  False``
    |   ``<< RetainBiggestFragment  =  False``
    |   ``<< DisconnectedCalculationOption  =  "0"``
    |   ``<< RoundCoordinates  =  True``
    |   ``<< RoundWeights  =  True``
    |   ``<< RoundDescriptorValues  =  True``
    |   ``<< knimemode  =  False``
    |   ``>> id molfile``
    |   ``>> df id``
    |
    .. note:: The rest of parameters (if any) can be set the same way.
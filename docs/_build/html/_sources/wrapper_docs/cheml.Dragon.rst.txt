.. _Dragon:

Dragon
=======

:task:
    | Prepare

:subtask:
    | feature representation

:host:
    | cheml

:function:
    | Dragon

:input tokens (receivers):
    | ``molfile`` : the molecule file path
    |   ("<type 'str'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:wrapper parameters:
    | ``script`` : , (default:new)
    |   
    |   choose one of: []

:required packages:
    | ChemML, 0.1.0
    | pandas, 0.20.3
    | Dragon, 7 or 6
    | lxml, 3.4.0

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = Dragon``
    |   ``<< script = new``
    |   ``<< SaveExcludeConst = False``
    |   ``<< MaxSR = '35'``
    |   ``<< SaveFilePath = Dragon_descriptors.txt``
    |   ``<< output_directory = ./``
    |   ``<< DisconnectedCalculationOption = '0'``
    |   ``<< SaveExcludeNearConst = False``
    |   ``<< Add2DHydrogens = False``
    |   ``<< SaveProject = False``
    |   ``<< Decimal_Separator = .``
    |   ``<< SaveOnlyData = False``
    |   ``<< script = new``
    |   ``<< RejectDisconnectedStrucuture = False``
    |   ``<< SaveExclusionOptionsToVariables = False``
    |   ``<< LogEdge = True``
    |   ``<< LogPathWalk = True``
    |   ``<< SaveLabelsOnSeparateFile = False``
    |   ``<< version = 6``
    |   ``<< DefaultMolFormat = '1'``
    |   ``<< MaxSRDetour = '30'``
    |   ``<< HelpBrowser = /usr/bin/xdg-open``
    |   ``<< SaveExcludeRejectedMolecules = False``
    |   ``<< knimemode = False``
    |   ``<< RejectUnusualValence = False``
    |   ``<< SaveProjectFile = Dragon_project.drp``
    |   ``<< SaveStdOut = False``
    |   ``<< SaveFormatSubBlock = %b-%s-%n-%m.txt``
    |   ``<< blocks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]``
    |   ``<< SaveExcludeCorrelated = False``
    |   ``<< molFile = required_required``
    |   ``<< consecutiveDelimiter = False``
    |   ``<< molInputFormat = SMILES``
    |   ``<< MaxAtomWalkPath = '2000'``
    |   ``<< SaveExcludeAllMisVal = False``
    |   ``<< SaveExcludeStdDev = False``
    |   ``<< Weights = ['Mass', 'VdWVolume', 'Electronegativity', 'Polarizability', 'Ionization', 'I-State']``
    |   ``<< external = False``
    |   ``<< RoundWeights = True``
    |   ``<< MaxSRforAllCircuit = '19'``
    |   ``<< fileName = None``
    |   ``<< RoundCoordinates = True``
    |   ``<< Missing_String = NaN``
    |   ``<< SaveExcludeMisVal = False``
    |   ``<< logFile = Dragon_log.txt``
    |   ``<< PreserveTemporaryProjects = True``
    |   ``<< SaveLayout = True``
    |   ``<< molInput = file``
    |   ``<< SaveFormatBlock = %b-%n.txt``
    |   ``<< MissingValue = NaN``
    |   ``<< SaveCorrThreshold = '0.95'``
    |   ``<< SaveType = singlefile``
    |   ``<< ShowWorksheet = False``
    |   ``<< delimiter = ,``
    |   ``<< RetainBiggestFragment = False``
    |   ``<< CheckUpdates = True``
    |   ``<< RoundDescriptorValues = True``
    |   ``<< SaveExcludeMisMolecules = False``
    |   ``<< SaveStdDevThreshold = '0.0001'``
    |   ``<< SaveFile = True``
    |   ``<< logMode = file``
    |   ``>> id molfile``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: 
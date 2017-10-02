
.. _Dragon:

Dragon
============

:task:
    | Prepare Data

:host:
    | cheml

:function:
    | Dragon

:parameters:
    |
    .. note:: The documentation for this method can be found here: :py:func:`cheml.chem.Dragon`

:send tokens:
    | ``df`` : pandas data frame, shape(n_samples, n_features)
    |   feature values matrix

:receive tokens:
    | ``molfile`` : string
    |   The path to the input molecule file

:requirements:
    | Dragon_, version: 6 or 7

    .. _Dragon: http://www.talete.mi.it/products/dragon_description.htm

:input file view:
    | ``## Prepare Data``
    |   ``<< host = cheml``
    |   ``<< function = Dragon``
    |   ``<< version = 7``
    |   ``<< script = "new"``
    |   ``<< output_directory = 'dragon'``

    |   ``<< CheckUpdates = True                  << SaveLayout = True``
    |   ``<< PreserveTemporaryProjects = True``

    |   ``<< RejectDisconnectedStrucuture = False << RetainBiggestFragment = False``
    |   ``<< DisconnectedCalculationOption = "0"``
    |   ``<< RoundCoordinates = True              << RoundWeights = True``
    |   ``<< RoundDescriptorValues = True         << knimemode = False``

    |   ``<< ShowWorksheet = False                << Decimal_Separator = "."``
    |   ``<< Missing_String = "Nan"               << DefaultMolFormat = "1"``
    |   ``<< HelpBrowser = "/usr/bin/xdg-open"    << RejectUnusualValence = False``
    |   ``<< Add2DHydrogens = False               << MaxSRforAllCircuit = "19"``
    |   ``<< MaxSR = "35"                         << MaxSRDetour = "30"``
    |   ``<< MaxAtomWalkPath = "2000"             << LogPathWalk = True``
    |   ``<< LogEdge = True                       << Weights = ["Mass","VdWVolume","Electronegativity","Polarizability","Ionization","I-State"]``
    |   ``<< SaveOnlyData = False                 << SaveLabelsOnSeparateFile = False``
    |   ``<< SaveFormatBlock = "%b-%n.txt"        << SaveFormatSubBlock = "%b-%s-%n-%m.txt"``
    |   ``<< SaveExcludeMisVal = False            << SaveExcludeAllMisVal = False``
    |   ``<< SaveExcludeConst = False             << SaveExcludeNearConst = False``
    |   ``<< SaveExcludeStdDev = False            << SaveStdDevThreshold = "0.0001"``
    |   ``<< SaveExcludeCorrelated = False        << SaveCorrThreshold = "0.95"``
    |   ``<< SaveExcludeMisMolecules = False      << SaveExclusionOptionsToVariables = False``
    |   ``<< SaveExcludeRejectedMolecules = False``

    |   ``<< blocks = range(1,2)``

    |   ``<< molInput = "file"                    << molInputFormat = "SMILES"``
    |   ``<< molFile = "@molfile"``

    |   ``<< SaveStdOut = False``
    |   ``<< SaveProject = False                  << SaveProjectFile = "Dragon_project.drp"``
    |   ``<< SaveFile = True``
    |   ``<< SaveType = "singlefile"              << SaveFilePath = "Dragon_descriptors.txt"``
    |   ``<< logMode = "file"                     << logFile = "Dragon_log.txt"``

    |   ``<< external = False                     << fileName = None``
    |   ``<< delimiter = ","                      << MissingValue = "NaN"``
    |   ``<< consecutiveDelimiter = False``


    |   ``>> id molfile    >> df id``

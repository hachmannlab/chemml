import numpy as np
import pandas as pd
from .containers import Input, Output, Parameter, req, regression_types, cv_types

class PyScript(object):
    task = 'Prepare'
    subtask = 'python script'
    host = 'cheml'
    function = 'PyScript'
    modules = ('cheml','')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        iv1 = Input("iv1","input variable, of any format", ())
        iv2 = Input("iv2","input variable, of any format", ())
        iv3 = Input("iv3","input variable, of any format", ())
        iv4 = Input("iv4","input variable, of any format", ())
        iv5 = Input("iv5","input variable, of any format", ())
        iv6 = Input("iv6", "input variable, of any format", ())
    class Outputs:
        ov1 = Output("ov1","output variable, of any format", ())
        ov2 = Output("ov2","output variable, of any format", ())
        ov3 = Output("ov3","output variable, of any format", ())
        ov4 = Output("ov4","output variable, of any format", ())
        ov5 = Output("ov5","output variable, of any format", ())
        ov6 = Output("ov6", "output variable, of any format", ())
    class WParameters:
        pass
    class FParameters:
        pass

class RDKFingerprint(object):
    task = 'Prepare'
    subtask = 'feature representation'
    host = 'cheml'
    function = 'RDKFingerprint'
    modules = ('cheml','chem')
    requirements = (req(0), req(2), req(3))
    documentation = ""

    class Inputs:
        molfile = Input("molfile","the molecule file path", ("<type 'str'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        removed_rows = Output("removed_rows","output variable, of any format", ())
    class WParameters:
        pass
    class FParameters:
        FPtype = Parameter('FPtype', 'Morgan')
        vector = Parameter('vector', 'bit')
        nBits = Parameter('nBits', 1024)
        radius = Parameter('radius', 2)
        removeHs = Parameter('removeHs', True)
        molfile = Parameter('molfile', '* required')
        path = Parameter('path', None)
        arguments = Parameter('arguments', [])

class Dragon(object):
    task = 'Prepare'
    subtask = 'feature representation'
    host = 'cheml'
    function = 'Dragon'
    modules = ('cheml','chem')
    requirements = (req(0), req(2), req(4), req(5))
    documentation = ""

    class Inputs:
        molfile = Input("molfile","the molecule file path", ("<type 'str'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        FPtype = Parameter('output_directory', './')
        script = Parameter('script', 'new')

        SaveStdOut = Parameter('SaveStdOut', False)
        DisconnectedCalculationOption = Parameter('DisconnectedCalculationOption', "0", format="string")
        MaxSR = Parameter('MaxSR', 35)
        SaveFilePath = Parameter('SaveFilePath', "Dragon_descriptors.txt")
        SaveExcludeMisMolecules = Parameter('SaveExcludeMisMolecules', False)
        SaveExcludeStdDev = Parameter('SaveExcludeStdDev', False)
        SaveExcludeNearConst = Parameter('SaveExcludeNearConst', False)
        SaveProject = Parameter('SaveProject', False)
        Add2DHydrogens = Parameter('Add2DHydrogens', False)
        blocks = Parameter('blocks',
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                            26, 27, 28, 29])
        SaveProjectFile = Parameter('SaveProjectFile', 'Dragon_project.drp')
        SaveOnlyData = Parameter('SaveOnlyData', False)
        MissingValue = Parameter('MissingValue', 'NaN')
        RejectDisconnectedStrucuture = Parameter('RejectDisconnectedStrucuture', False)
        SaveExclusionOptionsToVariables = Parameter('SaveExclusionOptionsToVariables', False)
        LogEdge = Parameter('LogEdge', True)
        LogPathWalk = Parameter('LogPathWalk', True)
        SaveLabelsOnSeparateFile = Parameter('SaveLabelsOnSeparateFile', False)
        version = Parameter('version', 6)
        DefaultMolFormat = Parameter('DefaultMolFormat', "1", format='string')
        molFile = Parameter('molFile', '* required')
        HelpBrowser = Parameter('HelpBrowser', "/usr/bin/xdg-open")
        SaveExcludeRejectedMolecules = Parameter('SaveExcludeRejectedMolecules', False)
        knimemode = Parameter('knimemode', False)
        RejectUnusualValence = Parameter('RejectUnusualValence', False)
        SaveStdDevThreshold = Parameter('SaveStdDevThreshold', "0.0001", format='string')
        SaveExcludeConst = Parameter('SaveExcludeConst', False)
        SaveFormatSubBlock = Parameter('SaveFormatSubBlock', "%b-%s-%n-%m.txt")
        Decimal_Separator = Parameter('Decimal_Separator','.')
        SaveExcludeCorrelated = Parameter('SaveExcludeCorrelated', False)
        MaxSRDetour = Parameter('MaxSRDetour', "30", format='string')
        consecutiveDelimiter = Parameter('consecutiveDelimiter', False)
        molInputFormat = Parameter('molInputFormat', "SMILES")
        SaveExcludeAllMisVal = Parameter('SaveExcludeAllMisVal', False)
        Weights = Parameter('Weights',
                            ['Mass', 'VdWVolume', 'Electronegativity', 'Polarizability', 'Ionization', 'I-State'])
        external = Parameter('external', False)
        RoundWeights = Parameter('RoundWeights', True)
        MaxSRforAllCircuit = Parameter('MaxSRforAllCircuit', 19)
        fileName = Parameter('fileName', None)
        RoundCoordinates = Parameter('RoundCoordinates', True)
        Missing_String = Parameter('Missing_String', "NaN")
        SaveExcludeMisVal = Parameter('SaveExcludeMisVal', False)
        logFile = Parameter('logFile', "Dragon_log.txt")
        RoundDescriptorValues = Parameter('RoundDescriptorValues', True)
        PreserveTemporaryProjects = Parameter('PreserveTemporaryProjects', True)
        SaveLayout = Parameter('SaveLayout', True)
        molInput = Parameter('molInput', "stdin")
        SaveFormatBlock = Parameter('SaveFormatBlock',"%b-%n.txt")
        SaveType = Parameter('SaveType', "singlefile")
        ShowWorksheet = Parameter('ShowWorksheet', False)
        delimiter = Parameter('delimiter',",")
        RetainBiggestFragment = Parameter('RetainBiggestFragment', False)
        CheckUpdates = Parameter('CheckUpdates', True)
        MaxAtomWalkPath = Parameter('MaxAtomWalkPath', "2000", format='string')
        logMode = Parameter('logMode', "file")
        SaveCorrThreshold = Parameter('SaveCorrThreshold', "0.95", format='string')
        SaveFile = Parameter('SaveFile', True)

class CoulombMatrix(object):
    task = 'Prepare'
    subtask = 'feature representation'
    host = 'cheml'
    function = 'CoulombMatrix'
    modules = ('cheml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        molfile = Input("molfile","the molecule file path", ("<type 'str'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        CMtype = Parameter('CMtype', 'SC')
        nPerm = Parameter('nPerm', '6')
        const = Parameter('const', 1)
        molfile = Parameter('molfile', '* required')
        path = Parameter('path', None)
        skip_lines = Parameter('skip_lines', [2,0])
        reader = Parameter('reader', 'auto')
        arguments = Parameter('arguments', [])

class DistanceMatrix(object):
    task = 'Prepare'
    subtask = 'feature representation'
    host = 'cheml'
    function = 'DistanceMatrix'
    modules = ('cheml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        norm_type = Parameter('norm_type', 'fro')
        nCores = Parameter('nCores', 1)

class MissingValues(object):
    task = 'Prepare'
    subtask = 'preprocessor'
    host = 'cheml'
    function = 'MissingValues'
    modules = ('cheml','preprocessing')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        api = Input("api","instance of ChemML's MissingValues class", ("<class 'cheml.preprocessing.handle_missing.missing_values'>",))
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of ChemML's MissingValues class", ("<class 'cheml.preprocessing.handle_missing.missing_values'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit_transform','transform', None))
    class FParameters:
        strategy = Parameter('strategy', 'ignore_row')
        string_as_null = Parameter('string_as_null', True)
        inf_as_null = Parameter('inf_as_null', True)
        missing_values = Parameter('missing_values', False)

class Merge(object):
    task = 'Prepare'
    subtask = 'basic operators'
    host = 'cheml'
    function = 'Merge'
    modules = ('cheml','initialization')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        df1 = Input("df1","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        df2 = Input("df2","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        pass

class Split(object):
    task = 'Prepare'
    subtask = 'basic operators'
    host = 'cheml'
    function = 'Split'
    modules = ('cheml','initialization')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        df = Input("df", "pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        df1 = Output("df1","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        df2 = Output("df2","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        selection = Parameter('selection', 1)

class Constant(object):
    task = 'Prepare'
    subtask = 'preprocessing'
    host = 'cheml'
    function = 'Constant'
    modules = ('cheml','preprocessing')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        df = Input("df", "pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Input("api", "instance of ChemML's Constant class", ("<class 'cheml.preprocessing.purge.Constant'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        removed_columns_ = Output("removed_columns_","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of ChemML's Constant class", ("<class 'cheml.preprocessing.purge.Constant'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit_transform','transform', None))
    class FParameters:
        selection = Parameter('selection', 1)

class mlp_hogwild(object):
    task = 'Model'
    subtask = 'regression'
    host = 'cheml'
    function = 'mlp_hogwild'
    modules = ('cheml','nn')
    requirements = (req(0), req(1), req(2))
    documentation = ""

    class Inputs:
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfy = Input("dfy", "pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Input("api", "instance of ChemML's mlp_hogwild class", ("<class 'cheml.nn.nn_psgd.mlp_hogwild'>",))
    class Outputs:
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api", "instance of ChemML's mlp_hogwild class", ("<class 'cheml.nn.nn_psgd.mlp_hogwild'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
    class FParameters:
        rms_decay = Parameter('rms_decay', 0.9)
        learn_rate = Parameter('learn_rate', 0.001)
        input_act_funcs = Parameter('input_act_funcs', '*required')
        nneurons = Parameter('nneurons', '*required')
        batch_size = Parameter('batch_size', 256)
        n_epochs = Parameter('n_epochs', 10000)
        validation_size = Parameter('validation_size', 0.2)
        print_level = Parameter('print_level', 1)
        n_hist = Parameter('n_hist', 20)
        threshold = Parameter('threshold', 0.1)
        model = Parameter('model', None)
        n_check = Parameter('n_check', 50)
        n_cores = Parameter('n_cores', 1)

class SaveFile(object):
    task = 'Store'
    subtask = 'file'
    host = 'cheml'
    function = 'SaveFile'
    modules = ('cheml','initialization')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        filepath = Output("filepath","pandas dataframe", ("<type 'str'>",))
    class WParameters:
        pass
    class FParameters:
        index = Parameter('index', False)
        record_time = Parameter('record_time', False)
        format = Parameter('format', 'csv')
        output_directory = Parameter('output_directory', None)
        header = Parameter('header', True)
        filename = Parameter('filename', '* required')

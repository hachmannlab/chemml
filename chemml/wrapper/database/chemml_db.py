from chemml.wrapper.database.containers import Input, Output, Parameter, req, regression_types, cv_classes

class PyScript(object):
    task = 'Input'
    subtask = 'python script'
    host = 'chemml'
    function = 'PyScript'
    modules = ('chemml','')
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
        line01 = Parameter('line01', 'type python code')
        line02 = Parameter('line02', 'input tokens are available as ...')
        line03 = Parameter('line03', '... python variables')
        line04 = Parameter('line04', 'type python code')
        line05 = Parameter('line05', 'type python code')
        line06 = Parameter('line06', 'type python code')
        line07 = Parameter('line07', 'type python code')
        line08 = Parameter('line08', 'type python code')
        line09 = Parameter('line09', 'type python code')
        line10 = Parameter('line10', 'type python code')
        line11 = Parameter('line11', 'type python code')
        line12 = Parameter('line12', 'type python code')
        line13 = Parameter('line13', 'type python code')
        line14 = Parameter('line14', 'type python code')
        line15 = Parameter('line15', 'type python code')
        line16 = Parameter('line16', 'type python code')
        line17 = Parameter('line17', 'type python code')
        line18 = Parameter('line18', 'type python code')
        line19 = Parameter('line19', 'type python code')
        line20 = Parameter('line20', 'type python code')

class RDKitFingerprint(object):
    task = 'Represent'
    subtask = 'molecular descriptors'
    host = 'chemml'
    function = 'RDKitFingerprint'
    modules = ('chemml','chem')
    requirements = (req(0), req(2), req(3))
    documentation = ""

    class Inputs:
        molfile = Input("molfile","the molecule file path", ("<class 'str'>",))
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
        molfile = Parameter('molfile', 'required_required')
        path = Parameter('path', None)
        arguments = Parameter('arguments', [])

class Dragon(object):
    task = 'Represent'
    subtask = 'molecular descriptors'
    host = 'chemml'
    function = 'Dragon'
    modules = ('chemml','chem')
    requirements = (req(0), req(2), req(4), req(5))
    documentation = ""

    class Inputs:
        molfile = Input("molfile","the molecule file path", ("<class 'str'>","<class 'dict'>","<class 'list'>"))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        script = Parameter('script', 'new')
    class FParameters:
        FPtype = Parameter('output_directory', './')
        script = Parameter('script', 'new')

        SaveStdOut = Parameter('SaveStdOut', False)
        DisconnectedCalculationOption = Parameter('DisconnectedCalculationOption', "'0'", format="string")
        MaxSR = Parameter('MaxSR', "'35'", format="string")
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
        DefaultMolFormat = Parameter('DefaultMolFormat', "'1'", format='string')
        molFile = Parameter('molFile', 'required_required')
        HelpBrowser = Parameter('HelpBrowser', "/usr/bin/xdg-open")
        SaveExcludeRejectedMolecules = Parameter('SaveExcludeRejectedMolecules', False)
        knimemode = Parameter('knimemode', False)
        RejectUnusualValence = Parameter('RejectUnusualValence', False)
        SaveStdDevThreshold = Parameter('SaveStdDevThreshold', "'0.0001'", format='string')
        SaveExcludeConst = Parameter('SaveExcludeConst', False)
        SaveFormatSubBlock = Parameter('SaveFormatSubBlock', "%b-%s-%n-%m.txt")
        Decimal_Separator = Parameter('Decimal_Separator','.')
        SaveExcludeCorrelated = Parameter('SaveExcludeCorrelated', False)
        MaxSRDetour = Parameter('MaxSRDetour', "'30'", format='string')
        consecutiveDelimiter = Parameter('consecutiveDelimiter', False)
        molInputFormat = Parameter('molInputFormat', "SMILES")
        SaveExcludeAllMisVal = Parameter('SaveExcludeAllMisVal', False)
        Weights = Parameter('Weights',
                            ['Mass', 'VdWVolume', 'Electronegativity', 'Polarizability', 'Ionization', 'I-State'])
        external = Parameter('external', False)
        RoundWeights = Parameter('RoundWeights', True)
        MaxSRforAllCircuit = Parameter('MaxSRforAllCircuit', "'19'",format="string")
        fileName = Parameter('fileName', None)
        RoundCoordinates = Parameter('RoundCoordinates', True)
        Missing_String = Parameter('Missing_String', "NaN")
        SaveExcludeMisVal = Parameter('SaveExcludeMisVal', False)
        logFile = Parameter('logFile', "Dragon_log.txt")
        RoundDescriptorValues = Parameter('RoundDescriptorValues', True)
        PreserveTemporaryProjects = Parameter('PreserveTemporaryProjects', True)
        SaveLayout = Parameter('SaveLayout', True)
        molInput = Parameter('molInput', "file")
        SaveFormatBlock = Parameter('SaveFormatBlock',"%b-%n.txt")
        SaveType = Parameter('SaveType', "singlefile")
        ShowWorksheet = Parameter('ShowWorksheet', False)
        delimiter = Parameter('delimiter',",")
        RetainBiggestFragment = Parameter('RetainBiggestFragment', False)
        CheckUpdates = Parameter('CheckUpdates', True)
        MaxAtomWalkPath = Parameter('MaxAtomWalkPath', "'2000'", format='string')
        logMode = Parameter('logMode', "file")
        SaveCorrThreshold = Parameter('SaveCorrThreshold', "'0.95'", format='string')
        SaveFile = Parameter('SaveFile', True)

class CoulombMatrix(object):
    task = 'Represent'
    subtask = 'molecular descriptors'
    host = 'chemml'
    function = 'CoulombMatrix'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        molecules = Input("molecules","the molecule numpy array or data frame", ("<class 'pandas.core.frame.DataFrame'>",
                                                                                 "<type 'numpy.ndarray'>","<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        CMtype = Parameter('CMtype', 'SC')
        max_n_atoms = Parameter('max_n_atoms', 'auto')
        nPerm = Parameter('nPerm', 3)
        const = Parameter('const', 1)

class BagofBonds(object):
    task = 'Represent'
    subtask = 'molecular descriptors'
    host = 'chemml'
    function = 'BagofBonds'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        molecules = Input("molecules","the molecule numpy array or data frame", ("<class 'pandas.core.frame.DataFrame'>",
                                                                                 "<type 'numpy.ndarray'>","<class 'dict'>",
                                                                                 "<class 'list'>"))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        const = Parameter('const', 1)

class DistanceMatrix(object):
    task = 'Represent'
    subtask = 'distance matrix'
    host = 'chemml'
    function = 'DistanceMatrix'
    modules = ('chemml','chem')
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

class Split(object):
    task = 'Prepare'
    subtask = 'data manipulation'
    host = 'chemml'
    function = 'Split'
    modules = ('chemml','initialization')
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

class ConstantColumns(object):
    task = 'Prepare'
    subtask = 'data cleaning'
    host = 'chemml'
    function = 'ConstantColumns'
    modules = ('chemml','preprocessing')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        df = Input("df", "pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Input("api", "instance of ChemML's Constant class", ("<class 'chemml.preprocessing.purge.ConstantColumns'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        removed_columns_ = Output("removed_columns_","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of ChemML's Constant class", ("<class 'chemml.preprocessing.purge.ConstantColumns'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit_transform','transform', None))
    class FParameters:
        pass

class Outliers(object):
    task = 'Prepare'
    subtask = 'data cleaning'
    host = 'chemml'
    function = 'Outliers'
    modules = ('chemml','preprocessing')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        df = Input("df", "pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Input("api", "instance of ChemML's Constant class", ("<class 'chemml.preprocessing.purge.Outliers'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        removed_columns_ = Output("removed_columns_","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of ChemML's Constant class", ("<class 'chemml.preprocessing.purge.Outliers'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit_transform','transform', None))
    class FParameters:
        m = Parameter('m', 2.0)
        strategy = Parameter('strategy', 'median')

class MissingValues(object):
    task = 'Prepare'
    subtask = 'data cleaning'
    host = 'chemml'
    function = 'MissingValues'
    modules = ('chemml','preprocessing')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        api = Input("api","instance of ChemML's MissingValues class", ("<class 'chemml.preprocessing.handle_missing.missing_values'>",))
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api","instance of ChemML's MissingValues class", ("<class 'chemml.preprocessing.handle_missing.missing_values'>",))
    class WParameters:
        func_method = Parameter('func_method','None','String',
                        description = "",
                        options = ('fit_transform','transform', None))
    class FParameters:
        strategy = Parameter('strategy', 'ignore_row',
                             format='String',
                             options = ['interpolate','zero','ignore_row','ignore_column'])
        string_as_null = Parameter('string_as_null', True, format = 'Boolean')
        inf_as_null = Parameter('inf_as_null', True, format = 'Boolean')
        missing_values = Parameter('missing_values', False, format = 'list of strings/floats/integers')

class MLP(object):
    task = 'Model'
    subtask = 'regression'
    host = 'chemml'
    function = 'MLP'
    modules = ('chemml','nn.keras')
    requirements = (req(0), req(8), req(9))
    documentation = ""

    class Inputs:
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfy = Input("dfy", "pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Input("api", "instance of chemml.nn.keras.MLP class", ("<class 'chemml.nn.keras.mlp.MLP'>",))
    class Outputs:
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api", "instance of chemml.nn.keras.MLP class", ("<class 'chemml.nn.keras.mlp.MLP'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
    class FParameters:
        nhidden = Parameter('nhidden', 1)
        nneurons = Parameter('nneurons', 100)
        activations = Parameter('activations', None)
        learning_rate = Parameter('learning_rate', 0.01)
        lr_decay = Parameter('lr_decay', 0.0)
        nepochs = Parameter('nepochs', 100)
        batch_size = Parameter('batch_size', 100)
        loss = Parameter('loss', 'mean_squared_error')
        regression = Parameter('regression', True)
        nclasses = Parameter('nclasses', None)
        layer_config_file = Parameter('layer_config_file', None)
        opt_config_file = Parameter('opt_config_file', None)

class MLP_sklearn(object):
    task = 'Model'
    subtask = 'regression'
    host = 'chemml'
    function = 'MLP_sklearn'
    modules = ('chemml','nn.keras')
    requirements = (req(0), req(1), req(8), req(9))
    documentation = ""

    class Inputs:
        dfx = Input("dfx","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfy = Input("dfy", "pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Input("api", "instance of chemml.nn.keras.MLP_sklearn class", ("<class 'chemml.nn.keras.mlp.MLP_sklearn'>",))
    class Outputs:
        dfy_predict = Output("dfy_predict","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        api = Output("api", "instance of chemml.nn.keras.MLP_sklearn class", ("<class 'chemml.nn.keras.mlp.MLP_sklearn'>",))
    class WParameters:
        func_method = Parameter('func_method','None','string',
                        description = "",
                        options = ('fit', 'predict', None))
    class FParameters:
        nhidden = Parameter('nhidden', 1)
        nneurons = Parameter('nneurons', 100)
        activations = Parameter('activations', None)
        learning_rate = Parameter('learning_rate', 0.01)
        lr_decay = Parameter('lr_decay', 0.0)
        nepochs = Parameter('nepochs', 100)
        batch_size = Parameter('batch_size', 100)
        loss = Parameter('loss', 'mean_squared_error')
        regression = Parameter('regression', True)
        nclasses = Parameter('nclasses', None)
        layer_config_file = Parameter('layer_config_file', None)
        opt_config_file = Parameter('opt_config_file', None)


class SaveFile(object):
    task = 'Output'
    subtask = 'file'
    host = 'chemml'
    function = 'SaveFile'
    modules = ('chemml','initialization')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class Outputs:
        filepath = Output("filepath","pandas dataframe", ("<class 'str'>",))
    class WParameters:
        pass
    class FParameters:
        index = Parameter('index', False)
        record_time = Parameter('record_time', False)
        format = Parameter('format', 'csv')
        output_directory = Parameter('output_directory', None)
        header = Parameter('header', True)
        filename = Parameter('filename', 'required_required')

class XYZreader(object):
    task = 'Input'
    subtask = 'xyz'
    host = 'chemml'
    function = 'XYZreader'
    modules = ('chemml','initialization')
    requirements = (req(0),)
    documentation = ""

    class Inputs:
        pass
    class Outputs:
        molecules = Output("molecules","dictionary of molecules with ['mol', 'file'] keys",
                           ("<class 'dict'>",))

    class WParameters:
        pass
    class FParameters:
        path_pattern = Parameter('path_pattern', 'required_required')
        path_root = Parameter('path_root', None)
        Z = Parameter('Z', {'Ru': 44.0, 'Re': 75.0, 'Rf': 104.0, 'Rg': 111.0, 'Ra': 88.0, 'Rb': 37.0, 'Rn': 86.0, 'Rh': 45.0, 'Be': 4.0, 'Ba': 56.0, 'Bh': 107.0, 'Bi': 83.0, 'Bk': 97.0, 'Br': 35.0, 'H': 1.0, 'P': 15.0, 'Os': 76.0, 'Es': 99.0, 'Hg': 80.0, 'Ge': 32.0, 'Gd': 64.0, 'Ga': 31.0, 'Pr': 59.0, 'Pt': 78.0, 'Pu': 94.0, 'C': 6.0, 'Pb': 82.0, 'Pa': 91.0, 'Pd': 46.0, 'Cd': 48.0, 'Po': 84.0, 'Pm': 61.0, 'Hs': 108.0, 'Uup': 115.0, 'Uus': 117.0, 'Uuo': 118.0, 'Ho': 67.0, 'Hf': 72.0, 'K': 19.0, 'He': 2.0, 'Md': 101.0, 'Mg': 12.0, 'Mo': 42.0, 'Mn': 25.0, 'O': 8.0, 'Mt': 109.0, 'S': 16.0, 'W': 74.0, 'Zn': 30.0, 'Eu': 63.0, 'Zr': 40.0, 'Er': 68.0, 'Ni': 28.0, 'No': 102.0, 'Na': 11.0, 'Nb': 41.0, 'Nd': 60.0, 'Ne': 10.0, 'Np': 93.0, 'Fr': 87.0, 'Fe': 26.0, 'Fl': 114.0, 'Fm': 100.0, 'B': 5.0, 'F': 9.0, 'Sr': 38.0, 'N': 7.0, 'Kr': 36.0, 'Si': 14.0, 'Sn': 50.0, 'Sm': 62.0, 'V': 23.0, 'Sc': 21.0, 'Sb': 51.0, 'Sg': 106.0, 'Se': 34.0, 'Co': 27.0, 'Cn': 112.0, 'Cm': 96.0, 'Cl': 17.0, 'Ca': 20.0, 'Cf': 98.0, 'Ce': 58.0, 'Xe': 54.0, 'Lu': 71.0, 'Cs': 55.0, 'Cr': 24.0, 'Cu': 29.0, 'La': 57.0, 'Li': 3.0, 'Lv': 116.0, 'Tl': 81.0, 'Tm': 69.0, 'Lr': 103.0, 'Th': 90.0, 'Ti': 22.0, 'Te': 52.0, 'Tb': 65.0, 'Tc': 43.0, 'Ta': 73.0, 'Yb': 70.0, 'Db': 105.0, 'Dy': 66.0, 'Ds': 110.0, 'I': 53.0, 'U': 92.0, 'Y': 39.0, 'Ac': 89.0, 'Ag': 47.0, 'Uut': 113.0, 'Ir': 77.0, 'Am': 95.0, 'Al': 13.0, 'As': 33.0, 'Ar': 18.0, 'Au': 79.0, 'At': 85.0, 'In': 49.0})
        reader = Parameter('reader', 'auto')
        skip_lines = Parameter('skip_lines', [2,0])

class ConvertFile(object):
    task = 'Input'
    subtask = 'Convert'
    host = 'chemml'
    function ='ConvertFile'
    modules = ('chemml','initialization')
    requirements = (req(0),req(6))
    documentation = "https://openbabel.org/wiki/Babel"

    class Inputs:
        file_path=Input("file_path","the path to the file that needs to be converted",("<class 'str'>","<class 'dict'>"))
    class Outputs:
        converted_file_paths = Output("converted_file_paths", "list of paths to the converted files", "<class 'list'>")
    class WParameters:
        pass
    class FParameters:
        file_path = Parameter('file_path', 'required_required')
        from_format = Parameter('from_format', 'required_required')
        to_format = Parameter('to_format', 'required_required')

class scatter2D(object):
    task='Visualize'
    subtask = 'plot'
    host = 'chemml'
    function = 'scatter2D'
    modules = ('chemml','visualization')
    requirements = (req(0),req(2),req(7))
    documentation = ""

    class Inputs:
        dfx = Input("dfx", "a pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfy = Input("dfy", "a pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))

    class Outputs:
        fig=Output("fig","a matplotlib.Figure object",("<class 'matplotlib.figure.Figure'>",))

    class WParameters:
        pass

    class FParameters:
        x = Parameter('x','required_required')
        y = Parameter('y','required_required')
        color = Parameter('color','b')
        marker = Parameter('marker','.')
        linestyle = Parameter('linestyle','')
        linewidth = Parameter('linewidth', 2)

class hist(object):
    task='Visualize'
    subtask = 'plot'
    host = 'chemml'
    function = 'hist'
    modules = ('chemml','visualization')
    requirements = (req(0),req(2),req(7))
    documentation = ""

    class Inputs:
        dfx = Input("dfx","a pandas dataframe",("<class 'pandas.core.frame.DataFrame'>",))

    class Outputs:
        fig = Output("fig","a matplotlib object",("<class 'matplotlib.figure.Figure'>",))

    class WParameters:
        pass

    class FParameters:
        x = Parameter('x','required_required')
        bins=Parameter('bins',None)
        color=Parameter('color',None)
        kwargs=Parameter('kwargs',{})

class decorator(object):
    task='Visualize'
    subtask = 'artist'
    host = 'chemml'
    function = 'decorator'
    modules = ('chemml','visualization')
    requirements = (req(0),req(2),req(7))
    documentation = ""

    class Inputs:
        fig = Input("fig", "a matplotlib object",
                    ("<class 'matplotlib.figure.Figure'>","<class 'matplotlib.axes._subplots.AxesSubplot'>"))

    class Outputs:
        fig = Output("fig","a matplotlib object",("<class 'matplotlib.figure.Figure'>",))

    class WParameters:
        pass

    class FParameters:
        title = Parameter('title', '')
        xlabel = Parameter('xlabel', '')
        ylabel = Parameter('ylabel', '')
        xlim = Parameter('xlim', (None , None))
        ylim = Parameter('ylim', (None , None))
        grid = Parameter('grid', True)
        grid_color = Parameter('grid_color', 'k')
        grid_linestyle = Parameter('grid_linestyle', '--')
        grid_linewidth = Parameter('grid_linewidth', 0.5)
        family = Parameter('family', 'normal')
        size = Parameter('size', 18)
        weight = Parameter('weight', 'normal')
        style = Parameter('style', 'normal')
        variant = Parameter('variant', 'normal')

class SavePlot(object):
    task='Output'
    subtask = 'figure'
    host = 'chemml'
    function = 'SavePlot'
    modules = ('chemml','visualization')
    requirements = (req(0),req(2),req(7))
    documentation = "https://matplotlib.org/users/index.html"

    class Inputs:
        fig = Input('fig', "a matplotlib object",
                    ("<class 'matplotlib.figure.Figure'>","<class 'matplotlib.axes._subplots.AxesSubplot'>"))

    class Outputs:
        pass

    class WParameters:
        pass

    class FParameters:
        filename=Parameter('filename','required_required')
        output_directory=Parameter('output_directory',None)
        format=Parameter('format','png')
        kwargs = Parameter('kwargs', {})

class GA(object):
    task='Optimize'
    subtask = 'genetic algorithm'
    host = 'chemml'
    function = 'GA'
    modules = ('chemml','optimize')
    requirements = (req(0),req(2),req(10))
    documentation = """Hyperparamter optimization is a time consuming process. The amount of time required to find the best hyperparameters relies on multiple factors which depend on the parameters specified below. Please be patient! 
    Link: https://github.com/hachmannlab/chemml/blob/master/chemml/optimization/genetic_algorithm.py """

    class Inputs:
        # evaluate = Input('evaluate', "a function that receives a list of individuals and returns the score",
                    # ("<type 'function'>",))
        dfx_test = Input("dfx_test","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfy_train = Input("dfy_train","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfy_test = Input("dfy_test","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        dfx_train = Input("dfx_train","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))

    class Outputs:
        best_ind_df = Output('best_ind_df',"pandas dataframe of best individuals after each iteration", ("<class 'pandas.core.frame.DataFrame'>",))
        best_individual = Output('best_individual', "pandas dataframe of the best individual",("<class 'pandas.core.frame.DataFrame'>",))

    class WParameters:
        algorithm = Parameter('algorithm','1','int',
                        description = "A method of the GA class that should be applied",
                        options = (1, 2, 3, 4), required=True)
        ml_model = Parameter('ml_model','MLPRegressor','string',
                        description = "ML model for which hyperparameters need to be optmized",
                        options = ('MLPRegressor', 'More options to be added'),required=True)

    class FParameters:

        #Note: do not change first 4!!!
        evaluate = Parameter('evaluate', 'ga_eval.txt', required=True)
        space = Parameter('space', 'space.txt', required=True)
        error_metric = Parameter('error_metric', 'error_metric.txt', required=True)
        test_hyperparameters = Parameter('test_hyperparameters', 'test_hyperparameters.txt', required=True)
        single_obj = Parameter('single_obj', 'single_obj.txt', required=True)
        fitness = Parameter('fitness','(min,)', 'tuple', required=True)
        pop_size = Parameter('pop_size', 5, 'int', required = True)
        crossover_size = Parameter('crossover_size', 30, 'int', required = True)
        mutation_size = Parameter('mutation_size', 20, 'int', required = True)
        n_splits = Parameter('n_splits',5,'int', required = True)
        crossover_type = Parameter('crossover_type', 'Blend', 'string')
        mutation_prob = Parameter('mutation_prob', 0.4, 'float')
        initial_population = Parameter('initial_population', None, 'list')
        n_generations = Parameter('n_generations', 5, 'int',required = True)
        early_stopping = Parameter('early_stopping',10, 'int')
        init_ratio = Parameter('init_ratio',0.4,'float')
        crossover_ratio = Parameter('crossover_ratio',0.3,'float')
        # algorithm = Parameter('algorithm','1','int',
        #                 description = "A method of the GA class that should be applied",
        #                 options = (1, 2, 3, 4), required=True)
        # n_splits = Parameter('n_splits',5,'int')
        # Weights = Parameter('Weights', (-1.0,))
        # chromosome_length = Parameter('chromosome_length', 1)
        # chromosome_type = Parameter('chromosome_type', (1,))
        # bit_limits = Parameter('bit_limits', ((0, 10),))
        # crossover_prob = Parameter('crossover_prob', 0.4)
        # mutation_prob = Parameter('mutation_prob', 0.4)
        # mut_float_mean = Parameter('mut_float_mean', 0)
        # mut_float_dev = Parameter('mut_float_dev', 1)
        # mut_int_lower = Parameter('mut_int_lower', (1,))
        # mut_int_upper = Parameter('mut_int_upper', (10,))
        

        # init_pop_frac = Parameter('init_pop_frac', 0.35,'float, only for algorithm 2')
        # crossover_pop_frac = Parameter('crossover_pop_frac', 0.35, 'float, only for algorithms 2 and 4')


####################################################
class load_cep_homo(object):
    task = 'Input'
    subtask = 'datasets'
    host = 'chemml'
    function = 'load_cep_homo'
    modules = ('chemml','datasets')
    requirements = (req(0),req(2))
    documentation = ""

    class Inputs:
        pass
    class Outputs:
        smiles = Output("smiles","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        homo = Output("homo","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        pass

class load_organic_density(object):
    task = 'Input'
    subtask = 'datasets'
    host = 'chemml'
    function = 'load_organic_density'
    modules = ('chemml','datasets')
    requirements = (req(0),req(2))
    documentation = ""

    class Inputs:
        pass
    class Outputs:
        smiles = Output("smiles","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        density = Output("density","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        features = Output("features","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        pass

class load_xyz_polarizability(object):
    task = 'Input'
    subtask = 'datasets'
    host = 'chemml'
    function = 'load_xyz_polarizability'
    modules = ('chemml','datasets')
    requirements = (req(0),req(2))
    documentation = ""

    class Inputs:
        pass
    class Outputs:
        coordinates = Output("coordinates","dictionary of molecules represented by their xyz coordinates and atomic numbers",
                             ("<class 'list'>",))
        polarizability = Output("polarizability","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        pass

class load_comp_energy(object):
    task = 'Input'
    subtask = 'datasets'
    host = 'chemml'
    function = 'load_comp_energy'
    modules = ('chemml','datasets')
    requirements = (req(0),req(2))
    documentation = ""

    class Inputs:
        pass
    class Outputs:
        entries = Output("entries","list of entries from "
                                                      "CompositionEntry "
                                  "class.",
                                      ("<class 'list'>",))
        formation_energy = Output("formation_energy","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        pass

class load_crystal_structures(object):
    task = 'Input'
    subtask = 'datasets'
    host = 'chemml'
    function = 'load_crystal_structures'
    modules = ('chemml','datasets')
    requirements = (req(0),req(2))
    documentation = ""

    class Inputs:
        pass
    class Outputs:
        entries = Output("entries","list of entries from "
                                                      "CrystalStructureEntry "
                                  "class.",
                                      ("<class 'list'>",))
    class WParameters:
        pass
    class FParameters:
        pass

####################################################
class APEAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'APEAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CompositionEntry class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        packing_threshold = Parameter('packing_threshold', None)
        n_nearest_to_eval = Parameter('n_nearest_to_eval', None)
        radius_property = Parameter('radius_property', None)

class ChargeDependentAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'ChargeDependentAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CompositionEntry class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        pass

class ElementalPropertyAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'ElementalPropertyAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CompositionEntry class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        elemental_properties = Parameter('elemental_properties', None)

    class FParameters:
        use_default_properties = Parameter('use_default_properties', True)

class ElementFractionAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'ElementFractionAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CompositionEntry class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        pass

class ElementPairPropertyAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'ElementPairPropertyAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CompositionEntry class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        elemental_pair_properties = Parameter('elemental_pair_properties', None)

class GCLPAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'GCLPAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CompositionEntry class.",
                                      ("<class 'list'>",))
        energies = Input("energies","to be passed to the parameter energies",
                         ("<class 'pandas.core.frame.DataFrame'>",))

        phases = Input("phases", "to be passed to the parameter phases",
                     ("<class 'pandas.core.frame.DataFrame'>",))

    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        count_phases = Parameter('count_phases', None)
        phases = Parameter('phases', [], required=True)
        energies = Parameter('energies', [], required=True)

class IonicCompoundProximityAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'IonicCompoundProximityAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CompositionEntry class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        max_formula_unit = Parameter('max_formula_unit', 14)

class IonicityAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'IonicityAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CompositionEntry class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        pass

class MeredigAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'MeredigAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CompositionEntry class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        pass

class StoichiometricAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'StoichiometricAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CompositionEntry class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        use_default_norms = Parameter('use_default_norms', None)
    class FParameters:
        p_norms = Parameter('p_norms', None)

class ValenceShellAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'ValenceShellAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CompositionEntry class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        pass

class YangOmegaAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'YangOmegaAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CompositionEntry class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        pass

class APRDFAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'APRDFAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CrystalStructureEntry "
                                  "class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        cut_off_distance = Parameter('cut_off_distance', 10.0)
        num_points = Parameter('num_points', 6)
        smooth_parameter = Parameter('smooth_parameter', 4.0)
        elemental_properties = Parameter('elemental_properties',
                                         'required_required', required=True)

class ChemicalOrderingAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'ChemicalOrderingAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CrystalStructureEntry "
                                  "class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        shells = Parameter('shells', [1, 2, 3])
        weighted = Parameter('weighted', True)

class CoordinationNumberAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'CoordinationNumberAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CrystalStructureEntry "
                                  "class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        pass

class CoulombMatrixAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'CoulombMatrixAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CrystalStructureEntry "
                                  "class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        n_eigenvalues = Parameter('n_eigenvalues', 30)

class EffectiveCoordinationNumberAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'EffectiveCoordinationNumberAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CrystalStructureEntry "
                                  "class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        pass

class LatticeSimilarityAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'LatticeSimilarityAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CrystalStructureEntry "
                                  "class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        pass

class LocalPropertyDifferenceAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'LocalPropertyDifferenceAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CrystalStructureEntry "
                                  "class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        shells = Parameter('shells', [1])
        elemental_properties = Parameter('elemental_properties',
                                         'required_required', required=True)

class LocalPropertyVarianceAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'LocalPropertyVarianceAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CrystalStructureEntry "
                                  "class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        shells = Parameter('shells', [1])
        elemental_properties = Parameter('elemental_properties',
                                         'required_required', required=True)

class PackingEfficiencyAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'PackingEfficiencyAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CrystalStructureEntry "
                                  "class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        pass

class PRDFAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'PRDFAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CrystalStructureEntry "
                                  "class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        cut_off_distance = Parameter('cut_off_distance', 10.0)
        n_points = Parameter('n_points', 20)

class StructuralHeterogeneityAttributeGenerator(object):
    task = 'Represent'
    subtask = 'inorganic descriptors'
    host = 'chemml'
    function = 'StructuralHeterogeneityAttributeGenerator'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        entries = Input("entries","list of entries from "
                                                      "CrystalStructureEntry "
                                  "class.",
                                      ("<class 'list'>",))
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
    class WParameters:
        pass
    class FParameters:
        pass

class CompositionEntry(object):
    task = 'Represent'
    subtask = 'inorganic input'
    host = 'chemml'
    function = 'CompositionEntry'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        pass
        # filepath = Input("filepath", "the file path", ("<class 'str'>",))

    class Outputs:
        entries = Output("entries","list of entries from "
                                                      "CompositionEntry "
                                  "class.",
                                      ("<class 'list'>",))
    class WParameters:
        pass
    class FParameters:
        filepath = Parameter("filepath", "required_required", required=True)

class CrystalStructureEntry(object):
    task = 'Represent'
    subtask = 'inorganic input'
    host = 'chemml'
    function = 'CrystalStructureEntry'
    modules = ('chemml','chem')
    requirements = (req(0), req(2))
    documentation = ""

    class Inputs:
        pass
        # directory_path = Input("directory_path", "path to the directory containing  VASP files", ("<class 'str'>",))

    class Outputs:
        entries = Output("entries","list of entries from "
                                                      "CrystalStructureEntry "
                                  "class.",
                                      ("<class 'list'>",))
    class WParameters:
        pass
    class FParameters:
        directory_path = Parameter("directory_path", "required_required", required=True)

# class local_features(object):   #newly added (10.21.2020)
#     task = 'Represent'
#     subtask = 'Local Features'
#     host = 'chemml'
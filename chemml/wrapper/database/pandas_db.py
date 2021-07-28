from chemml.wrapper.database.containers import Input, Output, Parameter, req, regression_types, cv_classes

class read_table(object):
    task = 'Input'
    subtask = 'table'
    host = 'pandas'
    function = 'read_table'
    modules = ('pandas','')
    requirements = (req(2),)
    documentation = "https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_table.html"

    class Inputs:
        pass
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))

    class WParameters:
        pass
    class FParameters:
        filepath_or_buffer = Parameter('filepath_or_buffer', 'required_required')
        sep = Parameter('sep', "required_required")
        delimiter = Parameter('delimiter', None)
        header = Parameter('header', 'infer')
        names = Parameter('names', None)
        index_col = Parameter('index_col', None)
        usecols = Parameter('usecols', None)
        squeeze = Parameter('squeeze', False)
        prefix = Parameter('prefix', None)
        mangle_dupe_cols = Parameter('mangle_dupe_cols', True)
        dtype = Parameter('dtype', None)
        engine = Parameter('engine', None)
        converters = Parameter('converters', None)
        true_values = Parameter('true_values', None)
        false_values = Parameter('false_values', None)
        skipinitialspace = Parameter('skipinitialspace', False)
        skiprows = Parameter('skiprows', None)
        nrows = Parameter('nrows', None)
        na_values = Parameter('na_values', None)
        keep_default_na = Parameter('keep_default_na', True)
        na_filter = Parameter('na_filter', True)
        verbose = Parameter('verbose', False)
        skip_blank_lines = Parameter('skip_blank_lines', True)
        parse_dates = Parameter('parse_dates', False)
        infer_datetime_format = Parameter('infer_datetime_format', False)
        keep_date_col = Parameter('keep_date_col', False)
        date_parser = Parameter('date_parser', None)
        dayfirst = Parameter('dayfirst', False)
        iterator = Parameter('iterator', False)
        chunksize = Parameter('chunksize', None)
        compression = Parameter('compression', 'infer')
        thousands = Parameter('thousands', None)
        decimal = Parameter('decimal', '.')
        lineterminator = Parameter('lineterminator', None)
        quotechar = Parameter('quotechar', '"')
        quoting = Parameter('quoting', 0)
        escapechar = Parameter('escapechar', None)
        comment = Parameter('comment', None)
        encoding = Parameter('encoding', None)
        dialect = Parameter('dialect', None)
        tupleize_cols = Parameter('tupleize_cols', None)
        error_bad_lines = Parameter('error_bad_lines', True)
        warn_bad_lines = Parameter('warn_bad_lines', True)
        skipfooter = Parameter('skipfooter', 0)
        skip_footer = Parameter('skip_footer', 0)
        doublequote = Parameter('doublequote', True)
        delim_whitespace = Parameter('delim_whitespace', False)
        as_recarray = Parameter('as_recarray', None)
        compact_ints = Parameter('compact_ints', None)
        use_unsigned = Parameter('use_unsigned', None)
        low_memory = Parameter('low_memory', True)
        buffer_lines = Parameter('buffer_lines', None)
        memory_map = Parameter('memory_map', False)
        float_precision = Parameter('float_precision', None)

class read_excel(object):
    task = 'Input'
    subtask = 'table'
    host = 'pandas'
    function = 'read_excel'
    modules = ('pandas','')
    requirements = (req(2),)
    documentation = "https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_excel.html"

    class Inputs:
        pass
    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))

    class WParameters:
        pass
    class FParameters:
        io = Parameter('io', 'required_required')
        sheet_name = Parameter('sheet_name', 0)
        header = Parameter('header', 0)
        skiprows = Parameter('skiprows', None)
        skip_footer = Parameter('skip_footer', 0)
        index_col = Parameter('index_col', None)
        names = Parameter('names', None)
        usecols = Parameter('usecols', None)
        parse_dates = Parameter('parse_dates', False)
        date_parser = Parameter('date_parser', None)
        na_values = Parameter('na_values', None)
        thousands = Parameter('thousands', None)
        convert_float = Parameter('convert_float', True)
        converters = Parameter('converters', None)
        dtype = Parameter('dtype', None)
        true_values = Parameter('true_values', None)
        false_values = Parameter('false_values', None)
        engine = Parameter('engine', None)
        squeeze = Parameter('squeeze', False)

class corr(object):
    task = 'Optimize'
    subtask = 'evaluate'
    host = 'pandas'
    function = 'corr'
    modules = ('pandas','')
    requirements = (req(2),)
    documentation = "https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html"

    class Inputs:
        df = Input("df", "pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))

    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))

    class WParameters:
        pass
    class FParameters:
        method = Parameter('method', 'pearson')
        min_periods = Parameter('min_periods', 1)

class concat(object):
    task = 'Prepare'
    subtask = 'data manipulation'
    host = 'pandas'
    function = 'concat'
    modules = ('pandas','')
    requirements = (req(2),)
    documentation = "http://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html"

    class Inputs:
        df1 = Input("df1","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        df2 = Input("df2","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))
        df3 = Input("df3", "pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))

    class Outputs:
        df = Output("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))

    class WParameters:
        pass
    class FParameters:
        join = Parameter('join', 'outer')
        axis = Parameter('axis', 0)
        join_axes = Parameter('join_axes', None)
        ignore_index = Parameter('ignore_index', False)
        keys = Parameter('keys', None)
        levels = Parameter('levels', None)
        names = Parameter('names', None)
        verify_integrity = Parameter('verify_integrity', False)
        copy = Parameter('copy', True)

class plot(object):
    task = 'Visualize'
    subtask = 'plot'
    host = 'pandas'
    function = 'plot'
    modules = ('pandas','')
    requirements = (req(2),req(7))
    documentation = "https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html"

    class Inputs:
        df = Input("df","pandas dataframe", ("<class 'pandas.core.frame.DataFrame'>",))

    class Outputs:
        fig = Output("fig","matplotlib figure or axes object",
                      ("<class 'matplotlib.axes._subplots.AxesSubplot'>","<class 'matplotlib.figure.Figure'>"))

    class WParameters:
        pass
    class FParameters:
        x = Parameter('x', None)
        y = Parameter('y', None)
        kind = Parameter('kind', 'line')
        ax = Parameter('ax', None)
        subplots = Parameter('subplots', False)
        sharex = Parameter('sharex', None)
        sharey = Parameter('sharey', False)
        layout = Parameter('layout', None)
        figsize = Parameter('figsize', None)
        use_index = Parameter('use_index', True)
        title = Parameter('title', None)
        grid = Parameter('grid', None)
        legend = Parameter('legend', True)
        style = Parameter('style', None)
        logx = Parameter('logx', False)
        logy = Parameter('logy', False)
        loglog = Parameter('loglog', False)
        xticks = Parameter('xticks', None)
        yticks = Parameter('yticks', None)
        xlim = Parameter('xlim', None)
        ylim = Parameter('ylim', None)
        rot = Parameter('rot', None)
        fontsize = Parameter('fontsize', None)
        colormap = Parameter('colormap', None)
        table = Parameter('table', False)
        yerr = Parameter('yerr', None)
        xerr = Parameter('xerr', None)
        secondary_y = Parameter('secondary_y', False)
        sort_columns = Parameter('sort_columns', False)

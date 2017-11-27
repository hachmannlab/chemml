.. _read_table:

read_table
===========

:task:
    | Enter

:subtask:
    | table

:host:
    | pandas

:function:
    | read_table


:output tokens (senders):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | p, a
    | 0, .

:config file view:
    | ``##``
    |   ``<< host = pandas    << function = read_table``
    |   ``<< comment = None``
    |   ``<< escapechar = None``
    |   ``<< float_precision = None``
    |   ``<< na_filter = True``
    |   ``<< iterator = False``
    |   ``<< sep = required_required``
    |   ``<< mangle_dupe_cols = True``
    |   ``<< skip_blank_lines = True``
    |   ``<< keep_default_na = True``
    |   ``<< false_values = None``
    |   ``<< header = infer``
    |   ``<< prefix = None``
    |   ``<< memory_map = False``
    |   ``<< names = None``
    |   ``<< skipfooter = 0``
    |   ``<< verbose = False``
    |   ``<< compact_ints = None``
    |   ``<< lineterminator = None``
    |   ``<< compression = infer``
    |   ``<< dayfirst = False``
    |   ``<< low_memory = True``
    |   ``<< encoding = None``
    |   ``<< parse_dates = False``
    |   ``<< skip_footer = 0``
    |   ``<< dtype = None``
    |   ``<< quotechar = "``
    |   ``<< thousands = None``
    |   ``<< converters = None``
    |   ``<< warn_bad_lines = True``
    |   ``<< as_recarray = None``
    |   ``<< engine = None``
    |   ``<< dialect = None``
    |   ``<< chunksize = None``
    |   ``<< tupleize_cols = None``
    |   ``<< na_values = None``
    |   ``<< infer_datetime_format = False``
    |   ``<< keep_date_col = False``
    |   ``<< use_unsigned = None``
    |   ``<< nrows = None``
    |   ``<< true_values = None``
    |   ``<< delim_whitespace = False``
    |   ``<< usecols = None``
    |   ``<< squeeze = False``
    |   ``<< buffer_lines = None``
    |   ``<< index_col = None``
    |   ``<< skipinitialspace = False``
    |   ``<< decimal = .``
    |   ``<< skiprows = None``
    |   ``<< filepath_or_buffer = required_required``
    |   ``<< date_parser = None``
    |   ``<< delimiter = None``
    |   ``<< error_bad_lines = True``
    |   ``<< doublequote = True``
    |   ``<< quoting = 0``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_table.html
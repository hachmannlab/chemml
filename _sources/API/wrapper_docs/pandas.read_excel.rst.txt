.. _read_excel:

read_excel
===========

:task:
    | Enter

:subtask:
    | table

:host:
    | pandas

:function:
    | read_excel

:input tokens (receivers):
    |   this block doesn't receive anything

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = pandas    << function = read_excel``
    |   ``<< engine = None``
    |   ``<< squeeze = False``
    |   ``<< index_col = None``
    |   ``<< date_parser = None``
    |   ``<< na_values = None``
    |   ``<< parse_dates = False``
    |   ``<< dtype = None``
    |   ``<< skiprows = None``
    |   ``<< sheet_name = 0``
    |   ``<< header = 0``
    |   ``<< skip_footer = 0``
    |   ``<< convert_float = True``
    |   ``<< names = None``
    |   ``<< io = required_required``
    |   ``<< usecols = None``
    |   ``<< true_values = None``
    |   ``<< false_values = None``
    |   ``<< thousands = None``
    |   ``<< converters = None``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_excel.html

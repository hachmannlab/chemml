.. _plot:

plot
=====

:task:
    | Visualize

:subtask:
    | plot

:host:
    | pandas

:function:
    | plot

:input tokens (receivers):
    | ``df`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``fig`` : matplotlib figure or axes object
    |   types: ("<class 'matplotlib.axes._subplots.AxesSubplot'>", "<class 'matplotlib.figure.Figure'>")


:required packages:
    | pandas, 0.20.3
    | matplotlib, 1.5.1

:config file view:
    | ``##``
    |   ``<< host = pandas    << function = plot``
    |   ``<< xlim = None``
    |   ``<< xerr = None``
    |   ``<< yerr = None``
    |   ``<< logx = False``
    |   ``<< logy = False``
    |   ``<< table = False``
    |   ``<< ax = None``
    |   ``<< rot = None``
    |   ``<< ylim = None``
    |   ``<< style = None``
    |   ``<< sharey = False``
    |   ``<< sharex = None``
    |   ``<< title = None``
    |   ``<< use_index = True``
    |   ``<< xticks = None``
    |   ``<< fontsize = None``
    |   ``<< sort_columns = False``
    |   ``<< loglog = False``
    |   ``<< colormap = None``
    |   ``<< grid = None``
    |   ``<< layout = None``
    |   ``<< legend = True``
    |   ``<< secondary_y = False``
    |   ``<< kind = line``
    |   ``<< subplots = False``
    |   ``<< figsize = None``
    |   ``<< yticks = None``
    |   ``<< y = None``
    |   ``<< x = None``
    |   ``>> id df``
    |   ``>> id fig``
    |
    .. note:: The documentation page for function parameters: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html

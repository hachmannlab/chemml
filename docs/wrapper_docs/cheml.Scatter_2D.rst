.. _Scatter_2D:

Scatter_2D
===========

:task:
    | Visualize

:subtask:
    | Plot

:host:
    | cheml

:function:
    | Scatter_2D

:input tokens (receivers):
    | ``df4`` : a pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``df1`` : a pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``df3`` : a pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``df2`` : a pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``fig`` : a matplotlib object
    |   types: ("<class 'matplotlib.figure.Figure'>",)


:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3
    | matplotlib, 1.5.1

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = Scatter_2D``
    |   ``<< ymax = 0``
    |   ``<< xheader = ['x']``
    |   ``<< title = Plot``
    |   ``<< nod = 1``
    |   ``<< kwargs = {}``
    |   ``<< lpos = Best``
    |   ``<< subplots = [1, 1, 1]``
    |   ``<< xmin = 0``
    |   ``<< legend_titles = []``
    |   ``<< xlabel = x``
    |   ``<< xmax = 0``
    |   ``<< ylabel = y``
    |   ``<< sc = []``
    |   ``<< yheader = ['y']``
    |   ``<< ymin = 0``
    |   ``<< legend = False``
    |   ``>> id df4``
    |   ``>> id df1``
    |   ``>> id df3``
    |   ``>> id df2``
    |   ``>> id fig``
    |
    .. note:: The documentation page for function parameters: https://matplotlib.org/users/index.html

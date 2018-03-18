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
    | ``dfy1`` : a pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx4`` : a pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfy2`` : a pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx2`` : a pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx3`` : a pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx1`` : a pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfy3`` : a pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfy4`` : a pandas dataframe
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
    |   ``<< kwargs = {}``
    |   ``<< l_pos = Best``
    |   ``<< xmin = 0``
    |   ``<< legend_titles = []``
    |   ``<< xlabel = x``
    |   ``<< xmax = 0``
    |   ``<< ylabel = y``
    |   ``<< sc = []``
    |   ``<< yheader = ['y']``
    |   ``<< ymin = 0``
    |   ``<< legend = False``
    |   ``>> id dfy1``
    |   ``>> id dfx4``
    |   ``>> id dfy2``
    |   ``>> id dfx2``
    |   ``>> id dfx3``
    |   ``>> id dfx1``
    |   ``>> id dfy3``
    |   ``>> id dfy4``
    |   ``>> id fig``
    |
    .. note:: The documentation page for function parameters: https://matplotlib.org/users/index.html

.. _scatter2D:

scatter2D
==========

:task:
    | Visualize

:subtask:
    | plot

:host:
    | cheml

:function:
    | scatter2D

:input tokens (receivers):
    | ``dfy`` : a pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfx`` : a pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``fig`` : a matplotlib.Figure object
    |   types: ("<class 'matplotlib.figure.Figure'>",)


:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3
    | matplotlib, 1.5.1

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = scatter2D``
    |   ``<< color = b``
    |   ``<< marker = .``
    |   ``<< y = required_required``
    |   ``<< x = required_required``
    |   ``<< linewidth = 2``
    |   ``<< linestyle = ``
    |   ``>> id dfy``
    |   ``>> id dfx``
    |   ``>> id fig``
    |
    .. note:: The documentation page for function parameters: 

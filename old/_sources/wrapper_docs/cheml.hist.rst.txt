.. _hist:

hist
=====

:task:
    | Visualize

:subtask:
    | plot

:host:
    | cheml

:function:
    | hist

:input tokens (receivers):
    | ``dfx`` : a pandas dataframe
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
    |   ``<< host = cheml    << function = hist``
    |   ``<< color = None``
    |   ``<< kwargs = {}``
    |   ``<< x = required_required``
    |   ``<< bins = None``
    |   ``>> id dfx``
    |   ``>> id fig``
    |
    .. note:: The documentation page for function parameters: 

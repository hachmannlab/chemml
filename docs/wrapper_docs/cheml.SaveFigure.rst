.. _SaveFigure:

SaveFigure
===========

:task:
    | Store

:subtask:
    | figure

:host:
    | cheml

:function:
    | SaveFigure

:input tokens (receivers):
    | ``fig`` : a matplotlib object
    |   types: ("<class 'matplotlib.figure.Figure'>",)

:input tokens (receivers):
    |   this block doesn't send anything


:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3
    | matplotlib, 1.5.1

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = SaveFigure``
    |   ``<< format = png``
    |   ``<< output_directory = None``
    |   ``<< kwargs = {}``
    |   ``<< filename = required_required``
    |   ``>> id fig``
    |
    .. note:: The documentation page for function parameters: https://matplotlib.org/users/index.html

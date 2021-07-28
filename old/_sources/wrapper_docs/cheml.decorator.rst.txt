.. _decorator:

decorator
==========

:task:
    | Visualize

:subtask:
    | artist

:host:
    | cheml

:function:
    | decorator

:input tokens (receivers):
    | ``fig`` : a matplotlib object
    |   types: ("<class 'matplotlib.figure.Figure'>", "<class 'matplotlib.axes._subplots.AxesSubplot'>")

:output tokens (senders):
    | ``fig`` : a matplotlib object
    |   types: ("<class 'matplotlib.figure.Figure'>",)


:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3
    | matplotlib, 1.5.1

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = decorator``
    |   ``<< weight = normal``
    |   ``<< family = normal``
    |   ``<< xlim = (None, None)``
    |   ``<< title = ``
    |   ``<< grid_color = k``
    |   ``<< variant = normal``
    |   ``<< style = normal``
    |   ``<< grid_linestyle = --``
    |   ``<< xlabel = ``
    |   ``<< grid_linewidth = 0.5``
    |   ``<< ylabel = ``
    |   ``<< grid = True``
    |   ``<< ylim = (None, None)``
    |   ``<< size = 18``
    |   ``>> id fig``
    |   ``>> id fig``
    |
    .. note:: The documentation page for function parameters: 

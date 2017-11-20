.. _PyScript:

PyScript
=========

:task:
    | Prepare

:subtask:
    | python script

:host:
    | cheml

:function:
    | PyScript

:input tokens (receivers):
    | ``iv4`` : input variable, of any format
    |   ()
    | ``iv5`` : input variable, of any format
    |   ()
    | ``iv6`` : input variable, of any format
    |   ()
    | ``iv1`` : input variable, of any format
    |   ()
    | ``iv2`` : input variable, of any format
    |   ()
    | ``iv3`` : input variable, of any format
    |   ()

:output tokens (senders):
    | ``ov2`` : output variable, of any format
    |   ()
    | ``ov3`` : output variable, of any format
    |   ()
    | ``ov1`` : output variable, of any format
    |   ()
    | ``ov6`` : output variable, of any format
    |   ()
    | ``ov4`` : output variable, of any format
    |   ()
    | ``ov5`` : output variable, of any format
    |   ()


:required packages:
    | ChemML, 0.1.0
    | pandas, 0.20.3

:config file view:
    | ``## ``
    |   ``<< host = cheml    << function = PyScript``
    |   ``>> id iv4``
    |   ``>> id iv5``
    |   ``>> id iv6``
    |   ``>> id iv1``
    |   ``>> id iv2``
    |   ``>> id iv3``
    |   ``>> id ov2``
    |   ``>> id ov3``
    |   ``>> id ov1``
    |   ``>> id ov6``
    |   ``>> id ov4``
    |   ``>> id ov5``
    |
    .. note:: The documentation page for function parameters: 
.. _Bag_of_Bonds:

Bag_of_Bonds
=============

:task:
    | Prepare

:subtask:
    | feature representation

:host:
    | cheml

:function:
    | Bag_of_Bonds

:input tokens (receivers):
    | ``molecules`` : the molecule numpy array or data frame
    |   ("<class 'pandas.core.frame.DataFrame'>", "<type 'numpy.ndarray'>", "<type 'dict'>")

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | ChemML, 0.1.0
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = cheml    << function = Bag_of_Bonds``
    |   ``<< const = 1``
    |   ``>> id molecules``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: 
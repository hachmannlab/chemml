.. _Binarizer:

Binarizer
==========

:task:
    | Prepare

:subtask:
    | feature representation

:host:
    | sklearn

:function:
    | Binarizer

:input tokens (receivers):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``api`` : instance of scikit-learn's Binarizer class
    |   ("<class 'sklearn.preprocessing.data.Binarizer'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``api`` : instance of scikit-learn's Binarizer class
    |   ("<class 'sklearn.preprocessing.data.Binarizer'>",)

:wrapper parameters:
    | ``track_header`` : Boolean, (default:True)
    |   if True, the input dataframe's header will be transformed to the output dataframe
    |   choose one of: (True, False)
    | ``func_method`` : string, (default:None)
    |   fit_transform: always make a new api; transform: must receive an api; None: only make a new api 
    |   choose one of: ('fit_transform', 'transform', None)

:required packages:
    | scikit-learn, 0.19.0
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = sklearn    << function = Binarizer``
    |   ``<< track_header = True``
    |   ``<< func_method = None``
    |   ``<< threshold = 0.0``
    |   ``<< copy = True``
    |   ``>> id df``
    |   ``>> id api``
    |   ``>> id df``
    |   ``>> id api``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer
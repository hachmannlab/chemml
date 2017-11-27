.. _RobustScaler:

RobustScaler
=============

:task:
    | Prepare

:subtask:
    | scale

:host:
    | sklearn

:function:
    | RobustScaler

:input tokens (receivers):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``api`` : instance of scikit-learn's RobustScaler class
    |   ("<class 'sklearn.preprocessing.data.RobustScaler'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``api`` : instance of scikit-learn's RobustScaler class
    |   ("<class 'sklearn.preprocessing.data.RobustScaler'>",)

:wrapper parameters:
    | ``track_header`` : Boolean, (default:True)
    |   if True, the input dataframe's header will be transformed to the output dataframe
    |   choose one of: (True, False)
    | ``func_method`` : string, (default:None)
    |   fit_transform: always make a new api; transform: must receive an api; inverse_transform: must receive an api; None: only make a new api 
    |   choose one of: ('fit_transform', 'transform', 'inverse_transform', None)

:required packages:
    | scikit-learn, 0.19.0
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = sklearn    << function = RobustScaler``
    |   ``<< track_header = True``
    |   ``<< func_method = None``
    |   ``<< copy = True``
    |   ``<< with_scaling = True``
    |   ``<< with_centering = True``
    |   ``<< quantile_range = (25.0, 75.0)``
    |   ``>> id df``
    |   ``>> id api``
    |   ``>> id df``
    |   ``>> id api``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler
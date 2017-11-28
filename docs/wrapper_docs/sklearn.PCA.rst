.. _PCA:

PCA
====

:task:
    | Prepare

:subtask:
    | feature transformation

:host:
    | sklearn

:function:
    | PCA

:input tokens (receivers):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``api`` : instance of scikit-learn's PCA class
    |   ("<class 'sklearn.decomposition.pca.PCA'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``api`` : instance of scikit-learn's PCA class
    |   ("<class 'sklearn.decomposition.pca.PCA'>",)

:wrapper parameters:
    | ``track_header`` : Boolean, (default:False)
    |   Always False, the header of input dataframe is not equivalent with the transformed dataframe
    |   choose one of: False
    | ``func_method`` : string, (default:None)
    |   fit_transform: always make a new api; transform: must receive an api; inverse_transform: must receive an api; None: only make a new api 
    |   choose one of: ('fit_transform', 'transform', 'inverse_transform', None)

:required packages:
    | scikit-learn, 0.19.0
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = sklearn    << function = PCA``
    |   ``<< track_header = False``
    |   ``<< func_method = None``
    |   ``<< svd_solver = auto``
    |   ``<< iterated_power = auto``
    |   ``<< random_state = None``
    |   ``<< whiten = False``
    |   ``<< tol = 0.0``
    |   ``<< copy = True``
    |   ``<< n_components = None``
    |   ``>> id df``
    |   ``>> id api``
    |   ``>> id df``
    |   ``>> id api``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
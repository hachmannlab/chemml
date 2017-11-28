.. _KernelPCA:

KernelPCA
==========

:task:
    | Prepare

:subtask:
    | feature transformation

:host:
    | sklearn

:function:
    | KernelPCA

:input tokens (receivers):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``api`` : instance of scikit-learn's KernelPCA class
    |   ("<class 'sklearn.decomposition.kernel_pca.KernelPCA'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``api`` : instance of scikit-learn's KernelPCA class
    |   ("<class 'sklearn.decomposition.kernel_pca.KernelPCA'>",)

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
    |   ``<< host = sklearn    << function = KernelPCA``
    |   ``<< track_header = False``
    |   ``<< func_method = None``
    |   ``<< fit_inverse_transform = False``
    |   ``<< kernel = linear``
    |   ``<< n_jobs = 1``
    |   ``<< eigen_solver = auto``
    |   ``<< degree = 3``
    |   ``<< max_iter = None``
    |   ``<< copy_X = True``
    |   ``<< kernel_params = None``
    |   ``<< random_state = None``
    |   ``<< n_components = None``
    |   ``<< remove_zero_eig = False``
    |   ``<< tol = 0``
    |   ``<< alpha = 1.0``
    |   ``<< coef0 = 1``
    |   ``<< gamma = None``
    |   ``>> id df``
    |   ``>> id api``
    |   ``>> id df``
    |   ``>> id api``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA
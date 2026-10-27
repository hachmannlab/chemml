.. _scorer_regression:

scorer_regression
==================

:task:
    | Search

:subtask:
    | evaluate

:host:
    | sklearn

:function:
    | scorer_regression

:input tokens (receivers):
    |   this block doesn't receive anything

:output tokens (senders):
    | ``scorer`` : Callable object that returns a scalar score
    |   types: ("<class 'sklearn.metrics.scorer._PredictScorer'>",)

:wrapper parameters:
    | ``track_header`` : Boolean, (default:True)
    |   if True, the input dataframe's header will be transformed to the output dataframe
    |   choose one of: (True, False)
    | ``metric`` : string: 'mae', 'mse', 'r2', (default:mae)
    |   http://scikit-learn.org/dev/modules/model_evaluation.html#regression-metrics
    |   choose one of: ('mae', 'mse', 'r2')

:required packages:
    | scikit-learn, 0.19.0
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = sklearn    << function = scorer_regression``
    |   ``<< track_header = True``
    |   ``<< metric = mae``
    |   ``<< greater_is_better = True``
    |   ``<< needs_threshold = False``
    |   ``<< needs_proba = False``
    |   ``<< kwargs = {}``
    |   ``>> id scorer``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/0.15/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer

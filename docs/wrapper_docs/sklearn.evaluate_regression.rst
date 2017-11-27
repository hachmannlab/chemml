.. _evaluate_regression:

evaluate_regression
====================

:task:
    | Search

:subtask:
    | evaluate

:host:
    | sklearn

:function:
    | evaluate_regression

:input tokens (receivers):
    | ``dfy`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``dfy_predict`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)

:output tokens (senders):
    | ``evaluation_results_`` : pandas dataframe
    |   ("<class 'pandas.core.frame.DataFrame'>",)
    | ``evaluator`` : dictionary of metrics and their score function
    |   ("<type 'dict'>",)

:wrapper parameters:
    | ``mae_multioutput`` : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error, (default:uniform_average)
    |   
    |   choose one of: ('raw_values', 'uniform_average')
    | ``r2_score`` : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score, (default:False)
    |   
    |   choose one of: (True, False)
    | ``mean_absolute_error`` : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error, (default:False)
    |   
    |   choose one of: (True, False)
    | ``multioutput`` : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score, (default:uniform_average)
    |   
    |   choose one of: ('raw_values', 'uniform_average', 'variance_weighted')
    | ``r2_sample_weight`` : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score, (default:None)
    |   
    |   choose one of: []
    | ``rmse_multioutput`` : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error, (default:uniform_average)
    |   
    |   choose one of: ('raw_values', 'uniform_average')
    | ``median_absolute_error`` : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error, (default:False)
    |   
    |   choose one of: (True, False)
    | ``mae_sample_weight`` : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error, (default:None)
    |   
    |   choose one of: []
    | ``rmse_sample_weight`` : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error, (default:None)
    |   
    |   choose one of: []
    | ``track_header`` : Boolean, (default:True)
    |   if True, the input dataframe's header will be transformed to the output dataframe
    |   choose one of: (True, False)
    | ``mean_squared_error`` : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error, (default:False)
    |   
    |   choose one of: (True, False)
    | ``root_mean_squared_error`` : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error, (default:False)
    |   
    |   choose one of: (True, False)
    | ``explained_variance_score`` : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score, (default:False)
    |   
    |   choose one of: (True, False)
    | ``mse_sample_weight`` : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error, (default:None)
    |   
    |   choose one of: []
    | ``ev_sample_weight`` : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score, (default:None)
    |   
    |   choose one of: []
    | ``ev_multioutput`` : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score, (default:uniform_average)
    |   
    |   choose one of: ('raw_values', 'uniform_average', 'variance_weighted')
    | ``mse_multioutput`` : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error, (default:uniform_average)
    |   
    |   choose one of: ('raw_values', 'uniform_average')

:required packages:
    | scikit-learn, 0.19.0
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = sklearn    << function = evaluate_regression``
    |   ``<< mae_multioutput = uniform_average``
    |   ``<< r2_score = False``
    |   ``<< mean_absolute_error = False``
    |   ``<< multioutput = uniform_average``
    |   ``<< r2_sample_weight = None``
    |   ``<< rmse_multioutput = uniform_average``
    |   ``<< median_absolute_error = False``
    |   ``<< mae_sample_weight = None``
    |   ``<< rmse_sample_weight = None``
    |   ``<< track_header = True``
    |   ``<< mean_squared_error = False``
    |   ``<< root_mean_squared_error = False``
    |   ``<< explained_variance_score = False``
    |   ``<< mse_sample_weight = None``
    |   ``<< ev_sample_weight = None``
    |   ``<< ev_multioutput = uniform_average``
    |   ``<< mse_multioutput = uniform_average``
    |   ``>> id dfy``
    |   ``>> id dfy_predict``
    |   ``>> id evaluation_results_``
    |   ``>> id evaluator``
    |
    .. note:: The documentation page for function parameters: http://scikit-learn.org/dev/modules/model_evaluation.html#regression-metrics
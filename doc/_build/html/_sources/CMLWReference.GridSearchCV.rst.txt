.. _GridSearchCV:

GridSearchCV
============

:task:
    | Define Search

:host:
    | sklearn

:function:
    | GridSearchCV

:parameters:
    | estimator
    | param_grid
    | scoring
    | kernel
    | fit_params
    | n_jobs
    | pre_dispatch
    | iid
    | cv
    | refit
    | verbose
    | error_score
    | return_train_score
    |
    .. note:: The documentation for parameters can be found here_.
    .. _here: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV

:send tokens:
    | ``best_estimator_`` : sklearn estimator
    |   \best_estimator_ attribute in the GridSearchCV function from sklearn: find it here_
    | ``cv_results_`` : pandas data frame
    |   \cv_results_ attribute in the GridSearchCV function from sklearn: find it here_
    | ``api`` : The interface for GridSearchCV function from sklearn

:receive tokens:
    | ``dfx`` : pandas data frame, shape(n_samples, n_features), requied
    |   feature values matrix
    | ``dfy`` : pandas data frame, shape(n_samples,), requied
    |   target values matrix
    | ``estimator`` : sklearn estimator, required
    |   will be passed to the 'estimator' parameter

:requirements:
    | Scikit-learn_, version: 0.18.1

    .. _Scikit-learn: http://scikit-learn.org/stable/index.html

:input file view:
    | ``## Define Search``
    |   ``<< host = sklearn``
    |   ``<< function = Grid_SearchCV``
    |   ``<< estimator = '@estimator'``
    |   ``<< param_grid = {}``
    |   ``<< scoring = None``
    |   ``<< kernel = 'rbf'``
    |   ``<< fit_params = None``
    |   ``<< n_jobs = 1``
    |   ``<< pre_dispatch = '2*n_jobs'``
    |   ``<< iid = True``
    |   ``<< cv = None``
    |   ``<< refit = True``
    |   ``<< verbose = 0``
    |   ``<< error_score = 'raise'``
    |   ``<< return_train_score = True``
    |   ``>> id dfx    >> id dfy    >> id estimator``
    |   ``>> best_estimator_ id    >> cv_results_ id    >> api id``

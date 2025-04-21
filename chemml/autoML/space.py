import numpy as np

space_models = {
                'MLPRegressor':[
                                {'alpha': {'uniform': [np.log(0.0001), np.log(0.1)],                
                                'mutation': [0, 1]}}, 
                                {'activation': {'choice': ['identity', 'logistic', 'tanh', 'relu']}},
                                {'neurons1':  {'choice': range(0,220,20)}},
                                {'neurons2':  {'choice': range(0,220,20)}},
                                {'neurons3':  {'choice': range(0,220,20)}}
                                ],
                'GradientBoostingRegressor':[
                                {'loss': {'choice': ['squared_error', 'absolute_error', 'huber', 'quantile']}},
                                # {'learning_rate': {'choice': np.arange(0,2000,50).tolist()}},
                                {'n_estimators': {'choice': np.random.randint(100,4000,size=10).tolist()}},
                                {'min_samples_split': {'choice': range(2,50,10)}},
                                {'min_samples_leaf': {'choice': range(10,100,10)}},
                                {'dummy': {'uniform': [np.log(0.0001), np.log(0.1)],                
                                'mutation': [0, 1]}},             
                                ],

                'RandomForestRegressor':[
                                {'n_estimators': {'choice': np.random.randint(1,400,size=10).tolist()}},
                                {'criterion': {'choice': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']}},
                                {'min_samples_split': {'choice': range(2,50,10)}},
                                {'min_samples_leaf': {'choice': range(1,100,10)}},
                                {'dummy': {'uniform': [np.log(0.0001), np.log(0.1)],                
                                'mutation': [0, 1]}},
                                ],

                'Ridge':[
                                {'alpha': {'choice': np.arange(0.1,200,4.9).tolist()}},
                                {'dummy': {'uniform': [np.log(0.0001), np.log(0.1)],                
                                'mutation': [0, 1]}}
                        ],

                'Lasso':[
                                {'alpha': {'uniform': [np.log(0.0001), np.log(0.1)],                
                                        'mutation': [0, 1]}},
                                {'dummy': {'choice': ['auto', 'svd', 'eigen']}}
                        ],

                'SVR':  [
                                {'kernel': {'choice': ['linear','rbf','poly']}},
                                {'C': {'uniform': [1,100], 
                                        'mutation': [0, 0.5]}}
                        ],

                'ElasticNet':[
                                {'alpha': {'uniform': [np.log(0.0001), np.log(0.1)],                
                                        'mutation': [0, 1]}},
                                {'l1_ratio': {'choice': np.arange(0.4,0.8,0.1).tolist()}}
                        ],

                'DecisionTreeRegressor':[
                                {'criterion': {'choice': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']}},
                                {'splitter': {'choice': ['best', 'random']}},
                                {'min_samples_split': {'choice': range(2,50,10)}},
                                {'min_samples_leaf': {'choice': range(1,100,10)}},
                                {'dummy': {'uniform': [np.log(0.0001), np.log(0.1)],                
                                'mutation': [0, 1]}},
                                ],
                # 'XGBRegressor':[
                #                 {'n_estimators': {'choice': np.random.randint(100,200,size=10).tolist()}},
                #                 {'reg_alpha': {'uniform': [np.log(0.0001), np.log(0.1)],                
                #                 'mutation': [0, 1]}}, 
                #                 {'reg_lambda': {'uniform': [np.log(0.0001), np.log(0.1)],                
                #                 'mutation': [0, 1]}},
                # ]
                }

'''
# XGBoost Hyper Parameter Optimization - Copilot data
hyperparameters = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],  # Typical values range from 0.01 to 0.3[^1^][4]
    'max_depth': [3, 5, 7, 9],  # Maximum depth of a tree
    'min_child_weight': [1, 3, 5],  # Minimum sum of instance weight needed in a child
    'gamma': [0, 0.1, 0.2],  # Minimum loss reduction required to make a further partition on a leaf node of the tree
    'colsample_bytree': [0.3, 0.5, 0.7, 1],  # Subsample ratio of columns when constructing each tree
    'n_estimators': [100, 200, 300, 400, 500],  # Number of gradient boosted trees. Equivalent to number of boosting rounds
    'subsample': [0.5, 0.7, 1],  # Subsample ratio of the training instances
    'reg_alpha': [0, 0.5, 1],  # L1 regularization term on weights
    'reg_lambda': [1, 1.5, 2]  # L2 regularization term on weights
}
LightGBM Regressor hyperparameter space
hyperparameters = {
    'num_leaves': [31, 50, 70, 90],  # Maximum tree leaves for base learners
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [-1, 5, 10, 15],  # Maximum tree depth for base learners
    'min_data_in_leaf': [10, 20, 30, 40, 50],  # Minimum number of data in one leaf
    'bagging_fraction': [0.5, 0.7, 0.9, 1]  # Subsample ratio of the training instance
}

'''

space_models_classifiers = {
                "LogisticRegression": [
                        {'C': {'choice': np.linspace(start=0.1, stop=100, num=20, endpoint=True).tolist()}},
                        {'fit_intercept': {'choice': [True, False]}},
                        {'solver': {'choice': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']}},
                        {'dummy': {'uniform': [np.log(0.0001), np.log(0.1)],             
                        'mutation': [0, 1]}}                                              
                        ], 

                "DecisionTreeClassifier": [
                        {"criterion": {"choice": ["gini", "entropy"]}},
                        {"splitter": {"choice": ["best", "random"]}},
                        {"min_samples_split": {"choice": range(2,10)}},
                        {'dummy': {'uniform': [np.log(0.0001), np.log(0.1)],                
                                'mutation': [0, 1]}}
                        ],
                
                "RandomForestClassifier": [
                        {"n_estimators": {"choice": range(10,200)}},
                        {"criterion": {"choice": ["gini", "entropy"]}},
                        {'dummy': {'uniform': [np.log(0.0001), np.log(0.1)],                
                                'mutation': [0, 1]}}
                        ],
                
                "SVC": [
                        {'C': {'uniform': [np.log(0.0001), np.log(0.1)],                
                                'mutation': [0, 1]}},
                        {"kernel": {"choice": ["linear", "poly", "rbf", "sigmoid"]}},
                        ],
                
                "KNeighborsClassifier": [
                        {"n_neighbors": {"choice": range(2,100)}},
                        {"weights": {"choice": ["uniform", "distance"]}},
                        {'dummy': {'uniform': [np.log(0.0001), np.log(0.1)],                
                                'mutation': [0, 1]}}
                        ],
                }


'''
XGBoost Classifier hyperparameter space
hyperparameters = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'colsample_bytree': [0.3, 0.5, 0.7, 1],
    'n_estimators': [100, 200, 300, 400, 500],
    'subsample': [0.5, 0.7, 1],
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [1, 1.5, 2]
}
LightGBM Classifier hyperparameter space
hyperparameters = {
    'num_leaves': [31, 50, 70, 90],  # Maximum tree leaves for base learners
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [-1, 5, 10, 15],  # Maximum tree depth for base learners
    'min_data_in_leaf': [10, 20, 30, 40, 50],  # Minimum number of data in one leaf
    'bagging_fraction': [0.5, 0.7, 0.9, 1]  # Subsample ratio of the training instance
}
'''
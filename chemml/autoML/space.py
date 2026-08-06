import numpy as np

space_models = {
                'single_core':{
                
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
                                {'criterion': {'choice': ['squared_error', 'absolute_error']}}, # Note from v1.3.4: Removing poisson temporarily due to incompatibility with negative y values; root cause uncertain
                                {'splitter': {'choice': ['best', 'random']}},
                                {'min_samples_split': {'choice': range(2,50,10)}},
                                {'min_samples_leaf': {'choice': range(1,100,10)}},
                                {'dummy': {'uniform': [np.log(0.0001), np.log(0.1)],                
                                'mutation': [0, 1]}},
                                ],
                },
                
                'multi_core':{
                
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
                                {'n_estimators': {'choice': np.random.randint(100,4000,size=10).tolist()}},
                                {'min_samples_split': {'choice': range(2,50,10)}},
                                {'min_samples_leaf': {'choice': range(10,100,10)}},
                                {'dummy': {'uniform': [np.log(0.0001), np.log(0.1)],                
                                'mutation': [0, 1]}},             
                                ],

                'RandomForestRegressor':[
                                {'n_estimators': {'choice': np.random.randint(1,400,size=10).tolist()}},
                                {'criterion': {'choice': ['squared_error', 'absolute_error']}}, # Note from v1.3.4: Removing poisson temporarily due to incompatibility with negative y values; root cause uncertain
                                {'min_samples_split': {'choice': range(2,50,10)}},
                                {'min_samples_leaf': {'choice': range(1,100,10)}},
                                {'dummy': {'uniform': [np.log(0.0001), np.log(0.1)],                
                                'mutation': [0, 1]}},
                                ],
                'XGBRegressor':[
                                {'n_estimators': {'choice': np.arange(20,500,10).tolist()}},
                                {'reg_alpha': {'uniform': [np.log(0.0001), np.log(1)],                
                                'mutation': [0, 1]}}, 
                                {'reg_lambda': {'uniform': [np.log(0.0001), np.log(1)],                
                                'mutation': [0, 1]}},
                                {'max_depth': {'choice': range(3,10)}},
                                {'learning_rate': {'uniform': [np.log(0.01), np.log(0.3)],                
                                'mutation': [0, 1]}},
                                {'colsample_bytree': {'uniform': [np.log(0.3), np.log(1)],                
                                'mutation': [0, 1]}},
                                {'subsample': {'uniform': [np.log(0.5), np.log(1)],                
                                'mutation': [0, 1]}},
                                {'gamma': {'uniform': [np.log(0.0001), np.log(0.2)],                
                                'mutation': [0, 1]}},
                                {'min_child_weight': {'choice': range(1,10)}},
                                ],
                'LGBMRegressor':[
                                {'n_estimators': {'choice': np.arange(20,500,10).tolist()}},
                                {'num_leaves': {'choice': [31, 50, 70, 90]}},
                                {'learning_rate': {'uniform': [np.log(0.01), np.log(0.3)],                
                                'mutation': [0, 1]}},
                                {'max_depth': {'choice': [-1, 5, 10, 15]}},
                                {'min_child_samples': {'choice': [10, 20, 30, 40, 50]}},
                                {'subsample': {'uniform': [np.log(0.5), np.log(1)],                
                                'mutation': [0, 1]}},
                                {'colsample_bytree': {'uniform': [np.log(0.3), np.log(1)],                
                                'mutation': [0, 1]}},
                                {'reg_alpha': {'uniform': [np.log(0.0001), np.log(1)],                
                                'mutation': [0, 1]}},
                                {'reg_lambda': {'uniform': [np.log(0.0001), np.log(1)],                
                                'mutation': [0, 1]}},
                                ],
                'MLP':[
                        {'engine': {'choice': ['tensorflow', 'pytorch']}},
                        {'alpha': {'uniform': [np.log(0.0001), np.log(0.1)], 'mutation': [0, 1]}}, 
                        {'neurons1':  {'choice': range(0,220,20)}},
                        {'neurons2':  {'choice': range(0,220,20)}},
                        {'neurons3':  {'choice': range(0,220,20)}},
                        {'activation1': {'choice': ['linear', 'sigmoid', 'tanh', 'ReLU']}},
                        {'activation2': {'choice': ['linear', 'sigmoid', 'tanh', 'ReLU']}},
                        {'activation3': {'choice': ['linear', 'sigmoid', 'tanh', 'ReLU']}},
                        {'nepochs': {'choice': np.arange(20,500,10).tolist()}},
                        {'batch_size': {'choice': np.arange(20,500,10).tolist()}},
                        {'opt_config': {'choice': ['adam', 'sgd']}},
                        {'learning_rate': {'uniform': [np.log(0.0001), np.log(0.1)], 'mutation': [0, 1]}},
                ]
                }
                }


space_models_classifiers = {
                'single_core': {

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
                },

                'multi_core': {

                "RandomForestClassifier": [
                        {"n_estimators": {"choice": range(10,200)}},
                        {"criterion": {"choice": ["gini", "entropy"]}},
                        {'dummy': {'uniform': [np.log(0.0001), np.log(0.1)],                
                                'mutation': [0, 1]}}
                        ],

                'XGBClassifier':[
                                {'n_estimators': {'choice': np.arange(20,500,10).tolist()}},
                                {'reg_alpha': {'uniform': [np.log(0.0001), np.log(1)],                
                                'mutation': [0, 1]}}, 
                                {'reg_lambda': {'uniform': [np.log(0.0001), np.log(1)],                
                                'mutation': [0, 1]}},
                                {'max_depth': {'choice': range(3,10)}},
                                {'learning_rate': {'uniform': [np.log(0.01), np.log(0.3)],                
                                'mutation': [0, 1]}},
                                {'colsample_bytree': {'uniform': [np.log(0.3), np.log(1)],                
                                'mutation': [0, 1]}},
                                {'subsample': {'uniform': [np.log(0.5), np.log(1)],                
                                'mutation': [0, 1]}},
                                {'gamma': {'uniform': [np.log(0.0001), np.log(0.2)],                
                                'mutation': [0, 1]}},
                                {'min_child_weight': {'choice': range(1,10)}},
                                ],

                'LGBMClassifier':[
                                {'n_estimators': {'choice': np.arange(20,500,10).tolist()}},
                                {'num_leaves': {'choice': [31, 50, 70, 90]}},
                                {'learning_rate': {'uniform': [np.log(0.01), np.log(0.3)],                
                                'mutation': [0, 1]}},
                                {'max_depth': {'choice': [-1, 5, 10, 15]}},
                                {'min_child_samples': {'choice': [10, 20, 30, 40, 50]}},
                                {'subsample': {'uniform': [np.log(0.5), np.log(1)],                
                                'mutation': [0, 1]}},
                                {'colsample_bytree': {'uniform': [np.log(0.3), np.log(1)],                
                                'mutation': [0, 1]}},
                                {'reg_alpha': {'uniform': [np.log(0.0001), np.log(1)],                
                                'mutation': [0, 1]}},
                                {'reg_lambda': {'uniform': [np.log(0.0001), np.log(1)],                
                                'mutation': [0, 1]}},
                                ],

                'MLP':[
                        {'engine': {'choice': ['tensorflow', 'pytorch']}},
                        {'alpha': {'uniform': [np.log(0.0001), np.log(0.1)], 'mutation': [0, 1]}}, 
                        {'neurons1':  {'choice': range(0,220,20)}},
                        {'neurons2':  {'choice': range(0,220,20)}},
                        {'neurons3':  {'choice': range(0,220,20)}},
                        {'activation1': {'choice': ['linear', 'sigmoid', 'tanh', 'ReLU']}},
                        {'activation2': {'choice': ['linear', 'sigmoid', 'tanh', 'ReLU']}},
                        {'activation3': {'choice': ['linear', 'sigmoid', 'tanh', 'ReLU']}},
                        {'nepochs': {'choice': np.arange(20,500,10).tolist()}},
                        {'batch_size': {'choice': np.arange(20,500,10).tolist()}},
                        {'opt_config': {'choice': ['adam', 'sgd']}},
                        {'learning_rate': {'uniform': [np.log(0.0001), np.log(0.1)], 'mutation': [0, 1]}},
                ],
                }
                }


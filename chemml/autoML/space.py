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
                                {'learning_rate': {'choice': np.arange(0,2000,50).tolist()}},
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
                                {'kernel': {'choice': ['linear','rbf','poly'], 
                                        'mutation': [0, 0.5]}},
                                {'C': {'uniform': [1,100], 
                                        'mutation': [0, 0.5]}}
                        ],

                'ElasticNet':[
                                {'alpha': {'uniform': [np.log(0.0001), np.log(0.1)],                
                                        'mutation': [0, 1]}},
                                {'l1_ratio': {'choice': np.arange(0.4,0.8,0.1).tolist()}}
                        ],
                        
                'DecisiontreeRegressor':[
                                {'criterion': {'choice': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']}},
                                {'splitter': {'choice': ['best', 'random']}},
                                {'min_samples_split': {'choice': range(2,50,10)}},
                                {'min_samples_leaf': {'choice': range(1,100,10)}},
                                {'dummy': {'uniform': [np.log(0.0001), np.log(0.1)],                
                                'mutation': [0, 1]}},
                                ],
                }
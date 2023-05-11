space_models = {
            # 'MLPRegressor()':[
            #                 {'alpha': {'uniform': [np.log(0.0001), np.log(0.1)],                
            #                 'mutation': [0, 1]}}, 
            #                 {'activation': {'choice': ['identity', 'logistic', 'tanh', 'relu']}},
            #                 {'neurons1':  {'choice': range(0,220,20)}},
            #                 {'neurons2':  {'choice': range(0,220,20)}},
            #                 {'neurons3':  {'choice': range(0,220,20)}}
            #                 ],
            # 'GradientBoostingRegressor()':[
            #                 {'loss': {'choice': ['log_loss', 'deviance', 'exponential']}},
            #                 {'learning_rate': {'uniform': [np.log(0.0001), np.log(0.1)],                
            #                 'mutation': [0, 1]}},
            #                 {'n_estimators': {'choice': np.random.randint(100,2000,size=10).tolist()}},
            #                 {'min_samples_split': {'choice': range(2,50,10)}},
            #                 {'min_samples_leaf': {'choice': range(10,100,10)}},
            #                 {'max_leaf_nodes': {'choice': range(20,50,5)}}             
            #                 ],

            # 'BayesianRidge()':[
            #                 {'alpha_1': {'uniform': [np.log(0.0001), np.log(0.1)],                
            #                 'mutation': [0, 1]}},
            #                 {'alpha_2': {'uniform': [np.log(0.0001), np.log(0.1)],                
            #                 'mutation': [0, 1]}},
            #                 {'lambda_1': {'uniform': [np.log(0.0001), np.log(0.1)],                
            #                 'mutation': [0, 1]}},
            #                 {'lambda_2': {'uniform': [np.log(0.0001), np.log(0.1)],                
            #                 'mutation': [0, 1]}}
            #                 ],

            # 'Ridge()': [
            #                 {'alpha': {'uniform': [np.log(0.0001), np.log(0.1)],                
            #                 'mutation': [0, 1]}},
            #                 {'dummy': {'choice': range(3,12,1)}}
            #                 ],

            'Lasso':[
                    {'alpha': {'uniform': [np.log(0.0001), np.log(0.1)],                
                            'mutation': [0, 1]}},
                    {'dummy': {'choice': ['auto', 'svd', 'eigen']}}
                    ],

        #    'SVR':  [
        #             {'kernel': {'choice': ['linear','rbf','poly'], 
        #                     'mutation': [0, 0.5]}},
        #             {'C': {'uniform': [1,100], 
        #                 'mutation': [0, 0.5]}}
        #             ],

            'ElasticNet':[
                    {'alpha': {'uniform': [np.log(0.0001), np.log(0.1)],                
                            'mutation': [0, 1]}},
                    {'l1_ratio': {'choice': np.arange(0.4,0.8,0.1).tolist()}}
                    ],

            # 'ElasticNetCV()':[
            #                 {'l1_ratio': {'uniform': [np.log(0.0001), np.log(0.1)],                
            #                 'mutation': [0, 1]}}, 
            #                 {'eps': {'choice': [1e-3, 1e-4, 1e-5, 0.01]}},
            #                 ],
            # 'LassoCV()':[ 
            #                 {'n_alphas': {'choice': [100, 50, 200, 10]}},
            #                 {'eps': {'choice': [1e-3, 1e-4, 1e-5, 0.01]}},
            #                 {'dummy_variable': {'uniform': [np.log(0.0001), np.log(0.1)],                
            #                 'mutation': [0, 1]}},
            #                 ],
            # 'BayesianRidge()':[
            #                 {'alpha_1': {'uniform': [np.log(0.0001), np.log(0.1)],                
            #                 'mutation': [0, 1]}}, 
            #                 {'lambda_1': {'uniform': [np.log(0.0001), np.log(0.1)],                
            #                 'mutation': [0, 1]}}, 
            #                 ],
            # 'OrthogonalMatchingPursuitCV()':[
            #                 {'dummy_variable': {'uniform': [np.log(0.0001), np.log(0.1)],                
            #                 'mutation': [0, 1]}}, 
            #                 {'cv':  {'choice': range(2,7)}},
            #                 ],
            # 'TweedieRegressor()':[
            #                 {'alpha': {'uniform': [np.log(0.0001), np.log(0.1)],                
            #                 'mutation': [0, 1]}}, 
            #                 {'link': {'choice': ['auto', 'identity', 'log']}},
            #                 {'solver': {'choice': ['lbfgs', 'newton-cholesky']}}
            #                 ],
                    
                    }
import warnings
import time
import os
import pandas as pd
import numpy as np
from sklearn.utils import all_estimators
from chemml.utils.utilities import regression_metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, recall_score, precision_score, f1_score
from chemml.optimization import GeneticAlgorithm
from chemml.chem import RDKitFingerprint
from chemml.chem import Molecule
import warnings
import random
from importlib import import_module
warnings.filterwarnings("ignore")

# The class takes in a dataframe, a target column name (string or integer), and a screener type
# (string). 
# 
# The class then checks if the target column name is a string or an integer. If it's a string, it
# checks if the string exists in the dataframe. If it does, it obtains the target column and the rest
# of the dataframe. If it doesn't, it throws an error. 
# 
# If the target column name is an integer, it checks if the integer exists in the dataframe. If it
# does, it obtains the target column and the rest of the dataframe. If it doesn't, it throws an error.
# 
# 
# The class then assigns the target column to the variable y and the rest of the dataframe to the
# variable x. 
# 
# The class then assigns the variables x and y to the class variables x and y. 
# 
# The class then returns nothing.

    # from chemml.autoML import ModelScreener

    # model_screener = ModelScreener(df=df, target="RI", 
    #                             screener_type="regressor",
    #                             output_file="regressor_scores_dragon.txt")
    # scores = model_screener.screen_models()


class ModelScreener(object):

    def __init__(self, df, target, featurization=None, smiles=None, screener_type="regressor", output_file="scores.txt"):
        """
        The function takes in a dataframe, a target column name or index, a screener type (classifier or
        regressor), and an output file name. 
        
        The function then splits the dataframe into a target column and a feature matrix. 
        
        The function then stores the feature matrix and target column as attributes of the class. 
        
        The function also stores the screener type and output file name as attributes of the class. 
        
        The function then returns nothing.
        
        :param df: The dataframe containing the data
        :param featurization: 
        :param target: The target column name or index
        :param screener_type: This is the type of screener you want to use. It can be either a
        classifier or a regressor, defaults to regressor (optional)
        :param output_file: The name of the file where the scores will be written to, defaults to
        scores.txt (optional)
        """
        
        if isinstance(df, pd.DataFrame):
            self.df = df
            # self.target = target
        else:
            raise TypeError("df must be a DataFrame!")

        if isinstance(target, str):
            if target in df.columns:
                self.target = target
            else:
                raise ValueError("Column name does not exist!")
        else:
            raise TypeError("Parameter target must be of type str !")
            
        
        if not isinstance(featurization, bool):
            raise TypeError("Featurization must be True or False !")
        self.featurization = featurization
        if self.featurization == True:
            if smiles == None:
                raise ValueError("If feature screeening is required, smiles column must be provided!")
            else:
                if isinstance(smiles, str):
                    if smiles in self.df.columns:
                        self.smiles = self.df[smiles]
                    else:
                        raise ValueError("Column name does not exist!")
                else:
                    raise TypeError("Parameter smiles must be of type str !")  
            self.x_list = []
        else:
            # make this a list of dataframes
            self.x_list = [self.df.loc[:, self.df.columns != self.target]]
                         
        if isinstance(screener_type, str):
            if screener_type not in ["classifier", "regressor"]:
                raise ValueError("Parameter screener_type must be 'classifier' or 'regressor' ")
            else:
                self.screener_type = screener_type
        else:
            raise TypeError("Parameter screener_type must be of type str")
        
        if isinstance(output_file, str):
            self.output_file = output_file
        else:
            raise TypeError("Parameter 'output_file' must be of type str")


    def get_all_models_sklearn(self, filter):
        """
        It returns a list of all the models in sklearn that match the filter
        
        :param filter: This is a function that takes a class and returns True if the class should be
        included in the list of estimators
        :return: A list of all the models that are in the sklearn library.
        """
        
        estimators = all_estimators(type_filter=filter)
        all_regs = []
        for name, RegClass in estimators:
            # print('Appending', name)
            try:
                reg = RegClass()
                all_regs.append(reg)
            except Exception as e:
                pass
        return all_regs
            
    def obtain_error_metrics(self, y_test, y_predict, model_name, model_start_time):
        """
        It takes in the true values of the target variable, the predicted values of the target variable,
        the name of the model, and the time at which the model started training. 
        
        It then returns a dictionary containing the error metrics for the model. 
        
        The error metrics returned depend on the type of the target variable. 
        
        If the target variable is a continuous variable, then the function returns the mean absolute
        error and the R2 score. 
        
        If the target variable is a categorical variable, then the function returns the accuracy,
        recall, precision, and F1 score. 
        
        If the target variable is neither continuous nor categorical, then the function returns None.
        
        :param y_test: the actual values of the target variable
        :param y_predict: the predicted values of the target variable
        :param model_name: name of the model
        :param model_start_time: The time at which the model started training
        """
        
        if self.screener_type == "regressor":
            # r2 = r2_score(y_test, y_predict)
            # mae = mean_absolute_error(y_test, y_predict)
            # time_taken = time.time() - model_start_time
            # scores = {"Model": model_name, "MAE": mae, "R2_score": r2, "time(seconds)": time_taken}
            
            scores = regression_metrics(y_true=y_test, y_predicted=y_predict)
            time_taken = time.time() - model_start_time
            scores = scores.iloc[:,4:]
            scores = scores.to_dict()
            scores["time(seconds)"]= time_taken
            scores["Model"]=model_name
            # print(scores)


        elif self.screener_type == "classifier":
            accuracy = accuracy_score(y_test, y_predict)
            recall = recall_score(y_test, y_predict, average='macro')
            precision = precision_score(y_test, y_predict, average='macro')
            f1score = f1_score(y_test, y_predict, average='macro')
            time_taken = time.time() - model_start_time
            scores = {"Model": model_name, "Accuracy": accuracy, "Recall": recall, "Precision": precision, "F1-score": f1score, "time(seconds)": time_taken}
        
        else:
            print("Work in progress...\n")
            print("classifier and regressor scores can be separately obtained: ")
            print("""set screener_type to 'regressor' or 'classifier'  """)
            scores = None
        return scores
        # return None

    def _represent_smiles(self):
        from chemml.chem import RDKitFingerprint
        from chemml.chem import CoulombMatrix
        # generate all representation techniques here

        mol_objs_list=[]
        for smi in self.smiles:
            mol = Molecule(smi, 'smiles')
            mol.hydrogens('add')
            mol.to_xyz('MMFF', maxIters=10000, mmffVariant='MMFF94s')
            mol_objs_list.append(mol)

        #The coulomb matrix type can be sorted (SC), unsorted(UM), unsorted triangular(UT), eigen spectrum(E), or random (RC)
        CM = CoulombMatrix(cm_type='SC',n_jobs=-1)
        self.x_list.append(CM.represent(mol_objs_list))

        # RDKit fingerprint types: 'morgan', 'hashed_topological_torsion' or 'htt' , 'MACCS' or 'maccs', 'hashed_atom_pair' or 'hap'
        morgan_fp = RDKitFingerprint(fingerprint_type='morgan', vector='bit', n_bits=1024, radius=3)
        self.x_list.append(morgan_fp.represent(mol_objs_list))
        
    def aggregate_scores(self,  scores_list, n_best):
        scores_combined = pd.concat(scores_list)
        self.scores_combined = scores_combined.sort_values(by='RMSE', ascending=False)
        return self.scores_combined[:n_best]


    def screen_models(self, n_best=10):
        """
        It takes in a dataframe, splits it into train and test, and then runs all the models in the
        screener_type list, and returns a dataframe with the error metrics for each model. 
        
        The screener_type list is a list of all the models that you want to run. 
        
        The output_file is the file where you want to store the error metrics for each model. 
        
        The obtain_error_metrics function is a function that takes in the actual and predicted values,
        and returns a dictionary with the error metrics. 
        
        The get_all_models_sklearn function is a function that returns a list of all the models that you
        want to run. 
        
        The function returns a dataframe with the error metrics for each model. 
        
        The function also stores the error metrics for each model in the output_file. 
        
        The function also prints the time taken to run
        :return: a dataframe with the error metrics for each model.
        """

        y = self.df[self.target]

        if self.featurization == True:
            # convert smiles to dataframe of x
            # return a lit of dataframes
            self._represent_smiles()
            
        scores_list=[]

        from .space import space_models
        for x in self.x_list:
            start_time = time.time()
            scores_df = pd.DataFrame()

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
            tmp_counter = 0
            
            def single_obj(model, x, y):
                # print("running single_obj")
                n_splits=4
                kf = KFold(n_splits)                                                      # cross validation based on Kfold (creates 5 validation train-test sets)
                accuracy_kfold = []
                for train_index, test_index in kf.split(x):
                    x_training, x_testing= x.iloc[train_index], x.iloc[test_index]
                    y_training, y_testing = y.iloc[train_index], y.iloc[test_index]
                    model.fit(x_training, y_training)
                    y_pred = model.predict(x_testing)
                    # y_pred, y_act = y_pred.reshape(-1,1), y_testing.reshape(-1,1)
                    # from chemml.utilities import regression_metrics
                    model_accuracy=r2_score(y_testing,y_pred)                             # evaluation metric:  r2_score
                    accuracy_kfold.append(model_accuracy)                                   # creates list of accuracies for each fold
                # print("def single_obj - completed")
                return np.mean(accuracy_kfold)
        
            def test_hyp(ml_model, x, y, xtest, ytest):                                          
                ml_model.fit(x, y)
                ypred = ml_model.predict(xtest)
                scores = self.obtain_error_metrics(y_test, y_predict, model_name, model_start_time)
                # print(" test_hyp completed ")
                return acc

            def set_hyper_params(parameters_list, model_name):
                print("parameters_list: ", parameters_list)
                module = import_module("sklearn.linear_model")
                model = getattr(module,model_name)()
                # print(type(model))
                # print(model)
                if model_name == 'MLPRegressor()':
                    layers = [parameters_list[i] for i in range(2,5) if parameters_list[i] != 0]
                    model = getattr(module,model_name)(alpha=np.exp(parameters_list[0]), activation=parameters_list[1], hidden_layer_sizes=tuple(layers), learning_rate='invscaling', max_iter=2000, early_stopping=True)  
                elif model_name == 'GradientBoostingRegressor()':
                    model = getattr(module,model_name)(l2_regularization=np.exp(parameters_list[0]), min_samples_leaf=parameters_list[1], max_leaf_nodes=parameters_list[2], random_state=42)
                elif model_name == 'TheilSenRegressor()':
                    from sklearn.linear_model import TheilSenRegressor
                    model = TheilSenRegressor(n_subsamples=parameters_list[0], max_subpopulation=parameters_list[1], random_state=42)
                elif model_name == 'RidgeCV()':
                    from sklearn.linear_model import RidgeCV
                    model = RidgeCV(cv=parameters_list[1], gcv_mode=parameters_list[1])
                elif model_name == 'KernelRidge()':
                    from sklearn.kernel_ridge import KernelRidge
                    model = KernelRidge(alpha = np.exp(parameters_list[0]), degree=parameters_list[1])
                elif model_name == 'ElasticNet':
                    model = getattr(module,model_name)(alpha=np.exp(parameters_list[0]), l1_ratio= parameters_list[1])
                elif model_name == 'Lasso':
                    model = getattr(module, model_name)(alpha=np.exp(parameters_list[0]))
                elif model_name == 'BayesianRidge()':
                    print("parameters_list: ", parameters_list)
                    from sklearn.linear_model import BayesianRidge
                    model = BayesianRidge(alpha_1=np.exp(parameters_list[0]), lambda_1=np.exp(parameters_list[1]))
                elif model_name == 'OrthogonalMatchingPursuitCV()':
                    from sklearn.linear_model import OrthogonalMatchingPursuitCV
                    model = OrthogonalMatchingPursuitCV(cv=parameters_list[1])
                elif model_name == 'TweedieRegressor()':
                    print("parameters_list: ", parameters_list)
                    from sklearn.linear_model import TweedieRegressor
                    model = TweedieRegressor(alpha=np.exp(parameters_list[0]), link=parameters_list[1], solver=parameters_list[2])
                else:
                    raise ValueError("Not yet incorporated!")
                
                # print("model obtained!")
                return model

            def ga(X_train, y_train, X_test, y_test, model_name, space_final, al):
                start_time_ga = time.time()
                            
                def ga_eval(indi,model_name=model_name):
                    # print("indi: ", indi)
                    # print("model_name: ", model_name)
                    with open (self.output_file,'a') as ga_progress:
                    # ga_progress = open(self.output_file,"a")
                        ga_progress.write(str(indi))
                        ga_progress.close()
                    model = set_hyper_params(parameters_list=indi, model_name=model_name)
                    ga_search = single_obj(model=model, x=X_train, y=y_train)
                    return ga_search 

                
                gann = GeneticAlgorithm(evaluate=ga_eval, space=space_final, fitness=('max',), pop_size = 5, crossover_size=5, mutation_size=2, algorithm=al)
                best_ind_df, best_individual = gann.search(n_generations=2, early_stopping=10)                     # set pop_size<30, n_generations*pop_size = no. of times GA runs                      
                print(model_name, ": GeneticAlgorithm - complete")
                
                all_items = list(gann.fitness_dict.items())
                all_items_df = pd.DataFrame(all_items, columns=['hyperparameters', 'Accuracy_score'])
                all_items_df.to_csv(model_name+'_fitness_dict.csv', index=False)
                
                best_ind_df = best_ind_df.sort_values(by='Fitness_values', ascending=False)
                best_ind_df.to_csv(model_name+'_ga_best.csv',index=False)
                ga_time = (time.time() - start_time_ga)/3600
                
                # print("type(best_ind_df['Best_individual'][0]): ", type(best_ind_df["Best_individual"][0]))
                # print("best_ind_df['Best_individual'][0]: ", best_ind_df["Best_individual"][0])
                best_hyper_params = best_ind_df["Best_individual"][0]
                # print("best_hyper_params: ", best_hyper_params)
                best_ga_model = set_hyper_params(parameters_list=best_hyper_params, model_name=model_name)
                
                ga_accuracy_test = test_hyp(ml_model=best_ga_model, x=X_train, y=y_train, xtest=X_test, ytest=y_test)
                print("Model:", model_name)
                print("GA time(hours): ", ga_time)
                print("Model params: ", best_ga_model.get_params())
                print("Test set R2_score for the best ga hyperparameter: ", ga_accuracy_test)
                print("\n")
                return 


            for model_name in space_models.keys():
            # for model in all_models[:3]:
                tmp_counter = tmp_counter+1
                # model_name = str(model)
                
                # system.exit()
                print("Running model no: ", tmp_counter, "; Name: ", model_name)
                try:
                    model_start_time = time.time()

                    space_final=tuple(space_models[model_name])
                    with open (self.output_file,'a') as ga_progress:
                        ga_progress.write("\n"+model_name)
                        ga_progress.write("\n")
                        ga_progress.close()
                    ga(X_train, y_train, X_test, y_test, model_name=model_name, space_final=space_final, al=3) 
                    ga_progress.write("\nPerforming GA on next model \n")
                        
                    if scores_df.empty==True:
                        scores = self.obtain_error_metrics(y_test, y_predict, model_name, model_start_time)
                        with open(self.output_file, 'w') as f: 
                            for key, value in scores.items(): 
                                f.write('%s:%s\n' % (key, value))
                            f.write('\n')
                        f.close()
                        scores_df = pd.DataFrame(data=scores, index=[0])
                        # print("scores_df: ", scores_df)
                    else:
                        scores = self.obtain_error_metrics(y_test, y_predict, model_name, model_start_time)
                        with open(self.output_file, 'a') as f: 
                            for key, value in scores.items(): 
                                f.write('%s:%s\n' % (key, value))
                            f.write('\n')
                        f.close()
                        scores_df_1 = pd.DataFrame(data=scores, index=[0])
                        scores_df = pd.concat([scores_df,scores_df_1], ignore_index=True)
                        # print("scores_df: ", scores_df)
                except Exception as e: 
                    print("model_name: ", model_name)
                    print(e)
                    print("\n")
            print("\n")
            print("--- %s seconds ---" % (time.time() - start_time))
            if self.screener_type == "regressor":
                scores_df = scores_df[['Model', 'time(seconds)', 'r_squared', 'ME', 'MAE', 'MSE', 'RMSE', 'MSLE', 'RMSLE', 'MAPE', 'MaxAPE', 'RMSPE', 'MPE', 'MaxAE', 'deltaMaxE', 'std']]
            scores_list.append(scores_df)

        # aggregate scores list
        best_models = self.aggregate_scores(scores_list,n_best)

        return best_models

    # def optimize_screened_models(self, file_name=None):


        if file_name == None:
            if self.scores_df.empty==False:
                print("scores_df obtained...")
            else:
                raise ValueError("Please ensure ModelScreener.scores_df has been created. Use ModelScreener.screen_models() to obtain scores_df \nOr provide 'file_name' to access file with error metric for each model")
        elif isinstance(file_name, str):
                split_name = os.path.splitext(file_name)
                # print("split_name: ", split_name)
                if split_name[1] == ".csv":
                    self.scores_df = pd.read_csv(file_name)
                else:
                    raise TypeError("Parameter 'file_name' must have '.csv' extension (e.g. file_name = 'scores_regressor.csv')")
        else:
            raise TypeError("Parameter 'file_name' must be of type str")
        
        if self.screener_type == "regressor":
            sorted_df = self.scores_df.sort_values(by='r_squared', ascending=False)
        elif self.screener_type == "classifier":
            sorted_df = self.scores_df.sort_values(by='Accuracy', ascending=False)
        else:
            raise TypeError("Parameter screener_type must be 'regressor' or 'classifier'")

        self.sorted_df = sorted_df
        # print("sorted_df.head() :", self.sorted_df.head())
             
        # space_models = {
        #     'MLPRegressor()':[
        #                     {'alpha': {'uniform': [np.log(0.0001), np.log(0.1)],                
        #                     'mutation': [0, 1]}}, 
        #                     {'activation': {'choice': ['identity', 'logistic', 'tanh', 'relu']}},
        #                     {'neurons1':  {'choice': range(0,220,20)}},
        #                     {'neurons2':  {'choice': range(0,220,20)}},
        #                     {'neurons3':  {'choice': range(0,220,20)}}
        #                     ],
        #     'HistGradientBoostingRegressor()':[
        #                     {'l2_regularization': {'uniform': [np.log(0.0001), np.log(0.1)],                
        #                     'mutation': [0, 1]}},
        #                     {'min_samples_leaf': {'choice': range(10,100,10)}},
        #                     {'max_leaf_nodes': {'choice': range(20,50,5)}}              
        #                     ],
        #     'TheilSenRegressor()':[
        #                     {'n_subsamples': {'choice': range(0,100,10)}},
        #                     {'max_subpopulation': {'uniform': [10, 50],                
        #                     'mutation': [0, 1]}}
        #                     ],
        #     'RidgeCV()':    [
        #                     {'cv': {'choice': range(3,10,1)}},
        #                     {'gcv_mode': {'choice': ['auto', 'svd', 'eigen']}},
        #                     {'dummy_variable': {'uniform': [np.log(0.0001), np.log(0.1)],                
        #                     'mutation': [0, 1]}},
        #                     ],
        #     'BayesianRidge()':[
        #                     {'alpha_1': {'uniform': [np.log(0.0001), np.log(0.1)],                
        #                     'mutation': [0, 1]}},
        #                     {'alpha_2': {'uniform': [np.log(0.0001), np.log(0.1)],                
        #                     'mutation': [0, 1]}},
        #                     {'lambda_1': {'uniform': [np.log(0.0001), np.log(0.1)],                
        #                     'mutation': [0, 1]}},
        #                     {'lambda_2': {'uniform': [np.log(0.0001), np.log(0.1)],                
        #                     'mutation': [0, 1]}}
        #                     ],
        #     'KernelRidge()': [
        #                     {'alpha': {'uniform': [np.log(0.0001), np.log(0.1)],                
        #                     'mutation': [0, 1]}},
        #                     {'degree': {'choice': range(3,12,1)}}
        #                     ],
        #     'ElasticNetCV()':[
        #                     {'l1_ratio': {'uniform': [np.log(0.0001), np.log(0.1)],                
        #                     'mutation': [0, 1]}}, 
        #                     {'eps': {'choice': [1e-3, 1e-4, 1e-5, 0.01]}},
        #                     ],
        #     'LassoCV()':[ 
        #                     {'n_alphas': {'choice': [100, 50, 200, 10]}},
        #                     {'eps': {'choice': [1e-3, 1e-4, 1e-5, 0.01]}},
        #                     {'dummy_variable': {'uniform': [np.log(0.0001), np.log(0.1)],                
        #                     'mutation': [0, 1]}},
        #                     ],
        #     'BayesianRidge()':[
        #                     {'alpha_1': {'uniform': [np.log(0.0001), np.log(0.1)],                
        #                     'mutation': [0, 1]}}, 
        #                     {'lambda_1': {'uniform': [np.log(0.0001), np.log(0.1)],                
        #                     'mutation': [0, 1]}}, 
        #                     ],
        #     'OrthogonalMatchingPursuitCV()':[
        #                     {'dummy_variable': {'uniform': [np.log(0.0001), np.log(0.1)],                
        #                     'mutation': [0, 1]}}, 
        #                     {'cv':  {'choice': range(2,7)}},
        #                     ],
        #     'TweedieRegressor()':[
        #                     {'alpha': {'uniform': [np.log(0.0001), np.log(0.1)],                
        #                     'mutation': [0, 1]}}, 
        #                     {'link': {'choice': ['auto', 'identity', 'log']}},
        #                     {'solver': {'choice': ['lbfgs', 'newton-cholesky']}}
        #                     ],
                    
        #             }
            
        # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
        # print("Train_Test_Split_done!")
        # print("Starting to optimize screened models...")

        # def single_obj(model, x, y):
        #     # print("running single_obj")
        #     n_splits=4
        #     kf = KFold(n_splits)                                                      # cross validation based on Kfold (creates 5 validation train-test sets)
        #     accuracy_kfold = []
        #     for train_index, test_index in kf.split(x):
        #         x_training, x_testing= x.iloc[train_index], x.iloc[test_index]
        #         y_training, y_testing = y.iloc[train_index], y.iloc[test_index]
        #         model.fit(x_training, y_training)
        #         y_pred = model.predict(x_testing)
        #         # y_pred, y_act = y_pred.reshape(-1,1), y_testing.reshape(-1,1)
        #         model_accuracy=r2_score(y_testing,y_pred)                             # evaluation metric:  r2_score
        #         accuracy_kfold.append(model_accuracy)                                   # creates list of accuracies for each fold
        #     # print("def single_obj - completed")
        #     return np.mean(accuracy_kfold)
        
        # def test_hyp(ml_model, x, y, xtest, ytest):                                          
        #     ml_model.fit(x, y)
        #     ypred = ml_model.predict(xtest)
        #     acc=r2_score(ytest,ypred)
        #     # print(" test_hyp completed ")
        #     return acc
        
        # def set_hyper_params(parameters_list, model_name):
        #     if model_name == 'MLPRegressor()':
        #         layers = [parameters_list[i] for i in range(2,5) if parameters_list[i] != 0]
        #         model = MLPRegressor(alpha=np.exp(parameters_list[0]), activation=parameters_list[1], hidden_layer_sizes=tuple(layers), learning_rate='invscaling', max_iter=2000, early_stopping=True)  
        #     elif model_name == 'HistGradientBoostingRegressor()':
        #         from sklearn.ensemble import HistGradientBoostingRegressor
        #         model = HistGradientBoostingRegressor(l2_regularization=np.exp(parameters_list[0]), min_samples_leaf=parameters_list[1], max_leaf_nodes=parameters_list[2], random_state=42)
        #     elif model_name == 'TheilSenRegressor()':
        #         from sklearn.linear_model import TheilSenRegressor
        #         model = TheilSenRegressor(n_subsamples=parameters_list[0], max_subpopulation=parameters_list[1], random_state=42)
        #     elif model_name == 'RidgeCV()':
        #         from sklearn.linear_model import RidgeCV
        #         model = RidgeCV(cv=parameters_list[1], gcv_mode=parameters_list[1])
        #     elif model_name == 'KernelRidge()':
        #         from sklearn.kernel_ridge import KernelRidge
        #         model = KernelRidge(alpha = np.exp(parameters_list[0]), degree=parameters_list[1])
        #     elif model_name == 'ElasticNetCV()':
        #         from sklearn.linear_model import ElasticNetCV
        #         model = ElasticNetCV(l1_ratio=np.exp(parameters_list[0]), eps= parameters_list[1])
        #     elif model_name == 'LassoCV()':
        #         from sklearn.linear_model import LassoCV
        #         model = LassoCV(n_alphas=parameters_list[0], eps=parameters_list[1])
        #     elif model_name == 'BayesianRidge()':
        #         print("parameters_list: ", parameters_list)
        #         from sklearn.linear_model import BayesianRidge
        #         model = BayesianRidge(alpha_1=np.exp(parameters_list[0]), lambda_1=np.exp(parameters_list[1]))
        #     elif model_name == 'OrthogonalMatchingPursuitCV()':
        #         from sklearn.linear_model import OrthogonalMatchingPursuitCV
        #         model = OrthogonalMatchingPursuitCV(cv=parameters_list[1])
        #     elif model_name == 'TweedieRegressor()':
        #         print("parameters_list: ", parameters_list)
        #         from sklearn.linear_model import TweedieRegressor
        #         model = TweedieRegressor(alpha=np.exp(parameters_list[0]), link=parameters_list[1], solver=parameters_list[2])
        #     else:
        #         raise ValueError("Not yet incorporated!")
            
        #     # print("model obtained!")
        #     return model

        # def ga(X_train, y_train, X_test, y_test, model_name, space_final, al):
        #     start_time_ga = time.time()
                        
        #     def ga_eval(indi,model_name=model_name):
        #         # print("indi: ", indi)
        #         # print("model_name: ", model_name)
        #         with open (self.output_file,'a') as ga_progress:
        #         # ga_progress = open(self.output_file,"a")
        #             ga_progress.write(str(indi))
        #             ga_progress.close()
        #         model = set_hyper_params(parameters_list=indi, model_name=model_name)
        #         ga_search = single_obj(model=model, x=X_train, y=y_train)
        #         return ga_search 

            
        #     gann = GeneticAlgorithm(evaluate=ga_eval, space=space_final, fitness=('max',), pop_size = 5, crossover_size=5, mutation_size=2, algorithm=al)
        #     best_ind_df, best_individual = gann.search(n_generations=2, early_stopping=10)                     # set pop_size<30, n_generations*pop_size = no. of times GA runs                      
        #     print(model_name, ": GeneticAlgorithm - complete")
            
        #     all_items = list(gann.fitness_dict.items())
        #     all_items_df = pd.DataFrame(all_items, columns=['hyperparameters', 'Accuracy_score'])
        #     all_items_df.to_csv(model_name+'_fitness_dict.csv', index=False)
            
        #     best_ind_df = best_ind_df.sort_values(by='Fitness_values', ascending=False)
        #     best_ind_df.to_csv(model_name+'_ga_best.csv',index=False)
        #     ga_time = (time.time() - start_time_ga)/3600
            
        #     # print("type(best_ind_df['Best_individual'][0]): ", type(best_ind_df["Best_individual"][0]))
        #     # print("best_ind_df['Best_individual'][0]: ", best_ind_df["Best_individual"][0])
        #     best_hyper_params = best_ind_df["Best_individual"][0]
        #     # print("best_hyper_params: ", best_hyper_params)
        #     best_ga_model = set_hyper_params(parameters_list=best_hyper_params, model_name=model_name)
            
        #     ga_accuracy_test = test_hyp(ml_model=best_ga_model, x=X_train, y=y_train, xtest=X_test, ytest=y_test)
        #     print("Model:", model_name)
        #     print("GA time(hours): ", ga_time)
        #     print("Model params: ", best_ga_model.get_params())
        #     print("Test set R2_score for the best ga hyperparameter: ", ga_accuracy_test)
        #     print("\n")

        # for model_name in self.sorted_df["Model"][:5]:
        #     # if model_name != 'TheilSenRegressor()':
        #     # if model_name == 'RidgeCV()':
        #     space_final=tuple(space_models[model_name])
        #     try:
        #         with open (self.output_file,'a') as ga_progress:
        #             ga_progress.write("\n"+model_name)
        #             ga_progress.write("\n")
        #             ga_progress.close()
        #         ga(X_train, y_train, X_test, y_test, model_name=model_name, space_final=space_final, al=3) 
        #         ga_progress.write("\nPerforming GA on next model \n")
                
        #     except Exception as e:
        #         print("model_name: ", model_name)
        #         print(e)
        #         print("\n")
        # # model_with_params
        # return None

    

    # for model_name in self.sorted_df["Model"][:5]:
        #     # if model_name != 'TheilSenRegressor()':
        #     # if model_name == 'RidgeCV()':
        #     space_final=tuple(space_models[model_name])
        #     try:
        #         with open (self.output_file,'a') as ga_progress:
        #             ga_progress.write("\n"+model_name)
        #             ga_progress.write("\n")
        #             ga_progress.close()
        #         ga(X_train, y_train, X_test, y_test, model_name=model_name, space_final=space_final, al=3) 
        #         ga_progress.write("\nPerforming GA on next model \n")
                
        #     except Exception as e:
        #         print("model_name: ", model_name)
        #         print(e)
        #         print("\n")
        # model_with_params
        # return None

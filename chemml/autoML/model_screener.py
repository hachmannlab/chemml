import warnings
import time
import os
import pandas as pd
import numpy as np
from sklearn.utils import all_estimators
from chemml.utils import regression_metrics
from sklearn.model_selection import train_test_split, KFold
from chemml.optimization import GeneticAlgorithm
from chemml.chem import RDKitFingerprint
from chemml.chem import Molecule
import warnings
import random
from importlib import import_module
warnings.filterwarnings("ignore")



class ModelScreener(object):

#   from chemml.autoML import ModelScreener

#   MS = ModelScreener(df, target="density_Kg/m3", featurization=True, smiles="smiles", 
#                    screener_type="regressor", output_file="testing.txt")

#   scores = MS.screen_models(n_best=4)

    def __init__(self, df, target, featurization=False, smiles=None, screener_type="regressor", n_gen=10, output_file="scores.txt"):
        """
        This is a constructor function that initializes various parameters for a machine learning model.
        

        Parameters
        ----------
        df : pandas DataFrame
            a pandas DataFrame containing the data to be used for modeling
        target : str
            The name of the target column in the input DataFrame that the model will predict
        featurization : bool, optional
            A boolean indicating whether feature screening is required or not, by default False
        smiles : str , optional
            A string representing the name of the column in the input DataFrame that contains
            the SMILES strings for the molecules. This is only required if featurization is set to True, by default None
        screener_type : str, optional
            This parameter specifies whether the screener model should be a classifier
            or a regressor. It must be set to either "classifier" or "regressor",, by default "regressor"
        n_gen : int, optional
            number of generations that genetic algorithm should run, by default 10
        output_file : str, optional
            The name of the file where the scores will be written to, by default "scores.txt"

        """        
        
        self.n_gen=n_gen
        
        if isinstance(df, pd.DataFrame):
            self.df = df
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
            self.x_list = {}
        else:
            # make this a list of dataframes
            self.x_list = {"user_given": self.df.loc[:, self.df.columns != self.target]}
                         
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

        Parameters
        ----------
        filter : str
            This is a function that takes a class and returns True if the class should be included in the list of estimators

        Returns
        -------
        list
            A list of all the models that are in the sklearn library
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
            
    # def obtain_error_metrics(self, y_test, y_predict, model_name, model_start_time):
    #     """
    #     It takes in the true values of the target variable, the predicted values of the target variable,
    #     the name of the model, and the time at which the model started training. 
        
    #     It then returns a dictionary containing the error metrics for the model. 
        
    #     The error metrics returned depend on the type of the target variable. 
        
    #     If the target variable is a continuous variable, then the function returns the mean absolute
    #     error and the R2 score. 
        
    #     If the target variable is a categorical variable, then the function returns the accuracy,
    #     recall, precision, and F1 score. 
        
    #     If the target variable is neither continuous nor categorical, then the function returns None.
        
    #     :param y_test: the actual values of the target variable
    #     :param y_predict: the predicted values of the target variable
    #     :param model_name: name of the model
    #     :param model_start_time: The time at which the model started training
    #     """
        
    #     if self.screener_type == "regressor":
    #         # r2 = r2_score(y_test, y_predict)
    #         # mae = mean_absolute_error(y_test, y_predict)
    #         # time_taken = time.time() - model_start_time
    #         # scores = {"Model": model_name, "MAE": mae, "R2_score": r2, "time(seconds)": time_taken}
            
    #         scores = regression_metrics(y_true=y_test, y_predicted=y_predict)
    #         time_taken = time.time() - model_start_time
    #         scores = scores.to_dict()
    #         scores["time(seconds)"]= time_taken
    #         scores["Model"]=model_name
    #         scores["Feature"]=key
    #         # print(scores)


    #     elif self.screener_type == "classifier":
    #         accuracy = accuracy_score(y_test, y_predict)
    #         recall = recall_score(y_test, y_predict, average='macro')
    #         precision = precision_score(y_test, y_predict, average='macro')
    #         f1score = f1_score(y_test, y_predict, average='macro')
    #         time_taken = time.time() - model_start_time
    #         scores = {"Model": model_name, "Accuracy": accuracy, "Recall": recall, "Precision": precision, "F1-score": f1score, "time(seconds)": time_taken}
        
    #     else:
    #         print("Work in progress...\n")
    #         print("classifier and regressor scores can be separately obtained: ")
    #         print("""set screener_type to 'regressor' or 'classifier'  """)
    #         scores = None
    #     return scores
        # return None

    def _represent_smiles(self):
        """
        This function generates various molecular representations (Coulomb matrix, RDKit fingerprints,
        and RDKit descriptors) for a list of molecules represented by SMILES strings.

        Returns
        -------
        list 
            list of pandas DataFrames consisting of various molecular representations
        """        
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
        self.x_list["CoulombMatrix"] = CM.represent(mol_objs_list)

        # RDKit fingerprint types: 'morgan', 'hashed_topological_torsion' or 'htt' , 'MACCS' or 'maccs', 'hashed_atom_pair' or 'hap'
        morgan_fp = RDKitFingerprint(fingerprint_type='morgan', vector='bit', n_bits=1024, radius=3)
        self.x_list["morganfingerprints_radius3"] = morgan_fp.represent(mol_objs_list)

        MACCS = RDKitFingerprint(fingerprint_type='MACCS', vector='bit', n_bits=1024, radius=3)
        self.x_list["MACCS_radius3"] = MACCS.represent(mol_objs_list)

        hashed_topological_torsion = RDKitFingerprint(fingerprint_type='hashed_topological_torsion', vector='bit', n_bits=1024, radius=3)
        self.x_list["hashedtopologicaltorsion_radius3"] = hashed_topological_torsion.represent(mol_objs_list)

        # RDKit Descriptors
        def getMolDescriptors(smiles_list, missingVal=np.nan):
            """ 
            Calculate the full list of descriptors for a molecule
            
            Parameters
            ----------
            smiles_list : list
                list of smiles codes of molecules
            missingVal : _type_, optional
                used if the descriptor cannot be calculated, by default np.nan

            Returns
            -------
            pandas Dataframe
                DataFrame consisting of all descriptors available in rdkit
            """            
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            res = {}
            descriptors_df = pd.DataFrame()
            for molecules_objs in mol_objs_list:
                # res["smiles"] = molecules_objs.smiles
                for nm,fn in Descriptors._descList:
                    # some of the descriptor functions can throw errors if they fail, catch those here:
                    try:
                        val = fn(Chem.MolFromSmiles(molecules_objs.smiles))
                    except:
                        # print the error message:
                        import traceback
                        traceback.print_exc()
                        # and set the descriptor value to whatever missingVal is
                        val = missingVal
                    res[nm] = val
                all_descriptors = pd.DataFrame(res,index=[0])
                descriptors_df = pd.concat([descriptors_df, all_descriptors], ignore_index=True)
            return descriptors_df
        
        
        allDescrs = getMolDescriptors(smiles_list = self.smiles)
        self.x_list["rdkit_descriptors"] = allDescrs
        
    def aggregate_scores(self,  scores_list, n_best):
        """ 
        This function aggregates a list of scores, combines them into a pandas dataframe, sorts them by
        RMSE in ascending order, and returns the top n_best scores.
        
        :param scores_list: 
        :param n_best: 
        

        Parameters
        ----------
        scores_list : list
            a list of pandas dataframes containing scores for different models or experiments
        n_best : int
            The number of best scores to return from the combined scores list
         

        Returns
        -------
        pandas DataFrame
            the top n_best scores from the combined scores list, sorted by RMSE in ascending order.
        """       
        scores_combined = pd.concat(scores_list)
        self.scores_combined = scores_combined.sort_values(by='RMSE', ascending=True)
        return self.scores_combined[:n_best]

    def screen_models(self, n_best=10):
        """
        This function performs genetic algorithm hyperparameter tuning on a list of regression models
        and returns the best performing models.
        

        Parameters
        ----------
        n_best : int, optional
            The number of best models to return as output, by default 10

        Returns
        -------
        pandas DataFrame
            the best models based on their scores, as determined by the genetic algorithm. The
        number of best models returned is determined by the `n_best` parameter

        Raises
        ------
        ValueError
            _description_
        """        
        
        y = self.df[self.target]

        if self.featurization == True:
            self._represent_smiles()
            
        scores_list=[]

        from .space import space_models
        for key in self.x_list.keys():
            start_time = time.time()
            scores_df = pd.DataFrame()

            X_train, X_test, y_train, y_test = train_test_split(self.x_list[key], y, test_size=0.1, random_state=42)
            tmp_counter = 0
            
            def single_obj(model, x, y):
                n_splits=4
                kf = KFold(n_splits)                                                      # cross validation based on Kfold (creates 5 validation train-test sets)
                accuracy_kfold = []
                for train_index, test_index in kf.split(x):
                    x_training, x_testing= x.iloc[train_index], x.iloc[test_index]
                    y_training, y_testing = y.iloc[train_index], y.iloc[test_index]
                    model.fit(x_training, y_training)
                    y_pred = model.predict(x_testing)
                    score = regression_metrics(y_testing,y_pred)['r_squared'][0]
                    # evaluation metric:  r2_score
                    accuracy_kfold.append(score)                                   # creates list of accuracies for each fold
                return np.mean(accuracy_kfold)
        
            def test_hyp(ml_model, x, y, xtest, ytest, key):                                          
                ml_model.fit(x, y)
                ypred = ml_model.predict(xtest)
                if self.screener_type == "regressor":            
                    scores = regression_metrics(y_true=y_test, y_predicted=ypred)
                    time_taken = time.time() - model_start_time
                    scores["time(seconds)"]= time_taken
                    scores["Model"]=model_name
                    scores['parameters']=[ml_model.get_params()]
                    scores['Feature']=key
                    

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

            def set_hyper_params(parameters_list, model_name):
                # print("parameters_list: ", parameters_list)
                from .models_dict import models_dict
                module = import_module(models_dict[model_name])

                if model_name == 'MLPRegressor':
                    layers = [parameters_list[i] for i in range(2,5) if parameters_list[i] != 0]
                    model = getattr(module,model_name)(alpha=np.exp(parameters_list[0]), activation=parameters_list[1], hidden_layer_sizes=tuple(layers), learning_rate='invscaling', max_iter=2000, early_stopping=True)  

                elif model_name == 'GradientBoostingRegressor':
                    model = getattr(module,model_name)(loss=parameters_list[0], n_estimators=parameters_list[1], min_samples_split=parameters_list[2], min_samples_leaf=parameters_list[3], random_state=42)

                elif model_name == 'RandomForestRegressor':
                    model = getattr(module,model_name)(n_estimators=parameters_list[0],criterion=parameters_list[1], min_samples_split=parameters_list[2], min_samples_leaf=parameters_list[3])
        
                elif model_name == 'Ridge':
                    model = getattr(module,model_name)(alpha=parameters_list[0])

                elif model_name == 'Lasso':
                    model = getattr(module, model_name)(alpha=np.exp(parameters_list[0]))

                elif model_name == 'SVR':
                    model = getattr(module,model_name)(kernel=parameters_list[0], C=parameters_list[1])
                                    
                elif model_name == 'ElasticNet':
                    model = getattr(module,model_name)(alpha=np.exp(parameters_list[0]), l1_ratio= parameters_list[1])

                elif model_name == 'DecisionTreeRegressor':
                    model = getattr(module,model_name)(criterion=parameters_list[0], splitter=parameters_list[1], min_samples_split=parameters_list[2], min_samples_leaf=parameters_list[3])

                else:
                    raise ValueError("This model cannot be used currently. Please refer to documentation. ")
                
                # print("model obtained!")
                return model

            def ga(X_train, y_train, X_test, y_test, model_name, space_final, al):
                start_time_ga = time.time()
                            
                def ga_eval(indi,model_name=model_name):
                    # print("indi: ", indi)
                    # print("model_name: ", model_name)
                    with open (self.output_file,'a') as ga_progress:
                        ga_progress.write(str(indi))
                    model = set_hyper_params(parameters_list=indi, model_name=model_name)
                    ga_search = single_obj(model=model, x=X_train, y=y_train)
                    return ga_search 

                
                gann = GeneticAlgorithm(evaluate=ga_eval, space=space_final, fitness=('max',), pop_size = 3, crossover_size=2, mutation_size=1, algorithm=al)
                best_ind_df, best_individual = gann.search(n_generations=self.n_gen, early_stopping=10)                     # set pop_size<30, n_generations*pop_size = no. of times GA runs                      
                print(model_name, ": GeneticAlgorithm - complete")
                
                all_items = list(gann.fitness_dict.items())
                all_items_df = pd.DataFrame(all_items, columns=['hyperparameters', 'Accuracy_score'])
                all_items_df.to_csv(model_name+'_fitness_dict.csv', index=False)
                
                best_ind_df = best_ind_df.sort_values(by='Fitness_values', ascending=False)
                best_ind_df.to_csv(model_name+'_ga_best.csv',index=False)
                ga_time = (time.time() - start_time_ga)/3600
                
                best_hyper_params = best_ind_df["Best_individual"][0]
                best_ga_model = set_hyper_params(parameters_list=best_hyper_params, model_name=model_name)
                
                ga_accuracy_test = test_hyp(ml_model=best_ga_model, x=X_train, y=y_train, xtest=X_test, ytest=y_test, key=key)
                print("Model:", model_name)
                print("GA time(hours): ", ga_time)
                # print("Model params: ", best_ga_model.get_params())
                # print("Test set R2_score for the best ga hyperparameter: ", ga_accuracy_test)
                print("\n")
                return ga_accuracy_test


            for model_name in space_models.keys():
            # for model_name in ['RandomForestRegressor', 'Ridge', 'Lasso', 'ElasticNet']:
                tmp_counter = tmp_counter+1
                print("\nRunning model no: ", tmp_counter, "; Name: ", model_name)
                try:
                    model_start_time = time.time()
                    space_final=tuple(space_models[model_name])
                    with open (self.output_file,'a') as ga_progress:
                        ga_progress.write("\n"+model_name)
                        ga_progress.write("\n")
                    scores_list.append(ga(X_train, y_train, X_test, y_test, model_name=model_name, space_final=space_final, al=3))
                    print("scores_list: ", scores_list)
                    print("--------------------------------------------------------------------------------")
                    with open (self.output_file,'a') as ga_progress:
                        ga_progress.write("\nPerforming GA on next model \n")
                        
                except Exception as e: 
                    print("model_name: ", model_name)
                    print(e)
                    print("\n")
            print("\n")
            print("--- %s seconds ---" % (time.time() - start_time))

        # aggregate scores list
        best_models = self.aggregate_scores(scores_list,n_best)

        return best_models


# Raises
#         ------
#         TypeError
#             _description_
#         ValueError
#             _description_
#         TypeError
#             _description_
#         TypeError
#             _description_
#         ValueError
#             _description_
#         ValueError
#             _description_
#         TypeError
#             _description_
#         ValueError
#             _description_
#         TypeError
#             _description_
#         TypeError
#             _description_
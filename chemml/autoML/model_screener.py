import os
import traceback
import pandas as pd
import numpy as np
from sklearn.utils import all_estimators
from chemml.utils import regression_metrics
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split, KFold
from chemml.optimization import GeneticAlgorithm
from chemml.chem import RDKitFingerprint, Mordred
from chemml.chem import Molecule
import warnings
import random
import time
from tqdm import tqdm
from importlib import import_module
from multiprocessing import Pool, Manager

warnings.filterwarnings("ignore")


def _log(message, output_file=None, to_console=True):
    if to_console:
        print(message)
    if output_file is not None:
        with open(output_file, 'a') as f:
            f.write(message)


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
            # List to gather locations of invalid SMILES if present and remove them from the targets
            self.discarded_indices = [] 
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

    def run_model(self, model_name, tmp_counter, output_file, X_train, y_train, X_test, y_test, space_models, scores_list, key):

        def single_obj(model, x, y):
            n_splits=4
            kf = KFold(n_splits)                                                      # cross validation based on Kfold (creates 5 validation train-test sets)
            accuracy_kfold = []
            for train_index, test_index in kf.split(x):
                x_training, x_testing= x.iloc[train_index], x.iloc[test_index]
                y_training, y_testing = y.iloc[train_index], y.iloc[test_index]
                model.fit(x_training, y_training)
                y_pred = model.predict(x_testing)
                if self.screener_type == "regressor":
                    score = regression_metrics(y_testing, y_pred)['r_squared'][0]
                else:
                    score = accuracy_score(y_testing, y_pred)
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
                accuracy = accuracy_score(y_test, ypred)
                recall = recall_score(y_test, ypred, average='macro')
                precision = precision_score(y_test, ypred, average='macro')
                f1score = f1_score(y_test, ypred, average='macro')
                time_taken = time.time() - model_start_time
                scores = {"Model": model_name, "Accuracy": accuracy, "Recall": recall, "Precision": precision, "F1-score": f1score, "time(seconds)": time_taken, "parameters": [ml_model.get_params()], "Feature": key}
                scores = pd.Series(scores)
                scores = pd.DataFrame(scores)
                scores = scores.T

            else:
                _log("Work in progress...\nclassifier and regressor scores can be separately obtained: \nset screener_type to 'regressor' or 'classifier'  ", output_file=self.output_file)
                scores = None

            return scores

        def set_hyper_params(parameters_list, model_name):
            from .models_dict import models_dict
            module = import_module(models_dict[model_name])

            if model_name == 'MLPRegressor':
                layers = [parameters_list[i] for i in range(2,5) if parameters_list[i] != 0]
                model = getattr(module,model_name)(alpha=np.exp(parameters_list[0]), activation=parameters_list[1], hidden_layer_sizes=tuple(layers), learning_rate='invscaling', max_iter=2000, early_stopping=True, random_state=42)  
            
            elif model_name == 'MLP':
                layers = [parameters_list[i] for i in range(2,5) if parameters_list[i] != 0]
                activations = [parameters_list[i] for i in range(5,8) if parameters_list[i-3] != 0]
                if parameters_list[0] == 'pytorch':
                    activation_map = {'ReLU':'ReLU', 'tanh':'Tanh', 'sigmoid':'Sigmoid','linear':'None'}
                    activations = [activation_map[act] for act in activations]
                is_regression = self.screener_type != 'classifier'
                nclasses = getattr(self, 'nclasses', None)
                model = getattr(module,model_name)(engine=parameters_list[0], alpha=np.exp(parameters_list[1]), activations=activations, nneurons=layers, nepochs=parameters_list[8], batch_size=parameters_list[9], opt_config=parameters_list[10], learning_rate=np.exp(parameters_list[11]), nfeatures=self.nfeatures, is_regression=is_regression, nclasses=nclasses, random_seed=42) 

            elif model_name == 'GradientBoostingRegressor':
                model = getattr(module,model_name)(loss=parameters_list[0], n_estimators=parameters_list[1], min_samples_split=parameters_list[2], min_samples_leaf=parameters_list[3], random_state=42)

            elif model_name == 'RandomForestRegressor':
                model = getattr(module,model_name)(n_estimators=parameters_list[0],criterion=parameters_list[1], min_samples_split=parameters_list[2], min_samples_leaf=parameters_list[3], random_state=42, n_jobs=-1)
    
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

            elif model_name in ("XGBRegressor", "XGBClassifier"):
                model = getattr(module,model_name)(n_estimators=parameters_list[0], reg_alpha=np.exp(parameters_list[1]), reg_lambda=np.exp(parameters_list[2]), max_depth=parameters_list[3],learning_rate=np.exp(parameters_list[4]),colsample_bytree=np.exp(parameters_list[5]),subsample=np.exp(parameters_list[6]),gamma=np.exp(parameters_list[7]),min_child_weight=parameters_list[8], device='cuda' if 'cuda' in os.environ.get('CUDA_VISIBLE_DEVICES', '') else 'cpu', random_state=42)

            elif model_name == "LogisticRegression":
                model = getattr(module,model_name)(C=parameters_list[0], fit_intercept=parameters_list[1], solver=parameters_list[2])

            elif model_name == "DecisionTreeClassifier":
                model = getattr(module,model_name)(criterion=parameters_list[0], splitter=parameters_list[1], min_samples_split=parameters_list[2])
            
            elif model_name == "RandomForestClassifier":
                model = getattr(module,model_name)(n_estimators=parameters_list[0], criterion=parameters_list[1])

            elif model_name == "SVC":
                model = getattr(module,model_name)(C=np.exp(parameters_list[0]), kernel=parameters_list[1])
            
            elif model_name == "KNeighborsClassifier":
                if parameters_list[0] > len(self.x_list[key]):
                    n_neighbors = max(1, len(self.x_list[key])//2)
                else:
                    n_neighbors = parameters_list[0]
                model = getattr(module,model_name)(n_neighbors=n_neighbors, weights=parameters_list[1])

            else:
                raise ValueError(f"This model ({model_name}) cannot be used currently. Please refer to documentation. ")
            
            return model
        
        def ga(X_train, y_train, X_test, y_test, model_name, space_final, al):
                    
            start_time_ga = time.time()
                    
            def ga_eval(indi,model_name=model_name):
                _log(model_name+':'+str(indi)+'\t', output_file=self.output_file, to_console=False)
                model = set_hyper_params(parameters_list=indi, model_name=model_name)
                ga_search = single_obj(model=model, x=X_train, y=y_train)
                return ga_search 

            gann = GeneticAlgorithm(evaluate=ga_eval, space=space_final, fitness=('max',), pop_size = 20, crossover_size=2, mutation_size=1, algorithm=al)
            try:
                best_ind_df, best_individual = gann.search(n_generations=self.n_gen, early_stopping=10)                     # set pop_size<30, n_generations*pop_size = no. of times GA runs                      
            except ZeroDivisionError:
                _log(f"\nZeroDivisionError occurred for model: {model_name}\n", output_file=self.output_file)
                return pd.DataFrame()
            _log(f"{model_name}: GeneticAlgorithm - complete", output_file=self.output_file)
            
            all_items = list(gann.fitness_dict.items())
            all_items_df = pd.DataFrame(all_items, columns=['hyperparameters', 'Accuracy_score'])
            all_items_df.to_csv(model_name+'_fitness_dict.csv', index=False)
            
            best_ind_df = best_ind_df.sort_values(by='Fitness_values', ascending=False)
            best_ind_df.to_csv(model_name+'_ga_best.csv',index=False)
            ga_time = (time.time() - start_time_ga)/3600
            
            best_hyper_params = best_ind_df["Best_individual"][0]
            best_ga_model = set_hyper_params(parameters_list=best_hyper_params, model_name=model_name)
            
            ga_accuracy_test = test_hyp(ml_model=best_ga_model, x=X_train, y=y_train, xtest=X_test, ytest=y_test, key=key)
            _log(f"Model: {model_name}\nGA time(hours): {ga_time}\n", output_file=self.output_file)
            return ga_accuracy_test

        try:
            _log(f"\nRunning model no: {tmp_counter}; Name: {model_name}", output_file=output_file)
            model_start_time = time.time()
            space_final = tuple(space_models[model_name])
            _log(f"\n{model_name}\n", output_file=output_file, to_console=False)

            scores_list.append(ga(X_train, y_train, X_test, y_test, model_name=model_name, space_final=space_final, al=3))
            _log("--------------------------------------------------------------------------------", output_file=output_file)
            _log(f"\n------------------------- {model_name} search complete, time taken: {round(time.time()-model_start_time,3)} seconds ------------------------ \n", output_file=output_file, to_console=False)
        except:
            _log(f"\nException occurred for model: {model_name}\n{traceback.format_exc()}\n", output_file=output_file)
        
        return scores_list

    def _represent_smiles(self):
        """
        This function generates various molecular representations (Coulomb matrix, RDKit fingerprints,
        and RDKit descriptors) for a list of molecules represented by SMILES strings.

        Returns
        -------
        list 
            list of pandas DataFrames consisting of various molecular representations
        """        
        from chemml.chem import RDKitFingerprint, CoulombMatrix, RDKDesc, Mordred
        # generate all representation techniques here

        mol_objs_list=[]
        
        i=0
        for i, smi in enumerate(tqdm(self.smiles, desc="Converting SMILES to ChemML Molecule objects")):
            mol = Molecule(smi, 'smiles')
            mol.hydrogens('add')
            try:
                mol.to_xyz('MMFF', maxIters=10000, mmffVariant='MMFF94s')
                mol_objs_list.append(mol)
            except Exception as e:
                _log(f"\nUnable to process SMILES: {smi}; Error: {e}", output_file=self.output_file)
                self.discarded_indices.append(i)
                
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
       
        
        allDescrs = RDKDesc().represent(mol_objs_list).drop(columns='SMILES')
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_allDescrs = scaler.fit_transform(allDescrs)
        scaled_allDescrs = pd.DataFrame(scaled_allDescrs)
        self.x_list["rdkit_descriptors"] = scaled_allDescrs

        mord = Mordred()
        mord_descriptors = mord.represent(mol_objs_list, remove_corr=True).drop(columns='SMILES')
        mord_scaler = StandardScaler()
        mord_descriptors = pd.DataFrame(mord_scaler.fit_transform(mord_descriptors))
        self.x_list['mord_descriptors'] = mord_descriptors


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

        
        scores_combined = pd.concat(scores_list, ignore_index=True)

        if self.screener_type == "regressor":
            self.scores_combined = scores_combined.drop_duplicates(subset='r_squared', keep='last').sort_values(by='RMSE', ascending=True)
        else:
            self.scores_combined = scores_combined.drop_duplicates(subset='Accuracy', keep='last').sort_values(by='Accuracy', ascending=False)

        return self.scores_combined[:n_best]

    def screen_models(self, n_best=10, multi_core=False):
        """
        This function performs genetic algorithm hyperparameter tuning on a list of regression models
        and returns the best performing models.
        

        Parameters
        ----------
        n_best : int, optional
            The number of best models to return as output, by default 10
        
        multi_core : bool, optional
            A boolean indicating whether to screen multi-core models, by default False
            Note that these models are more computationally expensive and take much longer to run
            Highly recommended to run on a HPC node if True

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
        _log(f"\n\n-------------------------Model screening started at {time.ctime()}-------------------------\n\n", output_file=self.output_file, to_console=False)

        y = self.df[self.target].reset_index(drop=True)

        if self.featurization == True:
            self._represent_smiles()
            y = y.drop(index=self.discarded_indices)

        if self.screener_type == 'classifier':
            self.nclasses = y.nunique()
            
        scores_list_overall=[]
        
        if self.screener_type == "classifier":
            from .space import space_models_classifiers as space_models
        else:
            from .space import space_models


        # Splitting model names into single- and multi-core models
        single_core_models = space_models['single_core']
        # Due to SVR and conventional GB scaling poorly with large datasets, we remove it from screening if dataset > 1k samples
        if len(y) > 1e3:
            if self.screener_type == "regressor":
                single_core_models.pop('SVR', None)
            else:
                single_core_models.pop('SVC', None)
        single_core_model_names = list(single_core_models.keys())
        # Multi-core model initialization
        if multi_core:
            multi_core_models = space_models['multi_core']
            if len(y) > 1e3:
                if self.screener_type == "regressor":
                    # Note: XGBRegressor performs better than GradientBoostingRegressor on large datasets, so we retain gradient boosting regression
                    multi_core_models.pop('GradientBoostingRegressor', None)
            multi_core_model_names = list(multi_core_models.keys())

        # write run parameters to output file
        params_msg = (
            "\n-------------------------Run parameters-------------------------\n"
            f"  Featurization: {self.featurization}\n"
            f"  Multi_core: {multi_core}\n"
            f"  Screener_type: {self.screener_type}\n"
            f"  Number of datapoints: {int(len(y))}\n"
        )
        if len(y) > 1000:
            params_msg += "  Note: Dataset > 1000 samples; GradientBoostingRegressor and SVR are excluded from screening due to inefficiency.\n"
        _log(params_msg, output_file=self.output_file, to_console=False)

        for key in self.x_list.keys():
            _log(f"\n------------------------- Screening started for feature set {key} at {time.ctime()} -------------------------\n", output_file=self.output_file, to_console=False)
            start_time = time.time()
            X_train, X_test, y_train, y_test = train_test_split(self.x_list[key], y, test_size=0.1, random_state=42)
            _log("split done!", output_file=self.output_file)
            tmp_counter = 0         
            output_file = self.output_file
            self.nfeatures = X_train.shape[1]

            # Running single-core models in parallel using multiprocessing manager
            with Manager() as manager:
                scores_list = manager.list()
                with Pool() as pool:
                    pool.starmap(self.run_model, [(model_name, tmp_counter + i, output_file, X_train, y_train, X_test, y_test, single_core_models, scores_list, key) for i, model_name in enumerate(single_core_model_names)])
                scores_list_overall.extend(list(scores_list))

                _log("Single-core complete \n", output_file=self.output_file)

            # For multi-core models, run them sequentially to avoid core contention
            if multi_core:
               for multi_core_model_name in multi_core_model_names:
                    tmp_counter += 1
                    scores_list_overall = self.run_model(multi_core_model_name, tmp_counter, output_file, X_train, y_train, X_test, y_test, multi_core_models, scores_list_overall, key)

            _log(f"\n------------------------- Screening complete for feature set {key}, time taken: {round(time.time() - start_time,3)} seconds -------------------------\n", output_file=self.output_file)

        # aggregate scores list
        best_models = self.aggregate_scores(scores_list=scores_list_overall, n_best=n_best)

        return best_models

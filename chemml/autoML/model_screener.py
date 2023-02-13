import warnings
import time
import os
import pandas as pd
import numpy as np
from sklearn.utils import all_estimators
from chemml.utils.utilities import regression_metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, recall_score, precision_score, f1_score

import warnings
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

class ModelScreener(object):

    def __init__(self, df=None, target=None, screener_type="regressor", output_file="scores.txt"):
        """
        The function takes in a dataframe, a target column name or index, a screener type (classifier or
        regressor), and an output file name. 
        
        The function then splits the dataframe into a target column and a feature matrix. 
        
        The function then stores the feature matrix and target column as attributes of the class. 
        
        The function also stores the screener type and output file name as attributes of the class. 
        
        The function then returns nothing.
        
        :param df: The dataframe containing the data
        :param target: The target column name or index
        :param screener_type: This is the type of screener you want to use. It can be either a
        classifier or a regressor, defaults to regressor (optional)
        :param output_file: The name of the file where the scores will be written to, defaults to
        scores.txt (optional)
        """
        
        if isinstance(df, pd.DataFrame):
            self.df = df
            self.target = target
        else:
            print("df must be a DataFrame!")
        
        if isinstance(screener_type, str):
            if screener_type in ["classifier", "regressor", "None"]:
                if screener_type == "None":
                    self.screener_type = None
                else:
                    self.screener_type = screener_type
            else:
                raise ValueError("Parameter screener_type must be 'classifier' or 'regressor' ")
        else:
            raise TypeError("Parameter screener_type must be of type str")
        
        if isinstance(output_file, str):
            split_name = os.path.splitext(output_file)
            # print("split_name: ", split_name)
            if split_name[1] != "":
                self.output_file = output_file
                print("Output file name: ", self.output_file)
            else:
                raise TypeError("'output_file' extension not provided. Parameter 'output_file' must have extension (e.g. output_file = 'scores.txt')")
        else:
            raise TypeError("Parameter 'output_file' must be of type str")
        
        if isinstance(self.target, str):
            print("Target column name given as string")
            print("Obtaining target column...")
            try:
                y_col_index = self.df.columns.get_loc(self.target)
                y = self.df.iloc[:, y_col_index]
                print("y values obtained..")
                print("y.shape: ", y.shape)
                x = self.df.loc[:, self.df.columns != self.target]
                print("x.shape: ", x.shape)
            except:
                print("Column name does not exist!")
        elif isinstance(self.target, int):
            print("Target column name given as integer")
            print("Obtaining target column...")
            try:
                y = self.df.iloc[:, self.target]
                print("y values obtained..")
                print("y.shape: ", y.shape)
                x = self.df.loc[:, self.df.columns != self.df.columns[target]]
                print("x.shape: ", x.shape)
            except:
                print("Column number does not exist!")
        else:
            raise TypeError("Parameter target must be of type str or int")

        self.x = x
        self.y = y

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
            r2 = r2_score(y_test, y_predict)
            # print("model_name: ", model_name)
            # print("r2: ", r2)
            mae = mean_absolute_error(y_test, y_predict)
            time_taken = time.time() - model_start_time
            scores = {"Model": model_name, "MAE": mae, "R2_score": r2, "time(seconds)": time_taken}

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

    def screen_models(self):
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
        
        start_time = time.time()
        scores_df = pd.DataFrame()
        all_models = self.get_all_models_sklearn(filter=self.screener_type)
        print("All possible Models: ", len(all_models))
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.1, random_state=42)
        tmp_counter = 0
        for model in all_models:
            tmp_counter = tmp_counter+1
            model_name = str(model)
            print("Running model no: ", tmp_counter, "; Name: ", model_name)
            try:
                model_start_time = time.time()
                model.fit(X_train, y_train)
                y_predict = model.predict(X_test)
                if scores_df.empty==True:
                    scores = self.obtain_error_metrics(y_test, y_predict, model_name, model_start_time)
                    with open(self.output_file, 'w') as f: 
                        for key, value in scores.items(): 
                            f.write('%s:%s\n' % (key, value))
                        f.write('\n')
                    f.close()
                    scores_df = pd.DataFrame(data=scores, index=[0])
                    # scores_df = pd.concat([scores_df, regression_metrics(y_test, y_predict)], axis=1)
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
                    # print(scores_df)
            except Exception as e: 
                print(e)
        print("\n")
        print("--- %s seconds ---" % (time.time() - start_time))
        self.scores_df = scores_df
        # print(self.scores_df)
        return self.scores_df
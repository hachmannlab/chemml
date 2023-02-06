import warnings
import time
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

    def __init__(self, df=None, target=None, screener_type="regressor"):
        """
        The function takes in a dataframe, a target column name (string or integer), and a screener type
        (string). 
        
        The function then checks if the target column name is a string or an integer. If it's a string,
        it checks if the string exists in the dataframe. If it does, it obtains the target column and
        the rest of the dataframe. If it doesn't, it throws an error. 
        
        If the target column name is an integer, it checks if the integer exists in the dataframe. If it
        does, it obtains the target column and the rest of the dataframe. If it doesn't, it throws an
        error. 
        
        The function then assigns the target column to the variable y and the rest of the dataframe to
        the variable x. 
        
        The function then assigns the variables x and y to the class variables x and y. 
        
        The function then returns nothing.
        
        :param df: The dataframe that you want to screen
        :param target: The target column name or index
        :param screener_type: This is the type of screener you want to use. It can be either a
        classifier or a regressor, defaults to regressor (optional)
        """
        if isinstance(df, pd.DataFrame):
            self.df = df
            self.target = target
        else:
            print("df must be a DataFrame!")
        if isinstance(screener_type, str):
            if screener_type == "None":
                self.screener_type = None
            else:
                self.screener_type = screener_type
        else:
            print("screener_type must be a string!")
        
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
        self.x = x
        self.y = y

    def get_all_models_sklearn(self, filter):
        """
        It returns a list of all the models in sklearn that match the filter
        
        :param filter: This is a string that is used to filter the models. For example, if you want to
        get all the linear models, you can pass 'linear' as the filter
        :return: A list of all the models that are available in sklearn.
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

    def obtain_error_metrics(self, y_test, y_predict, model_name):
        """
        It takes in the true values of the target variable, the predicted values of the target variable,
        and the name of the model as input and returns a dictionary of the error metrics for the model
        
        :param y_test: the actual values of the target variable
        :param y_predict: The predicted values of the target variable
        :param model_name: The name of the model
        :return: a dictionary of scores.
        """
        if self.screener_type == "regressor":
            r2 = r2_score(y_test, y_predict)
            # print("model_name: ", model_name)
            # print("r2: ", r2)
            mae = mean_absolute_error(y_test, y_predict)
            scores = {"Model": model_name, "MAE": mae, "R2_score": r2}

        elif self.screener_type == "classifier":
            accuracy = accuracy_score(y_test, y_predict)
            recall = recall_score(y_test, y_predict, average='macro')
            precision = precision_score(y_test, y_predict, average='macro')
            f1score = f1_score(y_test, y_predict, average='macro')
            scores = {"Model": model_name, "Accuracy": accuracy, "Recall": recall, "Precision": precision, "F1-score": f1score}
        
        else:
            print("Work in progress...\n")
            print("classifier and regressor scores can be separately obtained: ")
            print("""set screener_type to 'regressor' or 'classifier'  """)
            scores = None
        return scores

    def screen_models(self):
        """
        It takes a dataframe, splits it into train and test, and then runs all the models in the
        screener_type list, and returns a dataframe with the error metrics for each model. 
        
        The screener_type list is a list of all the models that we want to run. 
        
        The function is called like this: 
        
        screener_obj = ModelScreener(df, 'regression', 'target_variable')
        screener_obj.screen_models()
        
        The output is a dataframe with the error metrics for each model. 
        """
        start_time = time.time()
        scores_df = pd.DataFrame()
        all_models = self.get_all_models_sklearn(self.screener_type)
        print("All possible Models: ", len(all_models))
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.1, random_state=42)
        tmp_counter = 0
        for model in all_models:
            tmp_counter = tmp_counter+1
            model_name = str(model)
            print("Running model no: ", tmp_counter, "; Name: ", model_name)
            try:
                model.fit(X_train, y_train)
                y_predict = model.predict(X_test)
                if scores_df.empty==True:
                    scores = self.obtain_error_metrics(y_test, y_predict, model_name)
                    scores_df = pd.DataFrame(data=scores, index=[0])
                    # scores_df = pd.concat([scores_df, regression_metrics(y_test, y_predict)], axis=1)
                    # print("scores_df: ", scores_df)
                else:
                    scores = self.obtain_error_metrics(y_test, y_predict, model_name)
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
import warnings
import time
import pandas as pd
import numpy as np
from sklearn.utils import all_estimators
from chemml.utils.utilities import regression_metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings("ignore")



class ModelScreener(object):

    def __init__(self, df=None, target=None, screener_type="regressor"):
        if isinstance(df, pd.DataFrame):
            self.df = df
            self.target = target
        else:
            print("df must be a DataFrame!")
        if screener_type in ["regressor","classifier"]:
            self.screener_type = screener_type
        else:
            print("screener type must be a regressor or classifier")
        
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

    def screen_models(self):
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
                r2 = r2_score(y_test, y_predict)
                # print("model_name: ", model_name)
                # print("r2: ", r2)
                mae = mean_absolute_error(y_test, y_predict)
                # print("mae: ", mae)
                if scores_df.empty==True:
                    scores = {"Model": model_name, "MAE": mae, "R2_score": r2}
                    scores_df = pd.DataFrame(data=scores, index=[0])
                    # scores_df = pd.concat([scores_df, regression_metrics(y_test, y_predict)], axis=1)
                    # print("scores_df: ", scores_df)
                else:
                    scores = {"Model": model_name, "MAE": mae, "R2_score": r2}
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
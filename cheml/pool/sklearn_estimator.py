import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, ClusterMixin, TransformerMixin
from sklearn.model_selection import cross_val_predict

class fake(BaseEstimator, RegressorMixin):
    def __init__(self,alpha=0.02):
        self.alpha = alpha
    def fit(self,X,y):
        self.X = X
        self.y = y
        self.c = 7
        return self
    def predict(self, X):
        return (self.c / 7) * self.y - 2*self.alpha * self.X
    def score(self, X, y=None):
        y_pred = self.predict(X)
        score = np.mean(y-y_pred)
        return score


class regressor(BaseEstimator, RegressorMixin):
    def __init__(self,model = None, evaluator = None):
        self.model = model
        self.evaluator = evaluator
    def fit(self,X,y):
        self.model[0] = self.model[1](self.model[0],X,y)
        return self
    def predict(self, X):
        return self.model[2](self.model[0], X)
    def score(self, X, y):
        y_pred = self.predict(X)
        return self.evaluator(y, y_pred)

class classifier(BaseEstimator, ClassifierMixin):
    def __init__(self,model = None, evaluator = None):
        self.model = model
        self.evaluator = evaluator
    def fit(self,X,y):
        self.model[0] = self.model[1](self.model[0],X,y)
        return self
    def predict(self, X):
        return self.model[2](self.model[0], X)
    def score(self, X, y):
        y_pred = self.predict(X)
        return self.evaluator(y, y_pred)

if __name__ == '__main__':
    X = np.array([1,10, 100, 1000])
    y = np.array([1.04, 11.4, 104, 1041])
    from sklearn.utils.estimator_checks import check_estimator
    f = fake(0.02)
    # check_estimator(f)
    result = cross_val_predict(f, X, y, cv=2)
    print result
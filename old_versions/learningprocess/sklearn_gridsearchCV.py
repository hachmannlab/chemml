# from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
# iris = datasets.load_iris()

class fake(object):
    def __init__(self,p1=1,p2=1):
        self.p1 = p1
        self.p2 = p2
    def fit(self,X,Y):
        print self.p1, self.p2
        self.model = lambda X: sum(X) * self.p1 * self.p2
    def predict(self,X):
        self.model(X)
    def score(self,X,Y):
        return abs(sum(Y) - self.model(X))

model = fake()
parameters = {'p1':(1,2), 'p2':(30,4)}
clf = GridSearchCV(model, parameters)
X = [1,2,3]
Y = [0,0,6]
clf.fit(X,Y)


from sklearn.grid_search import ParameterGrid
param_grid = {'param1': [1,2,3], 'param2' : [4,5], 'param3': [6,7,8,9]}
grid = ParameterGrid(param_grid)
for params in grid:
    print params
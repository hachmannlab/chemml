# from ..chem import CoulombMatrix
import inspect
from sklearn.svm import SVR

# fn = CoulombMatrix.__init__
# print str(fn), inspect.getargspec(fn)[0]

def test_main():
    l=inspect.getmembers(SVR)
    fns = [i[1] for i in l if '__' not in i[0]]
    fns_name = [i[0] for i in l if '__' not in i[0]]
    for fn in fns:
        params = inspect.getargspec(fn)[0]
        print fn
        print params
        print '\n'


def check_functions():
    pass

class C(object):
    def __init__(self):
        self.store = []
        for i in range(3):
            self.i = i
            self.add()
        print self.store
    def add(self):
        self.store.append(self.i+7)
        return 0
from C1 import page1
from C2 import page2
import time

class G(object):
    def __init__(self,CompGraph):
        self.graph = CompGraph
        self.start_time = time.time()
        print 'initialised'

class MainFrame(object):
    def __init__(self,CompGraph):
        self.a = G(CompGraph)
        print self.a.graph

    def fit(self):
        b = page1(self.a)
        c = page2(self.a)
        print 'last graph: ', [i for i in self.a.graph]
        time.sleep(0)
        print time.time() - self.a.start_time

CompGraph = ((0, 1, 'data'), (2, 1, 'data'), (0, 3, 'data'), (3, 4, 'api'), (4, 2, 'fake'))
t = MainFrame(CompGraph)
t.fit()